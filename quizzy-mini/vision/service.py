"""VisionService: the one object Module 1 talks to.

Pipeline per frame:  FrameSource -> PoseEstimator -> SeatMap -> HandRaiseRule -> BuzzArbiter

Module 1 usage (blocking):

    vision = VisionService(WebcamSource(0), YoloPoseEstimator(), SeatMap.load("seats.json"))
    vision.start()                      # background thread
    result = vision.run_buzz_window(eligible_seats=[1, 2, 3, 4], timeout_s=10)
    if result is None: ...              # nobody buzzed in
    else: result.seat                   # 1..4

or non-blocking: open_buzz_window(..., on_result=callback), then wait_for_buzz().
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .buzzer import BuzzArbiter, BuzzConfig
from .hand_raise import HandRaiseRule
from .pose import PoseEstimator
from .seats import SeatMap, anchor_x
from .sources import FrameSource
from .datatypes import BuzzResult, Frame, FrameResult, SeatObservation

log = logging.getLogger(__name__)

BuzzCallback = Callable[[BuzzResult | None], None]


@dataclass
class _Window:
    opened_at: float
    deadline: float | None
    eligible: frozenset[int]
    on_result: BuzzCallback | None = None
    done: threading.Event = field(default_factory=threading.Event)
    result: BuzzResult | None = None


class VisionService:
    def __init__(
        self,
        source: FrameSource,
        estimator: PoseEstimator,
        seat_map: SeatMap | None = None,
        *,
        rule: HandRaiseRule | None = None,
        buzz_config: BuzzConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        event_log: str | Path | None = None,
    ):
        self.source = source
        self.estimator = estimator
        self.seat_map = seat_map or SeatMap.uniform()
        self.rule = rule or HandRaiseRule()
        self._clock = clock
        self._arbiter = BuzzArbiter(self.seat_map.seat_ids, buzz_config)
        self._lock = threading.RLock()
        self._window: _Window | None = None
        self._latest: FrameResult | None = None
        self._listeners: list[Callable[[FrameResult], None]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._event_log = Path(event_log) if event_log else None

    # ------------------------------------------------------------------ API for Module 1
    def open_buzz_window(
        self,
        eligible_seats: Iterable[int] | None = None,
        timeout_s: float | None = None,
        on_result: BuzzCallback | None = None,
    ) -> None:
        """Start accepting hand raises. Call only once the robot head is back in its neutral pose."""
        with self._lock:
            if self._window and not self._window.done.is_set():
                self._finish(None, reason="superseded")
            now = self._clock()
            self._arbiter.open(now, eligible_seats)
            self._window = _Window(
                opened_at=now,
                deadline=None if timeout_s is None else now + timeout_s,
                eligible=self._arbiter.eligible_seats,
                on_result=on_result,
            )
            self._log_event("window_open", eligible=sorted(self._window.eligible), timeout_s=timeout_s)

    def wait_for_buzz(self, timeout: float | None = None) -> BuzzResult | None:
        """Block until the current window resolves. None = timeout / no buzz / closed."""
        window = self._window
        if window is None:
            return None
        window.done.wait(timeout)
        return window.result

    def run_buzz_window(
        self, eligible_seats: Iterable[int] | None = None, timeout_s: float = 10.0
    ) -> BuzzResult | None:
        """Open a window, block until someone buzzes or it times out, return the winner."""
        self.open_buzz_window(eligible_seats, timeout_s)
        result = self.wait_for_buzz(timeout_s + 1.0)
        self.close_buzz_window()  # no-op if already resolved
        return result

    def close_buzz_window(self) -> None:
        with self._lock:
            if self._window and not self._window.done.is_set():
                self._finish(None, reason="closed")

    def manual_buzz(self, seat: int) -> bool:
        """Operator override: award the open window to `seat`. Returns False if no window is open."""
        with self._lock:
            w = self._window
            if w is None or w.done.is_set():
                return False
            if seat not in self.seat_map.seat_ids:
                raise ValueError(f"unknown seat {seat}")
            now = self._clock()
            self._finish(
                BuzzResult(seat=seat, confidence=1.0, onset_time=now, decided_time=now,
                           window_opened_time=w.opened_at, source="manual"),
                reason="manual",
            )
            return True

    @property
    def window_open(self) -> bool:
        with self._lock:
            return self._window is not None and not self._window.done.is_set()

    @property
    def latest(self) -> FrameResult | None:
        return self._latest

    def add_listener(self, fn: Callable[[FrameResult], None]) -> None:
        """Called with every FrameResult (from the processing thread). Keep it fast."""
        self._listeners.append(fn)

    def calibrate_seats_from_latest(self) -> SeatMap:
        """Rebuild seat zones from the people in the most recent frame (needs exactly one per seat)."""
        res = self._latest
        if res is None:
            raise RuntimeError("no frame processed yet")
        f = res.frame
        people = [p for p in res.people if p.box_height >= self.seat_map.min_box_height * f.height]
        new_map = SeatMap.from_person_centers(
            [anchor_x(p) / f.width for p in people],
            seat_ids=self.seat_map.seat_ids,
            min_box_height=self.seat_map.min_box_height,
        )
        with self._lock:
            self.seat_map = new_map
        return new_map

    # ------------------------------------------------------------------ processing
    def step(self) -> FrameResult | None:
        """Read and process one frame synchronously (used by the thread and the demo)."""
        frame = self.source.read()
        if frame is None:
            self._check_timeout()
            return None
        return self.process_frame(frame)

    def process_frame(self, frame: Frame) -> FrameResult:
        t0 = time.perf_counter()
        people = self.estimator.estimate(frame.image)
        inference_ms = (time.perf_counter() - t0) * 1000

        with self._lock:
            assigned = self.seat_map.assign(people, frame.width, frame.height)
            seats = {s: SeatObservation(s, p, self.rule.evaluate(p)) for s, p in assigned.items()}
            buzz = self._arbiter.update(frame.timestamp, {s: o.hand for s, o in seats.items()})
            if buzz is not None and self._window and not self._window.done.is_set():
                self._finish(buzz, reason="buzz")
            else:
                self._check_timeout()
            result = FrameResult(
                frame=frame,
                people=people,
                seats=seats,
                window_open=self.window_open,
                eligible_seats=self._window.eligible if self.window_open else frozenset(),
                buzz=buzz,
                inference_ms=inference_ms,
            )
            self._latest = result

        for fn in self._listeners:
            try:
                fn(result)
            except Exception:  # a broken listener must not stop the game
                log.exception("frame listener failed")
        return result

    # ------------------------------------------------------------------ background thread
    def start(self) -> "VisionService":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.close_buzz_window()
        self.source.close()

    def __enter__(self) -> "VisionService":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self.step() is None:
                    time.sleep(0.01)
            except Exception:
                log.exception("vision step failed; continuing")
                time.sleep(0.05)

    # ------------------------------------------------------------------ internals (call with lock held)
    def _check_timeout(self) -> None:
        with self._lock:
            w = self._window
            if w and not w.done.is_set() and w.deadline is not None and self._clock() >= w.deadline:
                self._finish(None, reason="timeout")

    def _finish(self, result: BuzzResult | None, reason: str) -> None:
        w = self._window
        assert w is not None
        self._arbiter.close()
        w.result = result
        w.done.set()
        self._log_event("window_close", reason=reason, result=asdict(result) if result else None)
        if w.on_result:
            try:
                w.on_result(result)
            except Exception:
                log.exception("on_result callback failed")

    def _log_event(self, kind: str, **data) -> None:
        log.info("%s %s", kind, data)
        if self._event_log:
            # Timestamps and seat IDs only. Never images (participants are minors).
            with self._event_log.open("a") as fh:
                fh.write(json.dumps({"event": kind, "t": self._clock(), "wall": time.time(), **data}) + "\n")
