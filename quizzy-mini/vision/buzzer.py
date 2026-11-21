"""Buzz-in rules: who raised a hand first, inside an explicitly opened window.

Pure logic, no camera or model: feed it per-seat HandRaise judgements with
timestamps and it tells you when someone has validly buzzed in. That makes
every rule below unit-testable with made-up timelines.

Rules
-----
1. Raises only count while a window is open, and only for eligible seats
   (e.g. just the opposing team during a steal).
2. Fresh raise required: a hand that is already up when the window opens
   doesn't count until it comes down and goes back up.
3. A raise is valid once the hand has been seen up for `min_hold_s`.
   Brief keypoint drop-outs shorter than `max_gap_s` don't break a raise.
4. Earliest onset wins. If two valid raises start within `tie_tolerance_s`
   of each other, the one with higher mean confidence wins.
5. The window closes on the first winner; later raises are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .datatypes import BuzzResult, HandRaise


@dataclass(frozen=True)
class BuzzConfig:
    min_hold_s: float = 0.25
    max_gap_s: float = 0.15
    tie_tolerance_s: float = 0.0  # 0 = only same-frame onsets count as a tie


@dataclass
class _SeatTrack:
    run_start: float | None = None  # onset of the current raise, None if hand is down
    last_raised: float | None = None
    conf_sum: float = 0.0
    conf_n: int = 0
    armed: bool = False  # may this seat win in the current window?

    def end_run(self) -> None:
        self.run_start = None
        self.last_raised = None
        self.conf_sum = 0.0
        self.conf_n = 0

    def expire_if_stale(self, t: float, max_gap: float) -> None:
        if self.run_start is not None and self.last_raised is not None and t - self.last_raised > max_gap:
            self.end_run()

    @property
    def mean_conf(self) -> float:
        return self.conf_sum / self.conf_n if self.conf_n else 0.0


class BuzzArbiter:
    def __init__(self, seat_ids: Iterable[int], config: BuzzConfig | None = None):
        self.config = config or BuzzConfig()
        self._tracks = {s: _SeatTrack() for s in seat_ids}
        self._open = False
        self._opened_at = 0.0
        self._eligible: frozenset[int] = frozenset()

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def eligible_seats(self) -> frozenset[int]:
        return self._eligible if self._open else frozenset()

    def open(self, now: float, eligible_seats: Iterable[int] | None = None) -> None:
        eligible = frozenset(self._tracks) if eligible_seats is None else frozenset(eligible_seats)
        unknown = eligible - set(self._tracks)
        if unknown:
            raise ValueError(f"unknown seats: {sorted(unknown)}")
        self._open = True
        self._opened_at = now
        self._eligible = eligible
        for tr in self._tracks.values():
            tr.expire_if_stale(now, self.config.max_gap_s)
            # Rule 2: anyone whose hand is already up must lower it first.
            tr.armed = tr.run_start is None

    def close(self) -> None:
        self._open = False

    def update(self, t: float, observations: Mapping[int, HandRaise]) -> BuzzResult | None:
        """Feed one frame's judgements. Returns a BuzzResult exactly once per window."""
        cfg = self.config
        for seat, hand in observations.items():
            tr = self._tracks.get(seat)
            if tr is None:
                continue
            tr.expire_if_stale(t, cfg.max_gap_s)
            if hand.raised:
                if tr.run_start is None:
                    tr.run_start = t
                tr.last_raised = t
                tr.conf_sum += hand.confidence
                tr.conf_n += 1
            elif tr.run_start is None and self._open:
                tr.armed = True  # hand is down: a new raise will count
        # Seats with no observation this frame keep their state; a stale run
        # is expired the next time we see them (or at the next open()).

        if not self._open:
            return None

        active = [
            (seat, tr)
            for seat, tr in self._tracks.items()
            if seat in self._eligible
            and tr.armed
            and tr.run_start is not None
            # A frame captured just before open() but processed after it must not count.
            and tr.run_start >= self._opened_at
        ]
        candidates = [(s, tr) for s, tr in active if tr.last_raised - tr.run_start >= cfg.min_hold_s]
        if not candidates:
            return None

        earliest = min(tr.run_start for _, tr in candidates)
        tied = [(s, tr) for s, tr in candidates if tr.run_start - earliest <= cfg.tie_tolerance_s]
        # A raise that started within the tie tolerance but hasn't been held long
        # enough yet still deserves its tie-break: wait for it (at most tie_tolerance_s).
        if any(
            (s, tr) not in candidates and tr.run_start - earliest <= cfg.tie_tolerance_s for s, tr in active
        ):
            return None
        seat, tr = max(tied, key=lambda st: (st[1].mean_conf, -st[1].run_start))

        self._open = False
        return BuzzResult(
            seat=seat,
            confidence=tr.mean_conf,
            onset_time=tr.run_start,
            decided_time=t,
            window_opened_time=self._opened_at,
        )
