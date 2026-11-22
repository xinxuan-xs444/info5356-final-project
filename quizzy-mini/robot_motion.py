"""Robot motion for Quizzy Mini: gaze, antenna/scoreboard needles, idle and
celebratory animation, and safe-movement limits for the Reachy Mini.

Module 1 usage (the names main.create_game(robot=...) calls on feature/game-logic):

    robot = RobotMotion()                 # connects to the real robot
    # or, headless for dev/tests/CI:
    robot = RobotMotion(dry_run=True)
    game = create_game(robot=robot)
    ...
    robot.on_state(state_name)            # NOT called by main.py yet; call it on every "state"
                                           # event to drive idle/celebratory/teaching animation
    robot.close()

See INTEGRATION.md for exactly which main.py events call which method, and for the one
open item flagged by Module 3 (returning the head to neutral before vision's buzz window
opens).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Callable

import config

log = logging.getLogger(__name__)

_CELEBRATE_STATES = {"SCORE"}
_TEACH_STATES = {"TEACH", "HINT_THINK"}
_END_STATES = {"END"}


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return max(lo, min(hi, value))


class RobotMotion:
    """Owns the Reachy Mini connection. Safe to construct with no hardware (dry_run=True)."""

    def __init__(
        self,
        mini=None,
        *,
        dry_run: bool = False,
        seat_angles: dict[int, tuple[float, float]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        event_log: str | None = None,
        idle_interval: float = config.IDLE_ANIMATION_INTERVAL,
        **reachy_kwargs,
    ):
        """`mini` lets Module 3 and Module 4 share one ReachyMini connection (vision reads the
        head camera, this module drives the head/antennas) -- pass the same object to both
        instead of each opening its own. `dry_run=True` skips the SDK entirely (unit tests, a
        laptop with no robot attached); a failed real connection also falls back to dry-run
        rather than crashing the game.
        """
        self.seat_angles = dict(seat_angles or config.SEAT_HEAD_ANGLES)
        self._clock = clock
        self._lock = threading.RLock()
        self._event_log = event_log
        self._scores = {"red": 0, "blue": 0}
        self._current_seat: int | None = None
        self._last_activity = clock()
        self._stop = threading.Event()
        self._idle_interval = idle_interval

        self.dry_run = dry_run
        self._owns = False
        self._mini = None
        if not self.dry_run:
            if mini is not None:
                self._mini = mini
            else:
                try:
                    from reachy_mini import ReachyMini  # imported lazily: optional dependency

                    self._mini = ReachyMini(**reachy_kwargs)
                    self._mini.__enter__()
                    self._owns = True
                except Exception:
                    log.exception("could not connect to Reachy Mini; continuing in dry-run mode")
                    self.dry_run = True

        self._idle_thread = threading.Thread(target=self._idle_loop, name="robot-idle", daemon=True)
        self._idle_thread.start()

    # ------------------------------------------------------------------ main.py interface
    # Exactly the methods main.create_game(robot=...) calls today. Don't rename without
    # updating main.py's output() dispatcher -- see INTEGRATION.md.
    def look_at_seat(self, seat: int) -> None:
        """Game event `look_at_seat`: called on CALL, when a player buzzes in."""
        angles = self.seat_angles.get(seat, config.HEAD_NEUTRAL_ANGLE)
        self._mark_activity()
        self._run_safely(lambda: self._move_head(*angles))
        with self._lock:
            self._current_seat = seat

    def look_neutral(self) -> bool:
        """Return the head to the pose vision assumes a buzz window will start from.

        Resolves vision/INTEGRATION.md's open item ("Agree on the exact call with the Module 4
        owner"): main.py's buzz_open branch should call this *before* robot.wiggle_antennas() /
        vision.open_buzz(), so the camera is already forward-facing when the window opens.
        See INTEGRATION.md step 3 for the exact main.py diff.
        """
        self._mark_activity()
        arrived = self._move_head(*config.HEAD_NEUTRAL_ANGLE)
        if arrived:
            with self._lock:
                self._current_seat = None
        return arrived

    def wiggle_antennas(self) -> None:
        """Game event `buzz_open`: a quick "go!" wiggle on both antennas."""
        self._mark_activity()
        self._run_safely(self._do_wiggle)

    def set_score(self, team: str, score: int) -> None:
        """Game event `score`: move that team's antenna needle to the score position."""
        self._mark_activity()
        with self._lock:
            self._scores[team] = score
        self._run_safely(lambda: self._set_antenna_angle(team, self._score_angle(score)))

    def reset(self) -> None:
        """Game event `reset`: new game -- zero both needles, head to neutral."""
        self._mark_activity()
        with self._lock:
            self._scores = {"red": 0, "blue": 0}
            self._current_seat = None
        self._run_safely(self._do_reset)

    # ------------------------------------------------------------------ extra: "state" events
    def on_state(self, state: str) -> None:
        """Not called by main.py today. GameController.transition() emits a `state` event on
        every FSM change (state=state.name, e.g. "SCORE", "TEACH", "END") -- wire this into
        main.py's output() if/when celebratory and teaching-mode motion is wanted (see
        INTEGRATION.md step 2, optional). Safe either way: it only animates, never changes
        game state."""
        self._mark_activity()
        if state in _CELEBRATE_STATES:
            self._run_safely(self._do_celebrate)
        elif state in _TEACH_STATES:
            self._run_safely(self._do_teach_wiggle)
        elif state in _END_STATES:
            self._run_safely(self._do_victory)

    # ------------------------------------------------------------------ calibration hooks
    # calibration.py only -- raw moves that skip seat_angles / score lookups but still go
    # through the same safety clamp and dry-run guard as the main.py-facing methods above.
    def move_head_raw(self, yaw: float, pitch: float) -> None:
        self._run_safely(lambda: self._move_head(yaw, pitch))

    def set_antenna_raw(self, team: str, angle: float) -> None:
        self._run_safely(lambda: self._set_antenna_angle(team, angle))

    def score_angle_for(self, score: int) -> float:
        return self._score_angle(score)

    # ------------------------------------------------------------------ idle animation
    def _idle_loop(self) -> None:
        while not self._stop.wait(1.0):
            with self._lock:
                idle_for = self._clock() - self._last_activity
            if idle_for >= self._idle_interval:
                self._run_safely(self._do_idle)
                self._mark_activity()  # don't replay the idle animation every second

    def _mark_activity(self) -> None:
        with self._lock:
            self._last_activity = self._clock()

    # ------------------------------------------------------------------ animations
    def _do_wiggle(self) -> None:
        self._wiggle_both(config.ANTENNA_WIGGLE_DEGREES, config.ANTENNA_WIGGLE_CYCLES)

    def _do_celebrate(self) -> None:
        self._wiggle_both(config.ANTENNA_WIGGLE_DEGREES * 1.5, config.ANTENNA_WIGGLE_CYCLES + 1)

    def _do_teach_wiggle(self) -> None:
        yaw, pitch = config.HEAD_NEUTRAL_ANGLE
        self._move_head(yaw, pitch + 8.0)
        time.sleep(0.2)
        self._move_head(yaw, pitch)

    def _do_victory(self) -> None:
        for _ in range(3):
            self._wiggle_both(config.ANTENNA_WIGGLE_DEGREES, 1)

    def _do_idle(self) -> None:
        yaw, pitch = config.HEAD_NEUTRAL_ANGLE
        self._move_head(yaw + 6.0, pitch)
        self._move_head(yaw - 6.0, pitch)
        self._move_head(yaw, pitch)

    def _do_reset(self) -> None:
        self._move_head(*config.HEAD_NEUTRAL_ANGLE)
        for team in ("red", "blue"):
            self._set_antenna_angle(team, self._score_angle(0))

    def _wiggle_both(self, degrees: float, cycles: int) -> None:
        for team in ("red", "blue"):
            base = self._score_angle(self._scores.get(team, 0))
            for _ in range(cycles):
                self._set_antenna_angle(team, base + degrees)
                self._set_antenna_angle(team, base - degrees)
            self._set_antenna_angle(team, base)

    # ------------------------------------------------------------------ safe hardware primitives
    def _score_angle(self, score: int) -> float:
        angles = config.ANTENNA_SCORE_ANGLES
        index = max(0, min(len(angles) - 1, score))
        return angles[index]

    def _move_head(self, yaw: float, pitch: float) -> bool:
        yaw = _clamp(yaw, config.HEAD_SAFE_YAW_RANGE)
        pitch = _clamp(pitch, config.HEAD_SAFE_PITCH_RANGE)
        try:
            self._log_event("look", yaw=yaw, pitch=pitch)
            if not self.dry_run:
                from reachy_mini.utils import create_head_pose

                self._mini.goto_target(head=create_head_pose(yaw=yaw, pitch=pitch, degrees=True),
                                       duration=config.HEAD_MOVE_DURATION, body_yaw=None)
        except Exception:
            log.exception("head move failed; continuing")
            return False
        return True

    def _set_antenna_angle(self, team: str, angle: float) -> None:
        angle = _clamp(angle, config.ANTENNA_SAFE_RANGE)
        self._log_event("antenna", team=team, angle=angle)
        if self.dry_run:
            return
        try:
            _, current_angles = self._mini.get_current_joint_positions()
            angles = list(current_angles)
            angles[0 if team == "red" else 1] = math.radians(angle)
            self._mini.goto_target(antennas=angles, duration=0.3, body_yaw=None)
        except Exception:
            log.exception("antenna move failed (team=%s); continuing", team)

    def _run_safely(self, fn: Callable[[], None]) -> None:
        """Hardware/animation calls must never raise into the game loop."""
        try:
            fn()
        except Exception:
            log.exception("robot motion step failed; continuing")

    def _log_event(self, kind: str, **data) -> None:
        log.info("%s %s", kind, data)
        if self._event_log:
            with Path(self._event_log).open("a") as fh:
                fh.write(json.dumps({"event": kind, "t": self._clock(), "wall": time.time(), **data}) + "\n")

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self._stop.set()
        self._idle_thread.join(timeout=2)
        if self._owns and self._mini is not None:
            try:
                self._mini.__exit__(None, None, None)
            except Exception:
                log.exception("error closing Reachy Mini connection")

    def __enter__(self) -> "RobotMotion":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
