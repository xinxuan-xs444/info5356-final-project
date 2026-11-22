"""RobotMotion against the game-logic interface, without needing real hardware.

FakeGame.emit copies exactly what main.create_game(robot=...)'s output() dispatcher does on
feature/game-logic today for look_at_seat / buzz_open / score / reset, plus the proposed
buzz_open -> look_neutral() call and the optional state -> on_state() wiring from
INTEGRATION.md. If main.py's real dispatcher keeps this shape, these tests say RobotMotion
fits it. Everything runs with dry_run=True: no `reachy_mini` import, no real movement.
"""

import json
import os

import pytest

import config
from robot_motion import RobotMotion


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeGame:
    """Mirrors main.create_game(robot=...)'s output() dispatcher, plus the Module-4-proposed
    buzz_open -> look_neutral() ordering and the optional state -> on_state() hook."""

    def __init__(self, robot: RobotMotion):
        self.robot = robot

    def emit(self, event, **data):
        if event == "look_at_seat":
            self.robot.look_at_seat(data["seat"])
        elif event == "buzz_open":
            self.robot.look_neutral()       # proposed: neutral before the window opens
            self.robot.wiggle_antennas()
        elif event == "score":
            for team, score in data["scores"].items():
                self.robot.set_score(team, score)
        elif event == "reset":
            self.robot.reset()
        elif event == "state":
            self.robot.on_state(data["state"])


@pytest.fixture
def robot(tmp_path):
    clock = FakeClock()
    r = RobotMotion(dry_run=True, clock=clock, idle_interval=5.0, event_log=str(tmp_path / "events.jsonl"))
    r._clock_for_test = clock  # stash for tests that need to advance it
    yield r
    r.close()


def events(robot) -> list[dict]:
    path = robot._event_log
    if path is None or not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_contract_methods_exist_and_dont_raise(robot):
    game = FakeGame(robot)
    game.emit("reset")
    game.emit("look_at_seat", seat=2)
    game.emit("buzz_open", seats=(1, 2, 3, 4))
    game.emit("score", scores={"red": 1, "blue": 0})
    game.emit("state", state="SCORE")
    game.emit("reset")
    # dry_run never touches hardware, so just getting here without raising is the contract.


def test_look_at_seat_uses_configured_angles(robot):
    robot.look_at_seat(2)
    last_look = [e for e in events(robot) if e["event"] == "look"][-1]
    assert (last_look["yaw"], last_look["pitch"]) == pytest.approx(config.SEAT_HEAD_ANGLES[2])


def test_look_at_unknown_seat_falls_back_to_neutral(robot):
    robot.look_at_seat(99)
    last_look = [e for e in events(robot) if e["event"] == "look"][-1]
    assert (last_look["yaw"], last_look["pitch"]) == pytest.approx(config.HEAD_NEUTRAL_ANGLE)


def test_look_neutral_resets_current_seat(robot):
    robot.look_at_seat(1)
    robot.look_neutral()
    last_look = [e for e in events(robot) if e["event"] == "look"][-1]
    assert (last_look["yaw"], last_look["pitch"]) == pytest.approx(config.HEAD_NEUTRAL_ANGLE)
    assert robot._current_seat is None


def test_head_angles_are_clamped_to_safe_range(robot):
    robot.seat_angles[1] = (999.0, -999.0)
    robot.look_at_seat(1)
    last_look = [e for e in events(robot) if e["event"] == "look"][-1]
    yaw_lo, yaw_hi = config.HEAD_SAFE_YAW_RANGE
    pitch_lo, pitch_hi = config.HEAD_SAFE_PITCH_RANGE
    assert yaw_lo <= last_look["yaw"] <= yaw_hi
    assert pitch_lo <= last_look["pitch"] <= pitch_hi


def test_antenna_angles_are_clamped_to_safe_range(robot):
    robot.set_antenna_raw("red", 9999.0)
    last_antenna = [e for e in events(robot) if e["event"] == "antenna"][-1]
    lo, hi = config.ANTENNA_SAFE_RANGE
    assert lo <= last_antenna["angle"] <= hi


def test_set_score_moves_the_right_teams_antenna(robot):
    robot.set_score("red", 3)
    last_antenna = [e for e in events(robot) if e["event"] == "antenna"][-1]
    assert last_antenna["team"] == "red"
    assert last_antenna["angle"] == pytest.approx(config.ANTENNA_SCORE_ANGLES[3])


def test_set_score_clamps_to_the_configured_angle_table(robot):
    # WIN_SCORE ends the game, but a hint/mini-question could still nudge past it defensively.
    robot.set_score("blue", config.WIN_SCORE + 5)
    last_antenna = [e for e in events(robot) if e["event"] == "antenna"][-1]
    assert last_antenna["angle"] == pytest.approx(config.ANTENNA_SCORE_ANGLES[-1])


def test_reset_zeroes_scores_and_returns_head_to_neutral(robot):
    robot.look_at_seat(3)
    robot.set_score("red", 4)
    robot.reset()
    assert robot._scores == {"red": 0, "blue": 0}
    assert robot._current_seat is None
    last_look = [e for e in events(robot) if e["event"] == "look"][-1]
    assert (last_look["yaw"], last_look["pitch"]) == pytest.approx(config.HEAD_NEUTRAL_ANGLE)


def test_on_state_score_triggers_a_celebration_wiggle(robot):
    before = len([e for e in events(robot) if e["event"] == "antenna"])
    robot.on_state("SCORE")
    after = len([e for e in events(robot) if e["event"] == "antenna"])
    assert after > before


def test_on_state_ignores_unrelated_states(robot):
    before = events(robot)
    robot.on_state("ASK")
    assert events(robot) == before


def test_idle_animation_fires_after_the_configured_interval(robot):
    clock = robot._clock_for_test
    before = len(events(robot))
    clock.advance(6.0)   # idle_interval=5.0 in the fixture
    import time

    time.sleep(1.3)   # the idle loop polls on a ~1s real-time tick; give it a chance to wake
    after = len(events(robot))
    assert after > before


def test_dry_run_never_imports_the_reachy_sdk(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fail_on_reachy(name, *a, **kw):
        if name == "reachy_mini":
            raise AssertionError("dry_run must not import reachy_mini")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fail_on_reachy)
    r = RobotMotion(dry_run=True)
    r.look_at_seat(1)
    r.close()
