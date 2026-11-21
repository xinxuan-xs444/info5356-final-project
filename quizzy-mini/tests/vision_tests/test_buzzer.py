"""Buzz-in rules, driven by hand-written timelines at 30 fps."""

import pytest

from vision.buzzer import BuzzArbiter, BuzzConfig
from vision.datatypes import HandRaise

DT = 1 / 30
UP = HandRaise(True, 0.9, "right")
DOWN = HandRaise(False, 0.0)


def up(conf):
    return HandRaise(True, conf, "right")


def run(arb, t0, frames):
    """frames: list of {seat: HandRaise}; returns (first result, time it arrived)."""
    t = t0
    for obs in frames:
        r = arb.update(t, obs)
        if r:
            return r
        t += DT
    return None


def hold(seat_states, seconds):
    return [dict(seat_states)] * int(round(seconds / DT))


@pytest.fixture
def arb():
    return BuzzArbiter([1, 2, 3, 4], BuzzConfig(min_hold_s=0.25, max_gap_s=0.15))


def test_nothing_counts_while_closed(arb):
    assert run(arb, 0.0, hold({1: UP}, 1.0)) is None


def test_first_valid_raise_wins(arb):
    arb.open(0.0)
    frames = hold({1: DOWN, 2: DOWN}, 0.5) + hold({1: DOWN, 2: UP}, 0.5)
    r = run(arb, 0.0, frames)
    assert r.seat == 2
    assert r.onset_time == pytest.approx(0.5, abs=DT)
    assert r.detection_delay_s == pytest.approx(0.25, abs=DT + 1e-9)
    assert r.reaction_time_s == pytest.approx(0.5, abs=DT)


def test_brief_twitch_is_not_a_buzz(arb):
    arb.open(0.0)
    frames = hold({1: UP}, 0.1) + hold({1: DOWN}, 0.5)
    assert run(arb, 0.0, frames) is None


def test_hand_already_up_at_open_must_come_down_first(arb):
    run(arb, -1.0, hold({1: UP}, 1.0))  # seat 1 raised before the window
    arb.open(0.0)
    # Keeps it up: never counts. Seat 3 raises later and wins.
    frames = hold({1: UP, 3: DOWN}, 0.3) + hold({1: UP, 3: UP}, 0.5)
    r = run(arb, 0.0, frames)
    assert r.seat == 3


def test_lowering_then_raising_again_counts(arb):
    run(arb, -1.0, hold({1: UP}, 1.0))
    arb.open(0.0)
    frames = hold({1: DOWN}, 0.3) + hold({1: UP}, 0.5)
    assert run(arb, 0.0, frames).seat == 1


def test_short_keypoint_dropout_does_not_reset_the_raise(arb):
    arb.open(0.0)
    frames = hold({2: UP}, 0.15) + hold({2: DOWN}, 0.1) + hold({2: UP}, 0.15)
    r = run(arb, 0.0, frames)
    assert r.seat == 2 and r.onset_time == pytest.approx(0.0)


def test_long_dropout_resets_the_raise(arb):
    arb.open(0.0)
    frames = hold({2: UP}, 0.2) + hold({2: DOWN}, 0.3) + hold({2: UP}, 0.4)
    r = run(arb, 0.0, frames)
    assert r.onset_time == pytest.approx(0.5, abs=DT)


def test_missing_frames_for_a_seat_do_not_reset_it(arb):
    arb.open(0.0)
    frames = hold({2: UP}, 0.15) + hold({}, 0.1) + hold({2: UP}, 0.15)
    assert run(arb, 0.0, frames).onset_time == pytest.approx(0.0)


def test_earliest_onset_wins_even_if_both_valid(arb):
    arb.open(0.0)
    frames = hold({1: DOWN, 4: UP}, 2 * DT) + hold({1: UP, 4: UP}, 0.5)
    assert run(arb, 0.0, frames).seat == 4


def test_same_frame_tie_goes_to_higher_confidence(arb):
    arb.open(0.0)
    frames = hold({1: up(0.6), 2: up(0.95), 3: up(0.7)}, 0.5)
    assert run(arb, 0.0, frames).seat == 2


def test_tie_tolerance_widens_the_tie():
    arb = BuzzArbiter([1, 2], BuzzConfig(tie_tolerance_s=0.1))
    arb.open(0.0)
    frames = hold({1: up(0.6), 2: DOWN}, DT) + hold({1: up(0.6), 2: up(0.9)}, 0.5)
    assert run(arb, 0.0, frames).seat == 2


def test_ineligible_seats_cannot_win(arb):
    arb.open(0.0, eligible_seats=[3, 4])  # steal: only Blue team
    frames = hold({1: UP, 2: UP, 3: DOWN}, 0.4) + hold({1: UP, 2: UP, 3: UP}, 0.4)
    assert run(arb, 0.0, frames).seat == 3


def test_window_resolves_only_once(arb):
    arb.open(0.0)
    assert run(arb, 0.0, hold({1: UP}, 0.5)).seat == 1
    assert not arb.is_open
    assert run(arb, 0.5, hold({1: UP, 2: UP}, 1.0)) is None


def test_reopen_for_next_round_requires_fresh_raise(arb):
    arb.open(0.0)
    run(arb, 0.0, hold({1: UP}, 0.5))  # seat 1 wins...
    run(arb, 0.3, hold({1: UP}, 0.7))  # ...and keeps the hand up until the next round
    arb.open(1.0)
    assert run(arb, 1.0, hold({1: UP}, 1.0)) is None


def test_stale_frame_from_before_open_does_not_count(arb):
    arb.open(10.0)
    # A frame captured at 9.98 (before open) processed late, then hand stays up.
    assert arb.update(9.98, {1: UP}) is None
    assert run(arb, 10.0, hold({1: UP}, 0.5)) is None


def test_unknown_eligible_seat_rejected(arb):
    with pytest.raises(ValueError):
        arb.open(0.0, eligible_seats=[5])
