"""End-to-end through VisionService with a fake camera, fake model and fake clock."""

import itertools
import json
import time

import numpy as np
import pytest

from vision import ListSource, SeatMap, VisionService
from vision.datatypes import Frame

from .helpers import FakeClock, ScriptedEstimator, make_pose

IMG = np.zeros((600, 800, 3), dtype=np.uint8)
X = {1: 100, 2: 300, 3: 500, 4: 700}  # seat centres in pixels (uniform zones)


def scene(raised: set[int]):
    return [make_pose(X[s], raised="right" if s in raised else None) for s in (1, 2, 3, 4)]


def feed(svc, clock, scenes, dt=1 / 30):
    """Process one frame per scene, advancing the fake clock; return the first buzz."""
    for sc in scenes:
        svc.estimator.script = [sc]
        res = svc.process_frame(Frame(IMG, clock(), 0))
        clock.advance(dt)
        if res.buzz:
            return res
    return None


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def svc(clock, tmp_path):
    return VisionService(
        ListSource([]), ScriptedEstimator([[]]), SeatMap.uniform(), clock=clock,
        event_log=tmp_path / "events.jsonl",
    )


def test_seats_and_hands_reported_per_frame(svc, clock):
    svc.estimator.script = [scene({3})]
    res = svc.process_frame(Frame(IMG, clock(), 0))
    assert set(res.seats) == {1, 2, 3, 4}
    assert [s for s, o in res.seats.items() if o.hand.raised] == [3]
    assert not res.window_open


def test_buzz_resolves_window_and_calls_back(svc, clock):
    got = []
    svc.open_buzz_window(on_result=got.append)
    res = feed(svc, clock, [scene(set())] * 5 + [scene({2})] * 15)
    assert res.buzz.seat == 2
    assert got == [res.buzz]
    assert svc.wait_for_buzz(0) == res.buzz
    assert not svc.window_open


def test_steal_window_only_accepts_eligible_seats(svc, clock):
    svc.open_buzz_window(eligible_seats=[3, 4])
    res = feed(svc, clock, [scene(set())] * 3 + [scene({1})] * 15 + [scene({1, 4})] * 15)
    assert res.buzz.seat == 4


def test_fifth_seat_can_buzz(clock):
    centers = [80, 240, 400, 560, 720]
    seat_map = SeatMap.from_person_centers([x / IMG.shape[1] for x in centers], seat_ids=range(1, 6))
    svc = VisionService(ListSource([]), ScriptedEstimator([[]]), seat_map, clock=clock)
    svc.open_buzz([5])
    hands_down = [make_pose(x) for x in centers]
    fifth_raised = hands_down[:-1] + [make_pose(centers[-1], raised="right")]
    res = feed(svc, clock, [hands_down] * 5 + [fifth_raised] * 15)
    assert res.buzz.seat == 5


def test_timeout_returns_none(svc, clock):
    got = []
    svc.open_buzz_window(timeout_s=1.0, on_result=got.append)
    assert feed(svc, clock, [scene(set())] * 40) is None
    assert got == [None] and not svc.window_open


def test_manual_override(svc, clock):
    svc.open_buzz_window()
    assert svc.manual_buzz(4)
    r = svc.wait_for_buzz(0)
    assert r.seat == 4 and r.source == "manual"
    assert not svc.manual_buzz(1)  # nothing open any more


def test_event_log_has_no_images(svc, clock, tmp_path):
    svc.open_buzz_window()
    feed(svc, clock, [scene(set())] * 3 + [scene({1})] * 15)
    lines = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [l["event"] for l in lines] == ["window_open", "window_close"]
    assert lines[1]["result"]["seat"] == 1


def test_calibrate_from_latest_frame(svc, clock):
    svc.estimator.script = [[make_pose(x) for x in (80, 250, 520, 760)]]
    svc.process_frame(Frame(IMG, clock(), 0))
    m = svc.calibrate_seats_from_latest()
    assert m.boundaries == pytest.approx([165 / 800, 385 / 800, 640 / 800])
    assert m.seat_for_x(160 / 800) == 1 and m.seat_for_x(170 / 800) == 2


def test_threaded_blocking_api_with_real_clock():
    """Module 1's typical call: background thread + run_buzz_window()."""
    frames_down = [scene(set())] * 5
    frames_up = [scene({1})] * 1000
    est = ScriptedEstimator(frames_down + frames_up)
    src = ListSource(itertools.repeat(IMG))  # endless frames

    class Paced:
        def read(self):
            time.sleep(1 / 60)
            return src.read()

        def close(self):
            pass

    with VisionService(Paced(), est, SeatMap.uniform()) as svc:
        deadline = time.monotonic() + 3
        while svc.latest is None:
            assert time.monotonic() < deadline, "vision thread produced no frames"
            time.sleep(0.005)
        est.calls = 0  # restart the script so the raise begins after the window opens
        result = svc.run_buzz_window(timeout_s=3.0)
    assert result is not None and result.seat == 1
    assert 0 <= result.detection_delay_s < 0.5
