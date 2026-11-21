import pytest

from vision.seats import SeatMap

from .helpers import make_pose

W, H = 800, 600


def test_uniform_zones():
    m = SeatMap.uniform()
    assert m.boundaries == [0.25, 0.5, 0.75]
    assert [m.seat_for_x(x) for x in (0.1, 0.3, 0.6, 0.9)] == [1, 2, 3, 4]


def test_reversed_seat_order():
    m = SeatMap.uniform(seat_ids=[4, 3, 2, 1])
    assert m.seat_for_x(0.1) == 4 and m.seat_for_x(0.9) == 1


def test_assign_one_person_per_zone():
    people = [make_pose(x) for x in (100, 300, 500, 700)]
    got = SeatMap.uniform().assign(people, W, H)
    assert {s: p.bbox[0] + 100 for s, p in got.items()} == {1: 100, 2: 300, 3: 500, 4: 700}


def test_assign_prefers_largest_person_in_zone():
    small = make_pose(120, width=40, box_height=200)
    big = make_pose(90, width=100, box_height=400)
    got = SeatMap.uniform().assign([small, big], W, H)
    assert got[1] is big


def test_assign_ignores_tiny_background_people():
    far = make_pose(300, width=20, box_height=50)
    assert SeatMap.uniform().assign([far], W, H) == {}


def test_anchor_uses_shoulders_not_box_when_arm_is_up():
    # Shoulder midpoint at x=195 (seat 1); a raised arm can't push it into seat 2.
    p = make_pose(195, raised="left")
    assert SeatMap.uniform().assign([p], W, H).keys() == {1}


def test_calibrate_from_centers():
    m = SeatMap.from_person_centers([0.8, 0.1, 0.3, 0.5])
    assert m.boundaries == pytest.approx([0.2, 0.4, 0.65])
    assert m.seat_for_x(0.62) == 3 and m.seat_for_x(0.7) == 4


def test_calibrate_needs_one_person_per_seat():
    with pytest.raises(ValueError):
        SeatMap.from_person_centers([0.2, 0.5, 0.8])


def test_invalid_boundaries_rejected():
    with pytest.raises(ValueError):
        SeatMap(boundaries=[0.5, 0.25, 0.75])


def test_save_load_roundtrip(tmp_path):
    m = SeatMap(boundaries=[0.2, 0.45, 0.7], seat_ids=[4, 3, 2, 1], min_box_height=0.2)
    m.save(tmp_path / "seats.json")
    assert SeatMap.load(tmp_path / "seats.json") == m
