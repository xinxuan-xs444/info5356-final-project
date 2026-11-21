from vision.hand_raise import HandRaiseRule
from vision.datatypes import RIGHT_WRIST

from .helpers import make_pose

rule = HandRaiseRule()


def test_hands_down_is_not_raised():
    assert not rule.evaluate(make_pose(200)).raised


def test_either_hand_up_is_raised():
    for side in ("left", "right"):
        h = rule.evaluate(make_pose(200, raised=side))
        assert h.raised and h.side == side
        assert h.confidence > 0.5


def test_wrist_barely_above_shoulder_is_not_enough():
    p = make_pose(200)
    p.keypoints[RIGHT_WRIST] = (150, 300 - 10)  # 0.1 shoulder widths above
    assert not rule.evaluate(p).raised


def test_low_confidence_keypoints_are_ignored():
    assert not rule.evaluate(make_pose(200, raised="right", conf=0.1)).raised


def test_rule_is_scale_invariant():
    far = make_pose(200, width=30, raised="left")
    near = make_pose(200, width=200, raised="left")
    assert rule.evaluate(far).raised and rule.evaluate(near).raised


def _wrist_out_of_frame(pose, elbow_y):
    from vision.datatypes import RIGHT_ELBOW

    pose.keypoint_scores[RIGHT_WRIST] = 0.05  # hand left the top of the image
    pose.keypoints[RIGHT_ELBOW] = (150, elbow_y)
    return pose


def test_elbow_above_shoulder_counts_when_wrist_is_missing():
    h = rule.evaluate(_wrist_out_of_frame(make_pose(200), elbow_y=250))  # 0.5 widths up
    assert h.raised and h.via == "elbow" and h.confidence < 0.9


def test_elbow_at_shoulder_height_does_not_count():
    assert not rule.evaluate(_wrist_out_of_frame(make_pose(200), elbow_y=298)).raised


def test_visible_low_wrist_overrides_high_elbow():
    from vision.datatypes import RIGHT_ELBOW

    p = make_pose(200)  # wrist visible and down
    p.keypoints[RIGHT_ELBOW] = (150, 250)
    assert not rule.evaluate(p).raised


def test_elbow_fallback_can_be_disabled():
    strict = HandRaiseRule(elbow_margin=None)
    assert not strict.evaluate(_wrist_out_of_frame(make_pose(200), elbow_y=250)).raised
