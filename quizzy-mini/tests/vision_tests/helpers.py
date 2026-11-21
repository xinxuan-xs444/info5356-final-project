"""Synthetic poses so tests don't need a camera or a model."""

from __future__ import annotations

import numpy as np

from vision.datatypes import (
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    NUM_KEYPOINTS,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    PersonPose,
)


def make_pose(
    cx: float,
    *,
    shoulder_y: float = 300.0,
    width: float = 100.0,
    raised: str | None = None,  # None, "left", "right"
    conf: float = 0.9,
    box_height: float = 400.0,
) -> PersonPose:
    kp = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    sc = np.full(NUM_KEYPOINTS, conf, dtype=np.float32)
    # Person faces the camera: their left shoulder is on the image right.
    kp[LEFT_SHOULDER] = (cx + width / 2, shoulder_y)
    kp[RIGHT_SHOULDER] = (cx - width / 2, shoulder_y)
    kp[NOSE] = (cx, shoulder_y - 0.6 * width)
    kp[LEFT_WRIST] = (cx + width / 2, shoulder_y + width)
    kp[RIGHT_WRIST] = (cx - width / 2, shoulder_y + width)
    if raised == "left":
        kp[LEFT_WRIST] = (cx + width / 2, shoulder_y - width)
    elif raised == "right":
        kp[RIGHT_WRIST] = (cx - width / 2, shoulder_y - width)
    top = shoulder_y - 1.5 * width
    bbox = (cx - width, top, cx + width, top + box_height)
    return PersonPose(bbox=bbox, keypoints=kp, keypoint_scores=sc, score=conf)


class ScriptedEstimator:
    """Returns a pre-set list of poses for each call, in order (last one repeats)."""

    def __init__(self, script: list[list[PersonPose]]):
        self.script = script
        self.calls = 0

    def estimate(self, image) -> list[PersonPose]:
        out = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return out


class FakeClock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt
