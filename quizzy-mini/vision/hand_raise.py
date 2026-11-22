"""Decides from a single pose whether a hand is raised.

Rule: a hand counts as raised when the wrist is above its own shoulder by at
least `margin` x shoulder width. Using shoulder width as the unit makes the
rule independent of how far the student sits from the camera.

Fallback: when the wrist isn't detected (typically because the raised hand
leaves the top of the frame), an elbow clearly above the shoulder also counts,
with slightly reduced confidence.

Timing (how long the hand must stay up) is not decided here; that is the
BuzzArbiter's job, so this function stays a pure per-frame check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datatypes import (
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    HandRaise,
    PersonPose,
)


@dataclass(frozen=True)
class HandRaiseRule:
    min_keypoint_score: float = 0.3
    # Wrist must be this many shoulder-widths above the shoulder.
    margin: float = 0.25
    # Used when only one shoulder is visible (fraction of box width).
    fallback_shoulder_width_ratio: float = 0.35
    # Elbow fallback: elbow this many shoulder-widths above the shoulder (None disables).
    elbow_margin: float | None = 0.15
    elbow_confidence_scale: float = 0.8

    def evaluate(self, pose: PersonPose) -> HandRaise:
        kp, sc = pose.keypoints, pose.keypoint_scores
        unit = self._shoulder_width(pose)
        ok = lambda i: sc[i] >= self.min_keypoint_score  # noqa: E731

        best: HandRaise | None = None
        for side, shoulder, elbow, wrist in (
            ("left", LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
            ("right", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        ):
            if not ok(shoulder):
                continue
            # Image y grows downward, so "above" means a smaller y.
            cand = None
            if ok(wrist):
                if kp[shoulder, 1] - kp[wrist, 1] > self.margin * unit:
                    cand = HandRaise(True, float(min(sc[shoulder], sc[wrist])), side, "wrist")
            elif self.elbow_margin is not None and ok(elbow):
                if kp[shoulder, 1] - kp[elbow, 1] > self.elbow_margin * unit:
                    conf = float(min(sc[shoulder], sc[elbow])) * self.elbow_confidence_scale
                    cand = HandRaise(True, conf, side, "elbow")
            if cand and (best is None or cand.confidence > best.confidence):
                best = cand

        return best if best is not None else HandRaise(raised=False, confidence=0.0)

    def _shoulder_width(self, pose: PersonPose) -> float:
        kp, sc = pose.keypoints, pose.keypoint_scores
        if sc[LEFT_SHOULDER] >= self.min_keypoint_score and sc[RIGHT_SHOULDER] >= self.min_keypoint_score:
            width = float(np.linalg.norm(kp[LEFT_SHOULDER] - kp[RIGHT_SHOULDER]))
            if width > 1.0:
                return width
        return self.fallback_shoulder_width_ratio * (pose.bbox[2] - pose.bbox[0])
