"""Plain data types shared across the vision module.

Everything here is a frozen dataclass so results can be passed between
threads and to other modules without defensive copying.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO-17 keypoint indices (used by YOLO-pose and most pretrained pose models).
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
NUM_KEYPOINTS = 17


@dataclass(frozen=True)
class Frame:
    """One camera image plus the time it was captured.

    `timestamp` must come from the same clock the VisionService uses
    (time.monotonic by default) so buzz windows and latencies line up.
    """

    image: np.ndarray  # BGR, HxWx3, uint8
    timestamp: float
    index: int

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass(frozen=True)
class PersonPose:
    """One detected person in pixel coordinates."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    keypoints: np.ndarray  # (17, 2) xy pixels
    keypoint_scores: np.ndarray  # (17,) in [0, 1]
    score: float  # person detection confidence

    @property
    def box_height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def box_area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass(frozen=True)
class HandRaise:
    """Per-frame hand-raise judgement for one person."""

    raised: bool
    confidence: float
    side: str | None = None  # "left" / "right" (the person's own side)
    via: str | None = None  # "wrist" or "elbow" (elbow = wrist out of frame / not detected)


@dataclass(frozen=True)
class SeatObservation:
    seat: int
    person: PersonPose
    hand: HandRaise


@dataclass(frozen=True)
class BuzzResult:
    """The answer Module 1 gets back: who buzzed in first, and when."""

    seat: int
    confidence: float
    onset_time: float  # first frame of the winning raise
    decided_time: float  # frame on which the raise became valid
    window_opened_time: float
    source: str = "vision"  # "vision" or "manual" (operator override)

    @property
    def reaction_time_s(self) -> float:
        """How long after the window opened the student started raising."""
        return self.onset_time - self.window_opened_time

    @property
    def detection_delay_s(self) -> float:
        """How long the system took to confirm the raise (hold time + frame gaps)."""
        return self.decided_time - self.onset_time


@dataclass(frozen=True)
class FrameResult:
    """Everything the pipeline concluded about one frame (used for debugging/visualization)."""

    frame: Frame
    people: list[PersonPose]
    seats: dict[int, SeatObservation]
    window_open: bool
    eligible_seats: frozenset[int] = field(default_factory=frozenset)
    buzz: BuzzResult | None = None
    inference_ms: float = 0.0
