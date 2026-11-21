"""Pretrained multi-person pose estimation.

PoseEstimator is the seam: anything that turns a BGR image into a list of
PersonPose (COCO-17 keypoints) can be plugged in. The default uses
Ultralytics YOLO-pose, which detects every person in one pass.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .datatypes import NUM_KEYPOINTS, PersonPose


class PoseEstimator(Protocol):
    def estimate(self, image: np.ndarray) -> list[PersonPose]: ...


class YoloPoseEstimator:
    """Ultralytics YOLO-pose. Weights download automatically on first use (~6 MB for 'n')."""

    def __init__(
        self,
        weights: str = "yolo11n-pose.pt",
        *,
        conf: float = 0.4,
        imgsz: int = 640,
        device: str | None = None,  # None = auto; "mps" for Apple GPU, "cpu", "cuda:0"
    ):
        from ultralytics import YOLO  # optional dependency, imported lazily

        self._model = YOLO(weights)
        self._conf = conf
        self._imgsz = imgsz
        self._device = device

    def estimate(self, image: np.ndarray) -> list[PersonPose]:
        res = self._model.predict(
            image, conf=self._conf, imgsz=self._imgsz, device=self._device, verbose=False
        )[0]
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        kxy = res.keypoints.xy.cpu().numpy()
        kconf = (
            res.keypoints.conf.cpu().numpy()
            if res.keypoints.conf is not None
            else np.ones(kxy.shape[:2], dtype=np.float32)
        )
        people = []
        for box, s, xy, kc in zip(boxes, scores, kxy, kconf):
            if xy.shape[0] != NUM_KEYPOINTS:
                continue
            people.append(
                PersonPose(
                    bbox=tuple(float(v) for v in box),
                    keypoints=xy.astype(np.float32),
                    keypoint_scores=kc.astype(np.float32),
                    score=float(s),
                )
            )
        return people
