"""Maps detected people to Seats 1-4 using fixed vertical zones in the image.

Seats never move during a game, so instead of tracking identities we split
the frame into left-to-right zones once (uniformly, or calibrated from where
the four students actually sit) and assign each person by the horizontal
position of their shoulders.

This relies on the robot's head (and so its camera) being in the same neutral
pose whenever a buzz window is open. See README "Contract with Module 1".
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .datatypes import LEFT_SHOULDER, RIGHT_SHOULDER, PersonPose

DEFAULT_SEAT_IDS = (1, 2, 3, 4)


def anchor_x(pose: PersonPose, min_score: float = 0.3) -> float:
    """Horizontal pixel position used to place a person in a zone.

    Shoulder midpoint is steadier than the box centre, which shifts sideways
    when an arm goes up.
    """
    sc = pose.keypoint_scores
    if sc[LEFT_SHOULDER] >= min_score and sc[RIGHT_SHOULDER] >= min_score:
        return float((pose.keypoints[LEFT_SHOULDER, 0] + pose.keypoints[RIGHT_SHOULDER, 0]) / 2)
    return float((pose.bbox[0] + pose.bbox[2]) / 2)


@dataclass
class SeatMap:
    # Normalised x positions (0..1) of the lines between zones, ascending.
    boundaries: list[float]
    # Seat ID for each zone from image-left to image-right. The robot's camera
    # looks at the students, so image-left is the students' right; flip the
    # order here if your seat numbering runs the other way.
    seat_ids: list[int] = field(default_factory=lambda: list(DEFAULT_SEAT_IDS))
    # Ignore people whose box is shorter than this fraction of the frame
    # (e.g. someone walking far behind the students).
    min_box_height: float = 0.15

    def __post_init__(self) -> None:
        if len(self.boundaries) != len(self.seat_ids) - 1:
            raise ValueError("need exactly len(seat_ids) - 1 boundaries")
        if any(not 0.0 < b < 1.0 for b in self.boundaries):
            raise ValueError("boundaries must be strictly between 0 and 1")
        if list(self.boundaries) != sorted(self.boundaries):
            raise ValueError("boundaries must be ascending")
        if len(set(self.seat_ids)) != len(self.seat_ids):
            raise ValueError("seat_ids must be unique")

    @classmethod
    def uniform(cls, seat_ids: Sequence[int] = DEFAULT_SEAT_IDS, **kw) -> "SeatMap":
        n = len(seat_ids)
        return cls(boundaries=[i / n for i in range(1, n)], seat_ids=list(seat_ids), **kw)

    @classmethod
    def from_person_centers(
        cls, centers_norm: Sequence[float], seat_ids: Sequence[int] = DEFAULT_SEAT_IDS, **kw
    ) -> "SeatMap":
        """Calibrate: put the zone lines halfway between where the students sit."""
        if len(centers_norm) != len(seat_ids):
            raise ValueError(f"expected {len(seat_ids)} people, saw {len(centers_norm)}")
        xs = sorted(centers_norm)
        return cls(
            boundaries=[(a + b) / 2 for a, b in zip(xs, xs[1:])], seat_ids=list(seat_ids), **kw
        )

    def seat_for_x(self, x_norm: float) -> int:
        return self.seat_ids[bisect_right(self.boundaries, x_norm)]

    def zone_bounds(self, seat: int) -> tuple[float, float]:
        i = self.seat_ids.index(seat)
        edges = [0.0, *self.boundaries, 1.0]
        return edges[i], edges[i + 1]

    def assign(self, people: Sequence[PersonPose], frame_w: int, frame_h: int) -> dict[int, PersonPose]:
        """At most one person per seat: the largest (nearest) one in that zone."""
        chosen: dict[int, PersonPose] = {}
        for p in people:
            if p.box_height < self.min_box_height * frame_h:
                continue
            seat = self.seat_for_x(anchor_x(p) / frame_w)
            if seat not in chosen or p.box_area > chosen[seat].box_area:
                chosen[seat] = p
        return chosen

    # --- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"boundaries": self.boundaries, "seat_ids": self.seat_ids, "min_box_height": self.min_box_height},
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "SeatMap":
        d = json.loads(Path(path).read_text())
        return cls(
            boundaries=list(d["boundaries"]),
            seat_ids=list(d["seat_ids"]),
            min_box_height=float(d.get("min_box_height", 0.15)),
        )
