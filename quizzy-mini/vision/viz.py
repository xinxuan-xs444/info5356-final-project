"""Debug overlay: seat zones, people, arm skeletons, raised hands, window state, winner."""

from __future__ import annotations

import cv2
import numpy as np

from .seats import SeatMap
from .datatypes import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    BuzzResult,
    FrameResult,
)

SEAT_COLORS = {1: (60, 60, 230), 2: (60, 160, 255), 3: (230, 120, 40), 4: (200, 80, 160)}  # BGR
GREY, GREEN, WHITE, BLACK = (150, 150, 150), (60, 220, 60), (255, 255, 255), (0, 0, 0)
ARMS = [
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
]


def _label(img, text, org, color, scale=0.6, thickness=2):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 4), BLACK, -1)
    cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw(
    result: FrameResult,
    seat_map: SeatMap,
    *,
    last_winner: BuzzResult | None = None,
    fps: float | None = None,
    min_kp_score: float = 0.3,
) -> np.ndarray:
    img = result.frame.image.copy()
    h, w = img.shape[:2]

    # Seat zones
    for b in seat_map.boundaries:
        x = int(b * w)
        cv2.line(img, (x, 0), (x, h), GREY, 1, cv2.LINE_AA)
    for seat in seat_map.seat_ids:
        lo, hi = seat_map.zone_bounds(seat)
        eligible = result.window_open and seat in result.eligible_seats
        color = SEAT_COLORS.get(seat, WHITE)
        _label(img, f"Seat {seat}" + (" *" if eligible else ""), (int(lo * w) + 6, h - 12), color)

    # Unassigned people (thin grey boxes)
    assigned = {id(o.person) for o in result.seats.values()}
    for p in result.people:
        if id(p) not in assigned:
            x1, y1, x2, y2 = map(int, p.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), GREY, 1)

    # Assigned people
    for seat, obs in result.seats.items():
        p, color = obs.person, SEAT_COLORS.get(seat, WHITE)
        x1, y1, x2, y2 = map(int, p.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if obs.hand.raised else 2)
        for a, b in ARMS:
            if p.keypoint_scores[a] >= min_kp_score and p.keypoint_scores[b] >= min_kp_score:
                pa = tuple(int(v) for v in p.keypoints[a])
                pb = tuple(int(v) for v in p.keypoints[b])
                cv2.line(img, pa, pb, GREEN if obs.hand.raised else color, 3, cv2.LINE_AA)
        tag = f"S{seat}" + (f" HAND UP {obs.hand.confidence:.2f}" if obs.hand.raised else "")
        _label(img, tag, (x1, max(y1, 24)), GREEN if obs.hand.raised else color)

    # Status banner
    if result.window_open:
        status, scolor = "BUZZ WINDOW OPEN", GREEN
    else:
        status, scolor = "window closed", GREY
    info = f"{status} | {result.inference_ms:.0f} ms/frame" + (f" | {fps:.1f} fps" if fps else "")
    _label(img, info, (8, 26), scolor)
    if last_winner:
        _label(
            img,
            f"WINNER: Seat {last_winner.seat} ({last_winner.source}) "
            f"reaction {last_winner.reaction_time_s:.2f}s, confirm {last_winner.detection_delay_s * 1000:.0f}ms",
            (8, 56), (0, 255, 255),
        )
        if last_winner.seat in seat_map.seat_ids:
            lo, hi = seat_map.zone_bounds(last_winner.seat)
            cv2.rectangle(img, (int(lo * w) + 2, 2), (int(hi * w) - 2, h - 2), (0, 255, 255), 4)
    return img
