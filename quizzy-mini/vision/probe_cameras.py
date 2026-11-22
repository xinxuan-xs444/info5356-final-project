"""Find which camera index is which: python -m vision.probe_cameras

For each OpenCV camera index, grabs ~2 s of frames and reports resolution,
brightness, whether the image changes over time (a live camera does; a static
placeholder such as an idle virtual camera doesn't), and how many people the
pose model sees. Prints numbers only; no images are saved.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time

import cv2
import numpy as np


def avfoundation_names() -> list[str]:
    """Device names in AVFoundation order (macOS, needs ffmpeg). Order usually matches OpenCV's."""
    if not shutil.which("ffmpeg"):
        return []
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    ).stderr
    names, in_video = [], False
    for line in out.splitlines():
        if "video devices" in line:
            in_video = True
        elif "audio devices" in line:
            break
        elif in_video and "] [" in line:
            names.append(line.split("] ", 2)[-1].split("] ", 1)[-1])
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-index", type=int, default=3)
    ap.add_argument("--weights", default="yolo11n-pose.pt")
    args = ap.parse_args()

    names = avfoundation_names()
    if names:
        # OpenCV's numbering does NOT reliably follow this order (on a MacBook Air with OBS
        # installed, ffmpeg said [0] OBS / [1] MacBook camera, but OpenCV's 0 was the real camera).
        # Judge by the "live" and "people" columns below instead.
        print("Cameras on this machine:", ", ".join(names))

    from .pose import YoloPoseEstimator

    loose = YoloPoseEstimator(args.weights, conf=0.1)

    for idx in range(args.max_index + 1):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"camera {idx}: cannot open")
            continue
        frames = []
        t_end = time.monotonic() + 2.0
        while time.monotonic() < t_end:
            ok, img = cap.read()
            if ok:
                frames.append(img)
        cap.release()
        if not frames:
            print(f"camera {idx}: opened but no frames")
            continue
        last = frames[-1]
        change = float(np.abs(last.astype(np.int16) - frames[len(frames) // 2].astype(np.int16)).mean())
        people = loose.estimate(last)
        confs = ", ".join(f"{p.score:.2f}" for p in people) or "-"
        print(
            f"camera {idx}: {last.shape[1]}x{last.shape[0]}, {len(frames)} frames in 2 s, "
            f"brightness {last.mean():.0f}, frame-to-frame change {change:.1f} "
            f"({'live' if change > 0.5 else 'STATIC - placeholder?'}), "
            f"people at conf>=0.1: {len(people)} [{confs}]"
        )


if __name__ == "__main__":
    main()
