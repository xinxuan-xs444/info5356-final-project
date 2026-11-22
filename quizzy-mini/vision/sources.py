"""Camera inputs. Swap these without touching the detection code.

Every source implements the FrameSource protocol: `read()` returns the next
Frame (or None when there is nothing more), `close()` releases the device.
Timestamps use time.monotonic() so they match VisionService's clock.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, Protocol

import cv2
import numpy as np

from .datatypes import Frame


class FrameSource(Protocol):
    def read(self) -> Frame | None: ...
    def close(self) -> None: ...


class OpenCVSource:
    """Webcam (int index) or video file (path) via cv2.VideoCapture."""

    def __init__(
        self,
        target: int | str,
        *,
        realtime: bool = True,
        loop: bool = False,
        width: int | None = None,
        height: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._cap = cv2.VideoCapture(target)
        if not self._cap.isOpened():
            hint = (
                " (on macOS, allow camera access for the app running Python: "
                "System Settings > Privacy & Security > Camera, then restart that app)"
                if isinstance(target, int)
                else ""
            )
            raise RuntimeError(f"could not open video source {target!r}{hint}")
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._is_file = isinstance(target, str)
        self._realtime = realtime
        self._loop = loop
        self._clock = clock
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._period = 1.0 / fps if self._is_file and fps and fps > 0 else 0.0
        self._next_due: float | None = None
        self._index = 0

    def read(self) -> Frame | None:
        ok, img = self._cap.read()
        if not ok and self._is_file and self._loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = self._cap.read()
        if not ok:
            return None
        if self._is_file and self._realtime and self._period:
            # Pace file playback like a live camera so buzz timings are realistic.
            now = self._clock()
            self._next_due = now if self._next_due is None else max(self._next_due + self._period, now - self._period)
            if self._next_due > now:
                time.sleep(self._next_due - now)
        frame = Frame(image=img, timestamp=self._clock(), index=self._index)
        self._index += 1
        return frame

    def close(self) -> None:
        self._cap.release()


def WebcamSource(index: int = 0, **kw) -> OpenCVSource:
    return OpenCVSource(index, **kw)


def VideoFileSource(path: str, **kw) -> OpenCVSource:
    return OpenCVSource(str(path), **kw)


class ReachyMiniSource:
    """Frames from the Reachy Mini head camera via the official SDK.

    Untested against hardware so far. Based on the SDK docs:
    `ReachyMini().media.get_frame()` returns a BGR uint8 array or None.
    Requires `pip install reachy-mini`.
    """

    def __init__(self, mini=None, *, clock: Callable[[], float] = time.monotonic, **reachy_kwargs):
        if mini is None:
            from reachy_mini import ReachyMini  # imported lazily: optional dependency

            mini = ReachyMini(**reachy_kwargs)
            mini.__enter__()
            self._owns = True
        else:
            self._owns = False  # shared with Module 4; don't close it here
        self._mini = mini
        self._clock = clock
        self._index = 0

    def read(self) -> Frame | None:
        img = self._mini.media.get_frame()
        if img is None:
            return None
        frame = Frame(image=np.ascontiguousarray(img), timestamp=self._clock(), index=self._index)
        self._index += 1
        return frame

    def close(self) -> None:
        if self._owns:
            self._mini.__exit__(None, None, None)


class ListSource:
    """Replays in-memory images (tests, offline evaluation)."""

    def __init__(self, frames: Iterable[np.ndarray | Frame], clock: Callable[[], float] = time.monotonic):
        self._it = iter(frames)
        self._clock = clock
        self._index = 0

    def read(self) -> Frame | None:
        item = next(self._it, None)
        if item is None:
            return None
        frame = item if isinstance(item, Frame) else Frame(image=item, timestamp=self._clock(), index=self._index)
        self._index += 1
        return frame

    def close(self) -> None:
        pass
