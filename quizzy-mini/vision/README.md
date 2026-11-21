# vision — Module 3: Computer Vision & Player Tracking

Detects which of four seated students raises a hand first during a buzz-in window
and reports the seat ID (1–4) to Module 1. No face or name recognition: Module 1
maps seat → player name/team.

## Architecture

```
FrameSource ──► PoseEstimator ──► SeatMap ──► HandRaiseRule ──► BuzzArbiter
(webcam/file/    (YOLO-pose,       (fixed x-zones  (wrist above      (window, eligibility,
 Reachy camera)   multi-person)     → seat 1–4)     shoulder?)        fresh raise, hold, ties)
                         └──────────── VisionService (thread + API for Module 1) ───────────┘
```

| File | Role | Depends on hardware/model? |
|---|---|---|
| `sources.py` | `FrameSource` protocol + `OpenCVSource` (webcam/video), `ReachyMiniSource`, `ListSource` | swap freely |
| `pose.py` | `PoseEstimator` protocol + `YoloPoseEstimator` (Ultralytics, pretrained COCO-17) | model |
| `seats.py` | `SeatMap`: fixed vertical zones, calibration, save/load JSON | pure |
| `hand_raise.py` | `HandRaiseRule`: per-frame raised/not + confidence | pure |
| `buzzer.py` | `BuzzArbiter`: all buzz-in rules | pure |
| `service.py` | `VisionService`: runs the pipeline, public API | — |
| `datatypes.py` | shared data shapes (`Frame`, `PersonPose`, `BuzzResult` …) | — |
| `viz.py`, `demo.py`, `probe_cameras.py` | debug overlay, interactive demo, camera finder | — |

Swapping the laptop webcam for the robot camera only means passing a different
`FrameSource`; nothing else changes.

## Rules (all configurable)

- **Raised hand:** wrist ≥ 0.25 × shoulder-width above its shoulder (`HandRaiseRule`).
- **Valid buzz:** hand held up ≥ 0.25 s; keypoint drop-outs < 0.15 s are tolerated (`BuzzConfig`).
- **Fresh raise:** a hand already up when the window opens must come down first.
- **Earliest onset wins;** same-frame tie → higher mean confidence (`tie_tolerance_s` widens "same frame").
- **Eligibility:** `open_buzz_window(eligible_seats=[3, 4])` for steals; other seats are ignored.
- **One winner per window;** the window closes on the first valid buzz, on timeout, or on `close_buzz_window()`.

## Contract with Module 1

```python
from vision import VisionService, WebcamSource, YoloPoseEstimator, SeatMap

vision = VisionService(WebcamSource(0), YoloPoseEstimator(), SeatMap.load("seats.json"))
vision.start()                                   # background thread

# Blocking: returns BuzzResult or None (nobody buzzed before timeout)
result = vision.run_buzz_window(eligible_seats=[1, 2, 3, 4], timeout_s=10)
if result:
    result.seat              # 1..4  -> Module 1 looks up name/team
    result.reaction_time_s   # window open -> student started raising
    result.detection_delay_s # raise start -> confirmed (≈ hold time)

# Non-blocking alternative
vision.open_buzz_window([3, 4], timeout_s=10, on_result=lambda r: ...)
vision.wait_for_buzz()      # or rely on the callback
vision.manual_buzz(2)       # operator override (result.source == "manual")
vision.stop()
```

**Head pose:** the camera is in Reachy Mini's head, and seat zones assume the head is
in its neutral pose. Module 1 should only call `open_buzz_window` after Module 4 reports
the head is back at neutral, and must not move the head while a window is open.

**Steals:** vision does not know about teams. Module 1 passes the opposing team's seats as `eligible_seats`.

## Running it

From the repository root (Python 3.10+):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-vision.txt
cd quizzy-mini
../.venv/bin/python -m pytest tests/vision_tests -q   # 43 tests, no camera/model needed
../.venv/bin/python -m vision.demo                    # webcam 0
../.venv/bin/python -m vision.demo --source clip.mp4 --loop
```

Always run as `python -m vision.<tool>` from `quizzy-mini/`, not `python vision/demo.py`.

> **Known issue on `main`:** the starter folder `quizzy-mini/operator/` has the same name as
> Python's built-in `operator` module, so *any* Python run from `quizzy-mini/` crashes with
> `ImportError: cannot import name ... from 'operator'`. `feature/game-logic` already fixes this
> by renaming it to `operator_ui/`. Until that branch is merged, test vision on a copy without
> the `operator/` folder (or rename it locally without committing).

**Which camera number?** Run `python -m vision.probe_cameras` (from `quizzy-mini/`). It reports
which indices are live and how many people each one sees. Virtual cameras (OBS, Camo, Zoom)
take up indices too, and OpenCV's numbering doesn't follow the device names that macOS or
ffmpeg list. On the first dev MacBook Air, camera 0 was the built-in webcam and camera 1
was OBS's frozen placeholder. On macOS, the app that launches Python (Terminal, Claude,
VS Code) needs camera access in System Settings → Privacy & Security → Camera.

The demo prints a `[see]` line whenever detections change and a `[heartbeat]` every 5 s
(fps, resolution, brightness), so you can debug from the terminal alone.

Demo keys: `SPACE` open window (all seats) · `r`/`b` window for seats 1–2 / 3–4 ·
`m` then `1`–`4` manual override · `c` calibrate zones from 4 visible people · `s` save
zones to `seats.json` · `q` quit.

Calibrate once per setup: sit four people in their seats, press `c`, then `s`.
If fixed zones prove unreliable (e.g. students lean into each other's zone), the
fallback plan is coloured markers in front of each seat. That would be a new
`SeatMap`-like component, and nothing else would change.

## Privacy

Participants are minors. Nothing saves frames or video. `--log events.jsonl` (or
`VisionService(event_log=...)`) writes only timestamps, seat IDs and confidences.

## Robot camera (not yet tested on hardware)

`ReachyMiniSource` uses the SDK's `mini.media.get_frame()` (BGR numpy array, or None).
Install with `pip install -e ".[robot]"`. If Module 4 already holds a `ReachyMini`
instance, pass it in (`ReachyMiniSource(mini)`) so the two modules share one connection.

## Not done yet

- Real-camera accuracy/latency evaluation with labelled clips of hand raises.
- Testing on the Reachy Mini camera (field of view: can it see all four students?).
- Tuning thresholds for seated middle-schoolers.
