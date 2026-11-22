"""Interactive debug demo.

    python -m vision.demo                      # webcam 0
    python -m vision.demo --source clip.mp4    # video file
    python -m vision.demo --source reachy      # Reachy Mini camera (needs reachy-mini installed)

Keys:
    SPACE  open a buzz window for all seats (10 s)
    r / b  open a steal-style window for seats 1-2 only / seats 3-4 only
    m 1-4  manual override: award the open window to that seat
    c      calibrate seat zones from the 4 people currently visible
    s      save seat zones to --seats file
    q/ESC  quit

Nothing is recorded to disk except the optional --log JSONL of window events (no images).
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2

from .seats import SeatMap
from .service import VisionService
from .sources import OpenCVSource, ReachyMiniSource
from .viz import draw

HELP = "SPACE all | r seats 1-2 | b seats 3-4 | m+1..4 manual | c calibrate | s save | q quit"


def _make_source(spec: str, loop: bool):
    if spec == "reachy":
        return ReachyMiniSource()
    if spec.isdigit():
        return OpenCVSource(int(spec), width=1280, height=720)
    return OpenCVSource(spec, realtime=True, loop=loop)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index, video path, or 'reachy'")
    ap.add_argument("--seats", default="seats.json", help="seat-zone file (loaded if it exists)")
    ap.add_argument("--weights", default="yolo11n-pose.pt")
    ap.add_argument("--device", default=None, help="cpu / mps / cuda:0 (default: auto)")
    ap.add_argument("--timeout", type=float, default=10.0, help="buzz window length in seconds")
    ap.add_argument("--loop", action="store_true", help="loop a video file")
    ap.add_argument("--log", default=None, help="append window events to this JSONL file")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from .pose import YoloPoseEstimator

    try:
        seat_map = SeatMap.load(args.seats)
        print(f"loaded seat zones from {args.seats}")
    except FileNotFoundError:
        seat_map = SeatMap.uniform()
        print("using uniform seat zones (press c with 4 people visible to calibrate)")

    svc = VisionService(
        _make_source(args.source, args.loop),
        YoloPoseEstimator(args.weights, device=args.device),
        seat_map,
        event_log=args.log,
    )
    last_winner = None
    manual_mode = False
    last_state = None
    n_frames, t_beat = 0, time.perf_counter()
    fps, t_prev = None, time.perf_counter()

    def on_result(r):
        nonlocal last_winner
        last_winner = r
        print("BUZZ:", r if r else "no buzz (timeout/closed)")

    print(HELP)
    try:
        while True:
            res = svc.step()
            if res is None:
                if args.source not in ("reachy",) and not args.source.isdigit():
                    break  # end of video file
                continue
            # Print what the pipeline sees whenever it changes (useful without the window).
            state = (
                len(res.people),
                tuple(sorted((s, o.hand.raised, o.hand.via) for s, o in res.seats.items())),
            )
            if state != last_state:
                seats_txt = ", ".join(
                    f"S{s}:" + (f"UP({o.hand.via},{o.hand.confidence:.2f})" if o.hand.raised else "down")
                    for s, o in sorted(res.seats.items())
                ) or "none in a seat zone"
                print(f"[see] {len(res.people)} people | {seats_txt} | window {'OPEN' if res.window_open else 'closed'}")
                last_state = state

            n_frames += 1
            if time.perf_counter() - t_beat >= 5.0:
                f = res.frame
                print(f"[heartbeat] {n_frames} frames, {fps or 0:.1f} fps, {f.width}x{f.height}, "
                      f"brightness {f.image.mean():.0f}/255, {len(res.people)} people, "
                      f"{res.inference_ms:.0f} ms inference")
                t_beat = time.perf_counter()

            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6)
            fps = inst if fps is None else 0.9 * fps + 0.1 * inst
            t_prev = now

            img = draw(res, svc.seat_map, last_winner=last_winner, fps=fps)
            cv2.putText(img, HELP + ("  [MANUAL: press 1-4]" if manual_mode else ""),
                        (8, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Quizzy vision debug", img)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                last_winner = None
                svc.open_buzz_window(timeout_s=args.timeout, on_result=on_result)
            elif key == ord("r"):
                last_winner = None
                svc.open_buzz_window(seat_map.seat_ids[:2], args.timeout, on_result)
            elif key == ord("b"):
                last_winner = None
                svc.open_buzz_window(seat_map.seat_ids[2:], args.timeout, on_result)
            elif key == ord("m"):
                manual_mode = True
            elif manual_mode and chr(key) in "1234":
                manual_mode = False
                if not svc.manual_buzz(int(chr(key))):
                    print("no window open")
            elif key == ord("c"):
                try:
                    seat_map = svc.calibrate_seats_from_latest()
                    print("calibrated:", seat_map.boundaries)
                except (ValueError, RuntimeError) as e:
                    print("calibration failed:", e)
            elif key == ord("s"):
                svc.seat_map.save(args.seats)
                print(f"saved seat zones to {args.seats}")
    finally:
        svc.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
