"""Physical calibration for Module 4: find the real head angles for each seat and the
real antenna angles for each score, then print config.py lines to paste in.

Usage (robot connected):

    python -m calibration seats       # jog the head seat by seat, type a correction, Enter
                                       # to lock it in
    python -m calibration antennas    # jog the RED antenna to each score 0..WIN_SCORE
                                       # (BLUE is assumed symmetric -- same angles)
    python -m calibration check       # replay the angles already in config.py, seat by seat
                                       # and score by score, so you can confirm physical
                                       # alignment (antenna points at the right scoreboard
                                       # number, head clears any decorations, antenna/pointer
                                       # don't collide) before a real session

This only exercises RobotMotion directly -- the same object main.py constructs for real
games -- never vision or game logic. Safe without hardware: pass --dry-run, or just run it;
a missing/failed SDK connection already falls back to dry-run and prints what it *would*
send (see RobotMotion.__init__).
"""

from __future__ import annotations

import argparse

import config
from robot_motion import RobotMotion


def _prompt_angle(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("not a number, keeping the previous value")
        return default


def calibrate_seats(robot: RobotMotion) -> dict[int, tuple[float, float]]:
    print("Head-angle calibration. Seat numbering (which physical side is Seat 1) must match")
    print("game logic and vision.SeatMap(seat_ids=...) -- confirm that with the team first.\n")
    result: dict[int, tuple[float, float]] = {}
    for seat, (yaw, pitch) in sorted(config.SEAT_HEAD_ANGLES.items()):
        print(f"-- Seat {seat} --")
        robot.move_head_raw(yaw, pitch)
        while True:
            yaw = _prompt_angle("yaw (deg)", yaw)
            pitch = _prompt_angle("pitch (deg)", pitch)
            robot.move_head_raw(yaw, pitch)
            if input("Looking at seat %d correctly? [Y/n] " % seat).strip().lower() in ("", "y"):
                break
        result[seat] = (yaw, pitch)
        print()
    robot.move_head_raw(*config.HEAD_NEUTRAL_ANGLE)
    return result


def calibrate_antennas(robot: RobotMotion) -> list[float]:
    print("Antenna-angle calibration: jog the RED antenna to point at each scoreboard number.")
    print("BLUE is assumed symmetric; swap in set_antenna_raw('blue', ...) above if it isn't.\n")
    result: list[float] = []
    for score, angle in enumerate(config.ANTENNA_SCORE_ANGLES):
        print(f"-- Score {score} --")
        robot.set_antenna_raw("red", angle)
        while True:
            angle = _prompt_angle("angle (deg)", angle)
            robot.set_antenna_raw("red", angle)
            if input("Pointing at %d correctly? [Y/n] " % score).strip().lower() in ("", "y"):
                break
        result.append(angle)
        print()
    robot.set_antenna_raw("red", result[0])
    return result


def check(robot: RobotMotion) -> None:
    print("Replaying the calibration currently in config.py -- this does not change or save")
    print("anything, it's just a final physical check before a session.\n")
    robot.move_head_raw(*config.HEAD_NEUTRAL_ANGLE)
    for seat, (yaw, pitch) in sorted(config.SEAT_HEAD_ANGLES.items()):
        input(f"Enter to look at seat {seat}...")
        robot.move_head_raw(yaw, pitch)
    input("Enter to return the head to neutral...")
    robot.move_head_raw(*config.HEAD_NEUTRAL_ANGLE)

    print("\nNow the antennas, 0..WIN_SCORE on both teams:")
    for score in range(len(config.ANTENNA_SCORE_ANGLES)):
        input(f"Enter to set both antennas to score {score}...")
        angle = robot.score_angle_for(score)
        robot.set_antenna_raw("red", angle)
        robot.set_antenna_raw("blue", angle)
    robot.reset()
    print("Done.")


def _print_config_lines(seats: dict, antennas: list) -> None:
    print("\nPaste into quizzy-mini/config.py, replacing the existing line(s):\n")
    if seats:
        print("SEAT_HEAD_ANGLES = {")
        for seat, (yaw, pitch) in sorted(seats.items()):
            print(f"    {seat}: ({yaw}, {pitch}),")
        print("}")
    if antennas:
        print(f"ANTENNA_SCORE_ANGLES = {tuple(antennas)}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["seats", "antennas", "check"])
    parser.add_argument("--dry-run", action="store_true", help="skip the SDK, just print what would move")
    args = parser.parse_args(argv)

    robot = RobotMotion(dry_run=args.dry_run, idle_interval=float("inf"))
    try:
        if args.mode == "seats":
            _print_config_lines(calibrate_seats(robot), [])
        elif args.mode == "antennas":
            _print_config_lines({}, calibrate_antennas(robot))
        else:
            check(robot)
    except KeyboardInterrupt:
        print("\ncalibration cancelled")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
