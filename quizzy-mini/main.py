"""Launch the integrated quiz or the keyboard-only simulation."""

import argparse
import logging
from contextlib import ExitStack
from pathlib import Path
from queue import Empty, Queue

from game.game_controller import GameController
from operator_ui.operator_console import OperatorConsole

log = logging.getLogger(__name__)


def terminal_output(event, **data) -> None:
    if event == "say":
        print(data["text"])
    elif event == "state":
        print(f"[{data['state']}]")
    elif event == "score":
        print(f"RED: {data['scores']['red']} | BLUE: {data['scores']['blue']}")
    elif event == "question":
        print(f"Difficulty: {data['difficulty'].title()}")


def create_game(*, speech=None, vision=None, robot=None, board=None,
                buzzes=None, answers=None) -> GameController:
    """Route controller events to the available adapters on the game thread."""
    def output(event, **data):
        if vision is not None and event in ("state", "reset"):
            vision.close_buzz()
            next_result(buzzes)
        if event in ("state", "reset", "answer_open"):
            next_result(answers)
        terminal_output(event, **data)
        if speech is not None:
            speech.handle_event(event, **data)
        if event == "buzz_open":
            # The camera is mounted on the head. Wait for the neutral move before opening vision.
            neutral = robot is None or robot.look_neutral()
            if vision is not None and neutral:
                vision.open_buzz(data["seats"])
            if robot is not None:
                robot.wiggle_antennas()
        elif event == "look_at_seat" and robot is not None:
            robot.look_at_seat(data["seat"])
        elif event == "score":
            if board is not None:
                board.update(data["scores"])
            if robot is not None:
                for team, score in data["scores"].items():
                    robot.set_score(team, score)
        elif event == "reset":
            if board is not None:
                board.reset()
            if robot is not None:
                robot.reset()

    game = GameController(emit=output)
    if speech is not None:
        speech.question_provider = lambda: game.round.question if game.round else None
    return game


def next_result(results):
    if results is not None:
        try:
            return results.get_nowait()
        except Empty:
            pass
    return None


def start_adapter(name, start):
    try:
        return start()
    except Exception as error:
        log.warning("%s unavailable (%s). Keyboard controls are available.", name, error)
        return None


def start_vision(stack, mini, args, on_buzz):
    from vision import ReachyMiniSource, SeatMap, VisionService, WebcamSource, YoloPoseEstimator

    seats = SeatMap.load(args.seats)
    estimator = YoloPoseEstimator()
    source = ReachyMiniSource(mini) if mini is not None else WebcamSource(args.camera)
    try:
        service = VisionService(source, estimator, seats, on_buzz=on_buzz)
    except BaseException:
        source.close()
        raise
    stack.callback(service.stop)
    return service.start()


def run_integrated(args) -> None:
    from robot_motion import RobotMotion
    from speech import apply_answer, build_speech_service

    def connect_robot():
        from reachy_mini import ReachyMini

        return stack.enter_context(ReachyMini())

    def start_speech():
        service = build_speech_service(mini=mini, on_answer=answers.put)
        stack.callback(service.stop)
        return service.start()

    def start_board():
        from scoreboard import Scoreboard

        board = Scoreboard()
        stack.callback(board.close)
        return board

    buzzes, answers = Queue(), Queue()
    with ExitStack() as stack:
        mini = start_adapter("Reachy Mini", connect_robot)
        # Idle head motion would move the camera out of its calibrated position during buzzing.
        robot = stack.enter_context(RobotMotion(mini=mini, dry_run=mini is None,
                                               idle_interval=float("inf")))
        board = start_adapter("Scoreboard", start_board)
        speech = start_adapter("Speech", start_speech)
        vision = start_adapter("Vision", lambda: start_vision(stack, mini, args, buzzes.put))
        game = create_game(speech=speech, vision=vision, robot=robot, board=board,
                           buzzes=buzzes, answers=answers)

        def poll_adapters():
            buzz = next_result(buzzes)
            if buzz is not None and not game.buzz(buzz.seat):
                vision.reopen()
            answer = next_result(answers)
            if answer is not None:
                apply_answer(game, answer)
            if board is not None:
                board.pump()

        OperatorConsole(game, on_tick=poll_adapters).run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyboard-only", action="store_true", help="run without hardware or audio")
    parser.add_argument("--seats", type=Path, default=Path(__file__).with_name("seats.json"),
                        help="path to the calibrated vision seat map")
    parser.add_argument("--camera", type=int, default=0, help="webcam index without Reachy Mini")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    try:
        if args.keyboard_only:
            OperatorConsole(create_game()).run()
        else:
            run_integrated(args)
    except KeyboardInterrupt:
        print("\nGame console closed.")


if __name__ == "__main__":
    main()
