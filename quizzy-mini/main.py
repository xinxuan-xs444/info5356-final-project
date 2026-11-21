"""Run the terminal simulation and route controller events to optional adapters."""

import logging

from game.game_controller import GameController
from operator_ui.operator_console import OperatorConsole


def terminal_output(event, **data) -> None:
    if event == "say":
        print(data["text"])
    elif event == "state":
        print(f"[{data['state']}]")
    elif event == "score":
        print(f"RED: {data['scores']['red']} | BLUE: {data['scores']['blue']}")
    elif event == "question":
        print(f"Difficulty: {data['difficulty'].title()}")


def create_game(*, speech=None, vision=None, robot=None) -> GameController:
    """Attach output adapters. Input adapters call buzz() and answer() on the game.

    Keep callbacks quick: hardware adapters can enqueue work on their own workers.
    Dispatch input events on the same loop that calls update().
    """
    def output(event, **data):
        terminal_output(event, **data)
        if speech is not None and event == "say":
            speech.say(data["text"])
        if vision is not None:
            if event == "buzz_open":
                vision.open_buzz(data["seats"])
            elif event in ("state", "reset"):
                vision.close_buzz()
        if robot is None:
            return
        if event == "look_at_seat":
            robot.look_at_seat(data["seat"])
        elif event == "buzz_open":
            robot.wiggle_antennas()
        elif event == "score":
            for team, score in data["scores"].items():
                robot.set_score(team, score)
        elif event == "reset":
            robot.reset()

    return GameController(emit=output)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    try:
        OperatorConsole(create_game()).run()
    except KeyboardInterrupt:
        print("\nGame console closed.")


if __name__ == "__main__":
    main()
