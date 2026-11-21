"""Quizzy Mini - application entry point."""

from game.game_controller import GameController
from operator.operator_console import OperatorConsole


def main() -> None:
    """Run the game or operator console based on command-line input."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "operator":
        OperatorConsole().run()
    else:
        GameController().run()


if __name__ == "__main__":
    main()
