"""Coordinates a quiz game."""

from game.player import Player
from game.question_bank import QuestionBank


class GameController:
    """Manage one game session from start to finish."""

    def __init__(self) -> None:
        self.question_bank = QuestionBank()
        self.player: Player | None = None

    def run(self) -> None:
        name = input("Enter player name: ").strip() or "Player"
        self.player = Player(name)

        for question in self.question_bank.get_questions():
            print(f"\n{question['question']}")
            for index, answer in enumerate(question["answers"], start=1):
                print(f"{index}. {answer}")

            choice = input("Choose an answer: ").strip()
            try:
                selected = int(choice) - 1
                if question["answers"][selected] == question["correct_answer"]:
                    self.player.score += 1
            except (ValueError, IndexError):
                pass

        print(f"\n{self.player.name}, your score: {self.player.score}")
