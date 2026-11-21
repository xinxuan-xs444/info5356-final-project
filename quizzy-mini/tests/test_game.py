"""Tests for the Quizzy Mini game package."""

from game.player import Player
from game.question_bank import QuestionBank


def test_player_starts_at_zero() -> None:
    player = Player("Ada")
    assert player.name == "Ada"
    assert player.score == 0


def test_question_bank_loads_questions() -> None:
    questions = QuestionBank("quizzy-mini/game/questions.json").get_questions()
    assert isinstance(questions, list)
