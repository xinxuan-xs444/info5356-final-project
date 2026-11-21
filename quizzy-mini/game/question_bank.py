"""Loads and provides quiz questions."""

import json
from pathlib import Path


class QuestionBank:
    """Read questions from the JSON question bank."""

    def __init__(self, path: str = "game/questions.json") -> None:
        self.path = Path(path)

    def get_questions(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as file:
            return json.load(file)
