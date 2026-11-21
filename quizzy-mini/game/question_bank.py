"""Load, validate, and serve math questions in increasing difficulty."""

import json
import math
import random
from pathlib import Path

from config import QUESTION_COUNT

DIFFICULTIES = ("easy", "medium", "hard")


class QuestionBank:
    def __init__(self, path: str | Path | None = None, *, rng=None) -> None:
        default_bank = path is None
        path = Path(__file__).with_name("questions.json") if default_bank else Path(path)
        with path.open(encoding="utf-8") as file:
            self.questions = json.load(file)
        self._validate()
        if default_bank and len(self.questions) != QUESTION_COUNT:
            raise ValueError(f"Expected {QUESTION_COUNT} questions in {path}")
        self.rng = rng if rng is not None else random.Random()
        self.reset()

    def _validate(self) -> None:
        if not isinstance(self.questions, list) or not self.questions:
            raise ValueError("The question bank must contain a nonempty list")
        seen = set()
        for question in self.questions:
            if not isinstance(question, dict):
                raise ValueError("Every question must be an object")
            question_id = question.get("id")
            if type(question_id) is not int or question_id in seen:
                raise ValueError(f"Invalid or duplicate question ID: {question_id}")
            seen.add(question_id)
            if question.get("difficulty") not in DIFFICULTIES:
                raise ValueError(f"Question {question_id}: invalid difficulty")
            for field in ("question", "unit", "hint", "mini_question"):
                value = question.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Question {question_id}: invalid {field}")
            for field in ("answer", "mini_answer"):
                value = question.get(field)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"Question {question_id}: invalid {field}")
            steps = question.get("solution")
            if not isinstance(steps, list) or not steps or any(
                not isinstance(step, str) or not step.strip() for step in steps
            ):
                raise ValueError(f"Question {question_id}: missing solution steps")

    def get_questions(self) -> list[dict]:
        return list(self.questions)

    def reset(self) -> None:
        ordered = []
        for difficulty in DIFFICULTIES:
            group = [q for q in self.questions if q["difficulty"] == difficulty]
            self.rng.shuffle(group)
            ordered.extend(group)
        self._remaining = iter(ordered)

    def next_question(self) -> dict | None:
        return next(self._remaining, None)
