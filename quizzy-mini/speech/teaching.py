"""What Quizzy says when it teaches: feedback, hints, worked solutions, mini-questions.

Teaching Mode in the proposal (when both teams miss):
  1. hint -> 15 s re-buzz for everyone
  2. if still wrong: step-by-step worked solution
  3. a similar mini-question worth 1 point

GameController owns *when* each step happens (timers, buzzes, scoring). This module owns
*what is said*, so the wording can improve without touching the game rules. Every line is
scripted from questions.json; the LLM only adds an optional extra explanation and the
scripted line is always the fallback.

    coach = TeachingCoach()
    coach.hint_lines(question)                 -> ["Here's a hint.", "First, find the price of one slice."]
    coach.solution_lines(question)             -> ["Let's solve it together.", "Step 1: ...", ..., "So the answer is 14 dollars."]
    coach.nudge(question, 2)                   -> "Close! 2 is a step along the way. There's one more step."
"""

from __future__ import annotations

import random
import re

from .answer_parser import format_number, parse_numbers
from .llm import LLMHelper

_CORRECT = ["Correct!", "That's right!", "Yes, exactly!", "Nailed it!", "Spot on!"]
_PRAISE = ["Nice work", "Great thinking", "Well done", "Awesome", "Super"]
_INCORRECT = ["Not quite.", "Not this time.", "Close, but no.", "Hmm, not quite."]
_ENCOURAGE = [
    "Good try. Mistakes help us learn!",
    "Nice effort. Let's figure it out together.",
    "That was a tricky one. You've got this next time!",
    "Keep going, you're getting closer!",
]
_HINT_INTRO = ["Here's a hint.", "Let me give you a hint.", "Hint time!"]
_SOLUTION_INTRO = ["Let's solve it together.", "Let's walk through it step by step.", "Here's how it works."]
_MINI_INTRO = ["Now try a similar one.", "Your turn with a practice question.", "Let's practice one more."]


class TeachingCoach:
    def __init__(self, llm: LLMHelper | None = None, *, rng: random.Random | None = None) -> None:
        self.llm = llm or LLMHelper.disabled()
        self.rng = rng or random.Random()
        self._last: dict[str, str] = {}

    def _pick(self, kind: str, options: list[str]) -> str:
        """Random choice that avoids saying the same line twice in a row."""
        choices = [line for line in options if line != self._last.get(kind)] or options
        line = self.rng.choice(choices)
        self._last[kind] = line
        return line

    # ------------------------------------------------------------ answers

    @staticmethod
    def answer_phrase(question: dict, *, mini: bool = False) -> str:
        value = question["mini_answer"] if mini else question["answer"]
        unit = question.get("unit", "").strip()
        return f"{format_number(value)} {unit}".strip()

    def feedback_correct(self, question: dict, address: str | None = None, *, mini: bool = False) -> str:
        line = f"{self._pick('correct', _CORRECT)} {self.answer_phrase(question, mini=mini)}."
        if address:
            line += f" {self._pick('praise', _PRAISE)}, {address}!"
        return line

    def feedback_incorrect(self) -> str:
        """Short and neutral: never reveals the answer, because the other team may steal."""
        return self._pick("incorrect", _INCORRECT)

    def encouragement(self) -> str:
        return self._pick("encourage", _ENCOURAGE)

    def nudge(self, question: dict, value: float | None) -> str | None:
        """A specific, safe comment on a wrong answer, or None when nothing specific applies.

        Catches the most common slip in multi-step ratio problems: answering with a number
        from a middle step (e.g. the unit price) instead of the final answer.
        """
        if value is None:
            return None
        answer = float(question["answer"])
        if _close(value, answer):
            return None
        givens = parse_numbers(question["question"])
        intermediate = []
        for step in question.get("solution", []):
            for number in parse_numbers(step):
                if number != 1 and not _close(number, answer) and not any(_close(number, g) for g in givens):
                    intermediate.append(number)
        if any(_close(value, number) for number in intermediate):
            return f"Close! {format_number(value)} is a step along the way. There's one more step."
        if any(_close(value, g) for g in givens):
            return f"{format_number(value)} is a number from the problem. Try using it to find the answer."
        if answer and (_close(value * 10, answer) or _close(value, answer * 10)):
            return "Check your place value. Is the answer bigger or smaller?"
        return None

    # ------------------------------------------------------------ teaching mode

    def hint_lines(self, question: dict) -> list[str]:
        return [self._pick("hint", _HINT_INTRO), question["hint"]]

    def solution_lines(self, question: dict) -> list[str]:
        steps = question.get("solution", [])
        lines = [self._pick("solution", _SOLUTION_INTRO)]
        if len(steps) == 1:
            lines.append(steps[0])
        else:
            lines.extend(f"Step {n}: {step}" for n, step in enumerate(steps, start=1))
        lines.append(f"So the answer is {self.answer_phrase(question)}.")
        return lines

    def mini_question_lines(self, question: dict) -> list[str]:
        return [self._pick("mini", _MINI_INTRO), question["mini_question"]]

    def mini_feedback(self, question: dict, correct: bool, address: str | None = None) -> list[str]:
        if correct:
            return [self.feedback_correct(question, address, mini=True)]
        return [f"The practice answer is {self.answer_phrase(question, mini=True)}.", self.encouragement()]

    def explain_wrong_answer(self, question: dict, value: float | None) -> str:
        """LLM explanation if available, else the scripted nudge, else the hint."""
        explanation = self.llm.explain(question, value) if self.llm.available else None
        if explanation and not _leaks_other_numbers(explanation, question):
            return explanation
        return self.nudge(question, value) or question["hint"]

    def script(self, question: dict) -> list[str]:
        """The whole teaching sequence in order, for demos and dry runs."""
        return [
            *self.hint_lines(question),
            *self.solution_lines(question),
            *self.mini_question_lines(question),
        ]


def _close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) < 1e-6


def _leaks_other_numbers(text: str, question: dict) -> bool:
    """Reject LLM text containing numbers that appear nowhere in the problem or solution.

    A cheap guard against a model inventing wrong arithmetic in front of kids.
    """
    known = set()
    for source in [question["question"], *question.get("solution", []), str(question["answer"])]:
        known.update(round(n, 6) for n in parse_numbers(source))
    for number in parse_numbers(re.sub(r"\bstep \w+\b", "", text.lower())):
        if round(number, 6) not in known and number not in (1, 2):   # "one", "two" appear in normal speech
            return True
    return False
