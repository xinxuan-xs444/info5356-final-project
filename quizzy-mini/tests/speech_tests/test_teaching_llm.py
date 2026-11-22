"""Teaching Mode wording and the LLM helper (with a fake network)."""

import json
import random
import unittest
from pathlib import Path

from speech.answer_parser import parse_answer
from speech.llm import LLMHelper, _parse_answer_json
from speech.teaching import TeachingCoach

QUESTIONS = json.loads((Path(__file__).parents[2] / "game" / "questions.json").read_text())
PIZZA = QUESTIONS[0]          # 3 slices for 6 dollars; 7 slices cost 14


def fake_transport(reply=None, error=None, calls=None):
    def transport(url, body, headers, timeout):
        if calls is not None:
            calls.append((url, body, headers, timeout))
        if error:
            raise error
        return {"choices": [{"message": {"content": reply}}]}
    return transport


class Coach(unittest.TestCase):
    def setUp(self):
        self.coach = TeachingCoach(rng=random.Random(1))

    def test_hint_solution_and_mini_question_come_from_the_bank(self):
        for question in QUESTIONS:
            hint = self.coach.hint_lines(question)
            self.assertEqual(hint[-1], question["hint"])
            solution = self.coach.solution_lines(question)
            for step in question["solution"]:
                self.assertTrue(any(step in line for line in solution))
            self.assertIn(self.coach.answer_phrase(question), solution[-1])
            self.assertEqual(self.coach.mini_question_lines(question)[-1], question["mini_question"])

    def test_steps_are_numbered(self):
        lines = self.coach.solution_lines(PIZZA)
        self.assertTrue(lines[1].startswith("Step 1: "))
        self.assertEqual(lines[-1], "So the answer is 14 dollars.")

    def test_feedback(self):
        line = self.coach.feedback_correct(PIZZA, "Maya")
        self.assertIn("14 dollars", line)
        self.assertIn("Maya", line)
        self.assertNotIn("14", self.coach.feedback_incorrect())     # never leaks the answer before a steal
        self.assertIn("10 dollars", self.coach.mini_feedback(PIZZA, False)[0])

    def test_lines_do_not_repeat_back_to_back(self):
        lines = [self.coach.feedback_incorrect() for _ in range(20)]
        self.assertTrue(all(a != b for a, b in zip(lines, lines[1:])))

    def test_nudge_spots_a_middle_step(self):
        self.assertIn("step along the way", self.coach.nudge(PIZZA, 2))       # price of one slice
        self.assertIn("from the problem", self.coach.nudge(PIZZA, 6))
        self.assertIn("place value", self.coach.nudge(PIZZA, 140))
        self.assertIsNone(self.coach.nudge(PIZZA, 14))                         # correct answer
        self.assertIsNone(self.coach.nudge(PIZZA, 37))                         # nothing specific
        self.assertIsNone(self.coach.nudge(PIZZA, None))

    def test_explain_falls_back_without_llm(self):
        self.assertIn("step along the way", self.coach.explain_wrong_answer(PIZZA, 2))
        self.assertEqual(self.coach.explain_wrong_answer(PIZZA, 37), PIZZA["hint"])

    def test_explain_uses_llm_but_rejects_invented_numbers(self):
        good = LLMHelper("http://x", "m", transport=fake_transport(
            "You found the price of one slice, 2 dollars. Multiply by 7 slices to finish."))
        self.assertIn("Multiply", TeachingCoach(good).explain_wrong_answer(PIZZA, 2))
        bad = LLMHelper("http://x", "m", transport=fake_transport("Seven slices cost 21 dollars."))
        self.assertIn("step along the way", TeachingCoach(bad).explain_wrong_answer(PIZZA, 2))

    def test_script_order(self):
        script = self.coach.script(PIZZA)
        self.assertLess(script.index(PIZZA["hint"]), script.index(PIZZA["mini_question"]))


class Llm(unittest.TestCase):
    def test_disabled_helper_returns_none(self):
        helper = LLMHelper.disabled()
        self.assertFalse(helper.available)
        self.assertIsNone(helper.interpret_answer("q", "fourteen"))
        self.assertIsNone(helper.explain(PIZZA, 2))

    def test_interpret_answer_sends_text_only_and_parses_json(self):
        calls = []
        helper = LLMHelper("http://localhost:11434/v1", "llama", transport=fake_transport('{"answer": 14}', calls=calls))
        self.assertEqual(helper.interpret_answer(PIZZA["question"], "it's 14 dollars for 7 slices", timeout=1.5), 14)
        url, body, headers, timeout = calls[0]
        self.assertTrue(url.endswith("/chat/completions"))
        self.assertEqual(timeout, 1.5)
        self.assertEqual(body["temperature"], 0)
        self.assertIn("it's 14 dollars for 7 slices", body["messages"][1]["content"])
        self.assertNotIn("Authorization", headers)

    def test_api_key_header(self):
        calls = []
        helper = LLMHelper("http://x", "m", api_key="secret", transport=fake_transport('{"answer": 1}', calls=calls))
        helper.interpret_answer("q", "one")
        self.assertEqual(calls[0][2]["Authorization"], "Bearer secret")

    def test_reply_parsing(self):
        self.assertEqual(_parse_answer_json('{"answer": 17.5}'), 17.5)
        self.assertEqual(_parse_answer_json('Sure! {"answer": "1,200"}'), 1200)
        self.assertIsNone(_parse_answer_json('{"answer": null}'))
        self.assertIsNone(_parse_answer_json('{"answer": true}'))
        self.assertIsNone(_parse_answer_json("fourteen"))
        self.assertIsNone(_parse_answer_json('{"answer": "NaN"}'))

    def test_failures_fall_back_and_pause_the_helper(self):
        now = [0.0]
        helper = LLMHelper("http://x", "m", transport=fake_transport(error=TimeoutError("slow")), clock=lambda: now[0])
        for _ in range(3):
            self.assertIsNone(helper.interpret_answer("q", "fourteen"))
        self.assertFalse(helper.available)
        now[0] = 61
        self.assertTrue(helper.available)

    def test_bad_reply_shape_is_a_failure(self):
        helper = LLMHelper("http://x", "m", transport=lambda *a: {"oops": 1})
        self.assertIsNone(helper.interpret_answer("q", "fourteen"))

    def test_from_config_none_missing_key_and_key(self):
        from unittest import mock
        self.assertFalse(LLMHelper.from_config("none").available)
        with mock.patch("speech.openai_api.load_env"), mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(LLMHelper.from_config("openai").available)
        with mock.patch("speech.openai_api.load_env"), mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            helper = LLMHelper.from_config("openai")
        self.assertTrue(helper.available)
        self.assertEqual((helper.model, helper.base_url), ("gpt-4o-mini", "https://api.openai.com/v1"))

    def test_parser_and_llm_agree_on_the_division_of_labor(self):
        # The parser handles clear answers itself; only unclear ones should reach the LLM.
        self.assertTrue(parse_answer("fourteen").has_value)
        self.assertFalse(parse_answer("It's 14 dollars for 7 slices").has_value)


if __name__ == "__main__":
    unittest.main()
