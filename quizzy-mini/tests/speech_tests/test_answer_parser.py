"""Spoken answers, commands and names -> structured results."""

import json
import unittest
from pathlib import Path

from speech.answer_parser import (
    Intent, extract_name, format_number, match_name, parse_answer, parse_numbers,
)

QUESTIONS = json.loads((Path(__file__).parents[2] / "game" / "questions.json").read_text())


class NumberParsing(unittest.TestCase):
    def check(self, text, expected):
        result = parse_answer(text)
        self.assertEqual(result.intent, Intent.ANSWER, f"{text!r}: {result}")
        self.assertAlmostEqual(result.value, expected, places=6, msg=text)

    def test_digits_and_words(self):
        cases = {
            "12": 12, "twelve": 12, "Twelve.": 12, "I think it's twelve": 12, "It's 14.": 14,
            "$14": 14, "Fourteen dollars.": 14, "twenty-four": 24, "Twenty four miles": 24,
            "ninety-nine dollars": 99, "forty": 40, "fourty eight liters": 48, "1,200": 1200,
            "one hundred fifty": 150, "a hundred and fifty grams": 150, "two hundred and fifty": 250,
            "three hundred milliliters": 300, "fifteen hundred": 1500, "2 hundred": 200,
            "negative five": -5, "50 kilometers per hour": 50, "um... nine": 9,
        }
        for text, expected in cases.items():
            self.check(text, expected)

    def test_decimals_and_fractions(self):
        cases = {
            "17.5": 17.5, "seventeen point five": 17.5, "seventeen and a half": 17.5,
            "17 point 5": 17.5, "one and a half": 1.5, "a half": 0.5, "point five": 0.5,
            "three quarters": 0.75, "3/4": 0.75, "3 / 4": 0.75, "1 1/2": 1.5, "seven and a quarter": 7.25,
            "2½": 2.5, "point two five": 0.25,
        }
        for text, expected in cases.items():
            self.check(text, expected)

    def test_every_answer_in_the_bank_parses_from_digits_and_words(self):
        words = {14: "fourteen", 12: "twelve", 9: "nine", 20: "twenty", 5: "five", 10: "ten",
                 24: "twenty four", 33: "thirty three", 32: "thirty two", 6: "six", 40: "forty",
                 48: "forty eight", 150: "one hundred fifty", 99: "ninety nine", 50: "fifty",
                 300: "three hundred", 17.5: "seventeen and a half"}
        for question in QUESTIONS:
            for field in ("answer", "mini_answer"):
                value = question[field]
                unit = question["unit"]
                self.check(f"I think it's {format_number(value)} {unit}", value)
                if value in words:
                    self.check(f"{words[value]} {unit}", value)

    def test_self_correction_uses_the_last_number(self):
        self.check("um 14, no wait, 15", 15)
        self.check("twelve... actually twenty", 20)
        self.check("I think 2, sorry, I mean 12 dollars", 12)

    def test_working_out_loud_uses_the_stated_result(self):
        self.check("7 times 2 is 14", 14)
        self.check("3 slices is 6 dollars so 7 slices is 14", 14)
        self.check("the answer is 33", 33)

    def test_same_number_twice_is_fine(self):
        self.check("14, 14 dollars", 14)

    def test_guessing_between_numbers_is_unclear(self):
        for text in ["is it 14 or 15?", "fourteen or fifteen", "It's 14 dollars for 7 slices", "one two three"]:
            result = parse_answer(text)
            self.assertEqual(result.intent, Intent.UNCLEAR, text)
            self.assertIsNone(result.value)
            self.assertGreater(len(result.candidates), 1)

    def test_hedged_answer_has_lower_confidence(self):
        result = parse_answer("not sure, maybe 14")
        self.assertEqual(result.value, 14)
        self.assertLess(result.confidence, parse_answer("14").confidence)

    def test_homophones_only_for_short_utterances(self):
        self.check("For.", 4)
        self.check("ate", 8)
        self.assertLess(parse_answer("For.").confidence, 0.95)
        self.assertEqual(parse_answer("I want to go for a walk").intent, Intent.UNCLEAR)

    def test_parse_numbers_lists_everything(self):
        self.assertEqual(parse_numbers("A pizza shop sells 3 slices for 6 dollars. How much do 7 slices cost?"),
                         [3, 6, 7])
        self.assertEqual(parse_numbers("Three packs contain 4 × 3 = 12 pens."), [3, 4, 3, 12])
        self.assertEqual(parse_numbers("A 750-gram mix in a ratio of 2 to 3"), [750, 2, 3])


class Intents(unittest.TestCase):
    def test_repeat_requests(self):
        for text in ["Repeat, please.", "Can you repeat please?", "Could you say that again?",
                     "What's the question again?", "One more time please", "Pardon?",
                     "Can you repeat? How many for 7 slices?"]:
            self.assertEqual(parse_answer(text).intent, Intent.REPEAT, text)

    def test_dont_know(self):
        for text in ["I don't know", "I dont know", "Pass.", "no idea", "I give up", "not sure"]:
            self.assertEqual(parse_answer(text).intent, Intent.DONT_KNOW, text)

    def test_silence_and_whisper_hallucinations(self):
        for text in ["", "   ", "Thank you.", "Thanks for watching!", "you", "Bye.", None]:
            self.assertEqual(parse_answer(text).intent, Intent.NONE, repr(text))

    def test_words_without_numbers(self):
        self.assertEqual(parse_answer("hmm let me think about it").intent, Intent.UNCLEAR)


class Names(unittest.TestCase):
    def test_extract_name(self):
        cases = {
            "Hi, my name is maya chen": "Maya Chen", "My name's Sam.": "Sam",
            "I'm Sam and I'm on red team": "Sam", "It's Emma.": "Emma", "Maya": "Maya",
            "Hello Maya": "Maya", "um, Alex": "Alex", "call me Vishnu": "Vishnu",
            "This is Isaiah": "Isaiah", "I am Xinxuan": "Xinxuan",
        }
        for text, expected in cases.items():
            self.assertEqual(extract_name(text), expected, text)

    def test_no_name(self):
        for text in ["I'm ready", "Thank you.", "", "I am not sure what to say here today"]:
            self.assertIsNone(extract_name(text), text)

    def test_match_name(self):
        roster = ["Maya Chen", "Sam", "Emma"]
        self.assertEqual(match_name("Maya", roster), "Maya Chen")
        self.assertEqual(match_name("my name is emma", roster), "Emma")
        self.assertEqual(match_name("Emmah", roster), "Emma")
        self.assertIsNone(match_name("Tom", roster))


class Formatting(unittest.TestCase):
    def test_format_number(self):
        self.assertEqual(format_number(14.0), "14")
        self.assertEqual(format_number(17.5), "17.5")
        self.assertEqual(format_number(1 / 3), "0.33")


if __name__ == "__main__":
    unittest.main()
