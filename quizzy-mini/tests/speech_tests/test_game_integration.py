"""Module 2 wired to the real GameController from feature/game-logic, with fake audio.

The loop below is what the operator console does once speech is integrated:
update the game, hand over any heard answer with apply_answer(), repeat.
"""

import queue
import random
import time
import unittest
from unittest import mock

import config
from game.game_controller import GameController, GameState
from game.question_bank import QuestionBank
from speech.speech import SpeechService, apply_answer

from .helpers import PromptListener, QuietTTS

FAST = {"THINK_DURATION": 0.2, "ANSWER_DURATION": 2.5, "HINT_THINK_DURATION": 0.2, "HINT_BUZZ_DURATION": 1.0}


class GameWithSpeech(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.multiple(config, **FAST)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.events = []
        self.answers = queue.Queue()
        self.tts = QuietTTS()
        self.listener = PromptListener()
        self.speech = SpeechService(self.tts, self.listener, on_answer=self.answers.put, transcribe_budget=0.2)
        self.listener.service = self.speech

        def emit(event, **data):
            self.events.append((event, data))
            self.speech.handle_event(event, **data)

        self.game = GameController(QuestionBank(rng=random.Random(3)), emit=emit)
        self.speech.question_provider = lambda: self.game.round.question if self.game.round else None
        self.speech.start()
        self.addCleanup(self.speech.stop)
        for seat, name, color in [(1, "Maya Chen", "red"), (2, "Alex", "blue"), (3, "Sam", "red"), (4, "Emma", "blue")]:
            self.game.add_player(name, seat, color)

    def run_until(self, predicate, timeout=6.0):
        """The game loop: update, apply heard answers, until predicate() or timeout."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.game.update()
            try:
                apply_answer(self.game, self.answers.get(timeout=0.01))
            except queue.Empty:
                pass
            if predicate():
                return True
        return predicate()

    def answer_now(self, offset=0):
        return lambda: f"I think it's {self.game.round.question['answer'] + offset:g} {self.game.round.question['unit']}"

    def test_spoken_correct_answer_scores(self):
        self.game.start()
        self.assertTrue(self.run_until(lambda: self.game.state is GameState.BUZZ_OPEN))
        self.listener.add_answers(self.answer_now())
        self.assertTrue(self.game.buzz(1))
        self.assertTrue(self.run_until(lambda: self.game.teams["red"].score == 1))
        self.speech.wait_until_quiet(2)
        self.assertIn("Maya, what's your answer?", self.tts.spoken)

    def test_wrong_answer_then_spoken_steal(self):
        self.game.start()
        self.run_until(lambda: self.game.state is GameState.BUZZ_OPEN)
        self.listener.add_answers(self.answer_now(offset=1), self.answer_now())
        self.game.buzz(1)
        self.assertTrue(self.run_until(lambda: self.game.state is GameState.STEAL))
        self.assertTrue(self.game.buzz(2))
        self.assertTrue(self.run_until(lambda: self.game.teams["blue"].score == 1))
        self.assertEqual(self.game.teams["red"].score, 0)

    def test_dont_know_hands_over_to_the_other_team_at_once(self):
        self.game.start()
        self.run_until(lambda: self.game.state is GameState.BUZZ_OPEN)
        self.listener.add_answers("I don't know")
        self.game.buzz(2)
        started = time.monotonic()
        self.assertTrue(self.run_until(lambda: self.game.state is GameState.STEAL))
        self.assertLess(time.monotonic() - started, config.ANSWER_DURATION)

    def test_silence_times_out_through_the_game_timer(self):
        self.game.start()
        self.run_until(lambda: self.game.state is GameState.BUZZ_OPEN)
        self.game.buzz(1)
        self.assertTrue(self.run_until(lambda: self.game.state is GameState.STEAL, timeout=4))
        self.speech.wait_until_quiet(2)
        self.assertTrue(any(line.startswith("Time is up.") for line in self.tts.spoken))

    def test_teaching_mode_mini_question_answered_by_voice(self):
        self.game.start()
        self.run_until(lambda: self.game.state is GameState.BUZZ_OPEN)
        # Both teams miss, then nobody answers after the hint -> worked solution + mini-question.
        self.listener.add_answers(self.answer_now(offset=1), self.answer_now(offset=2))
        self.game.buzz(1)
        self.run_until(lambda: self.game.state is GameState.STEAL)
        self.game.buzz(2)
        self.assertTrue(self.run_until(lambda: self.game.state is GameState.HINT_THINK))
        question = self.game.round.question
        self.assertTrue(self.run_until(lambda: self.game.round.question is question
                                       and self.game.state is GameState.TEACH
                                       and question["mini_question"] in [d.get("text") for e, d in self.events if e == "say"]))
        self.listener.add_answers(lambda: f"{question['mini_answer']:g}")
        self.assertTrue(self.game.buzz(3))
        self.assertTrue(self.run_until(lambda: self.game.state is GameState.THINK and self.game.round.question is not question))
        self.speech.wait_until_quiet(2)
        self.assertIn("Correct!", self.tts.spoken)
        self.assertEqual(self.game.teams["red"].score, 0)        # practice question is not a point

    def test_stale_answer_from_another_seat_is_ignored(self):
        from speech.speech import ANSWER, HeardAnswer
        self.game.start()
        self.run_until(lambda: self.game.state is GameState.BUZZ_OPEN)
        self.game.buzz(1)
        late = HeardAnswer(seat=2, kind=ANSWER, value=1, text="one", confidence=1, source="parser", latency=0)
        self.assertFalse(apply_answer(self.game, late))
        self.assertEqual(self.game.state, GameState.JUDGE)


if __name__ == "__main__":
    unittest.main()
