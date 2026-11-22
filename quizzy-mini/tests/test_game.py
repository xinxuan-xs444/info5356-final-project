"""Exercise the game rules using an injected clock."""

import random
import unittest

import config
from game.game_controller import Attempt, GameController, GameState
from game.question_bank import DIFFICULTIES, QuestionBank


class GameTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.events = []
        self.game = GameController(
            QuestionBank(rng=random.Random(7)),
            clock=lambda: self.now,
            emit=lambda event, **data: self.events.append((event, data)),
        )

    def start(self):
        for seat, name, color in [(1, "Maya Chen", "red"), (2, "Alex", "blue"),
                                  (3, "Sam", "red"), (4, "Emma", "blue")]:
            self.game.add_player(name, seat, color)
        self.game.start()

    def advance(self, duration):
        self.now += duration
        self.game.update()

    def test_question_bank(self):
        bank = self.game.question_bank
        questions = [bank.next_question() for _ in range(config.QUESTION_COUNT)]
        self.assertEqual(len({question["id"] for question in questions}), 20)
        levels = [question["difficulty"] for question in questions]
        self.assertEqual(set(levels), set(DIFFICULTIES))
        self.assertEqual(levels, sorted(levels, key=DIFFICULTIES.index))
        self.assertIsNone(bank.next_question())

    def test_steal_and_turn_taking(self):
        self.start()
        self.assertFalse(self.game.buzz(1))
        self.advance(config.THINK_DURATION)
        self.assertTrue(self.game.buzz(1))
        self.assertFalse(self.game.buzz(2))
        self.game.answer(-999)
        self.assertEqual(self.game.state, GameState.STEAL)
        self.assertFalse(self.game.buzz(3, override=True))
        self.assertTrue(self.game.buzz(2))
        self.game.answer(self.game.round.question["answer"])
        self.assertEqual(self.game.teams["blue"].score, 1)
        self.advance(config.THINK_DURATION)
        self.assertEqual(self.game.eligible_seats(), (3, 4))

    def test_question_is_said_once(self):
        self.start()
        question = self.game.round.question["question"]
        self.assertEqual(self.events.count(("say", {"text": question})), 1)

    def test_steal_opens_immediately_and_hints_on_timeout(self):
        self.start()
        self.game.players[2].answers_given = 1
        self.advance(config.THINK_DURATION)
        self.assertTrue(self.game.buzz(1))
        self.events.clear()
        self.game.answer(-999)
        self.assertEqual(self.game.state, GameState.STEAL)
        self.assertEqual(self.game.phase_deadline, self.now + config.HINT_BUZZ_DURATION)
        self.assertEqual(
            self.events[:2],
            [("state", {"state": "STEAL"}), ("buzz_open", {"seats": (2, 4)})],
        )
        for seat in (1, 3):
            self.assertFalse(self.game.buzz(seat))
            self.assertFalse(self.game.buzz(seat, override=True))
        self.advance(config.HINT_BUZZ_DURATION - 1)
        self.assertEqual(self.game.state, GameState.STEAL)
        self.advance(1)
        self.assertEqual(self.game.round.attempt, Attempt.HINT)
        self.assertIn(("say", {"text": f"Hint: {self.game.round.question['hint']}"}), self.events)

    def test_steal_calls_player_before_opening_answer_window(self):
        self.start()
        self.advance(config.THINK_DURATION)
        self.game.buzz(2)
        self.game.answer(judgment=False)
        self.assertEqual(self.game.eligible_seats(), (1, 3))
        self.events.clear()
        self.assertTrue(self.game.buzz(3))
        self.assertEqual(self.events, [
            ("state", {"state": "CALL"}),
            ("look_at_seat", {"seat": 3}),
            ("say", {"text": "Sam, what's your answer?"}),
            ("state", {"state": "JUDGE"}),
            ("answer_open", {"seat": 3, "timeout": config.ANSWER_DURATION}),
        ])

    def test_teaching_and_timeouts(self):
        self.start()
        self.advance(config.THINK_DURATION)
        self.game.buzz(1)
        self.advance(config.ANSWER_DURATION)
        self.assertEqual(self.game.state, GameState.STEAL)
        self.game.buzz(2)
        self.game.answer(judgment=False)
        self.assertEqual(self.game.state, GameState.HINT_THINK)
        self.assertEqual(self.game.round.attempt, Attempt.HINT)
        self.advance(config.HINT_THINK_DURATION)
        self.assertEqual(self.game.state, GameState.TEACH)
        question = self.game.round.question
        self.advance(config.HINT_BUZZ_DURATION)
        self.assertEqual(self.game.round.attempt, Attempt.MINI)
        self.assertIn(("say", {"text": question["mini_question"]}), self.events)
        self.game.buzz(3)
        self.game.answer(question["mini_answer"])
        self.assertEqual(self.game.teams["red"].score, 0)
        self.assertEqual(self.game.state, GameState.THINK)

    def test_hint_gives_time_to_calculate_then_accepts_all_players(self):
        self.start()
        self.advance(config.THINK_DURATION)
        self.assertTrue(self.game.buzz(1))
        self.game.answer(judgment=False)
        self.assertTrue(self.game.buzz(2))
        self.events.clear()
        self.game.answer(judgment=False)

        self.advance(config.HINT_THINK_DURATION - 1)
        self.assertEqual(self.game.state, GameState.HINT_THINK)
        self.assertFalse(self.game.buzz(1))
        self.assertFalse(self.game.buzz(2, override=True))
        self.assertFalse(any(event == "buzz_open" for event, _ in self.events))

        self.advance(1)
        self.assertEqual(self.game.phase_deadline, self.now + config.HINT_BUZZ_DURATION)
        self.assertIn(("buzz_open", {"seats": (1, 2, 3, 4)}), self.events)
        self.assertTrue(self.game.buzz(1))
        self.assertTrue(self.game.answer(self.game.round.question["answer"]))
        self.assertEqual(self.game.teams["red"].score, 1)

    def test_five_points(self):
        self.start()
        for seat in [1, 3, 1, 3, 1]:
            self.advance(config.THINK_DURATION)
            self.assertTrue(self.game.buzz(seat))
            self.game.answer(self.game.round.question["answer"])
        self.assertEqual(self.game.teams["red"].score, config.WIN_SCORE)
        self.assertEqual(self.game.state, GameState.END)
        self.assertEqual(self.game.winner, "red")

    def test_deadline_sudden_death(self):
        self.start()
        question_id = self.game.round.question["id"]
        self.advance(config.GAME_DURATION)
        self.assertTrue(self.game.sudden_death)
        self.assertNotEqual(self.game.round.question["id"], question_id)
        self.advance(config.THINK_DURATION)
        self.game.buzz(2)
        self.game.answer(self.game.round.question["answer"])
        self.assertEqual(self.game.winner, "blue")

    def test_requires_four_players_and_two_per_team(self):
        self.game.add_player("Maya", 1, "red")
        self.game.add_player("Sam", 2, "red")
        self.game.add_player("Riley", 3, "red")
        with self.assertRaisesRegex(ValueError, "Register 4 or 5 players"):
            self.game.start()
        self.game.add_player("Emma", 4, "blue")
        with self.assertRaisesRegex(ValueError, "at least two players on each team"):
            self.game.start()
        self.game.add_player("Alex", 5, "blue")
        self.game.start()
        self.assertEqual(self.game.state, GameState.THINK)

    def test_five_players_can_buzz_and_steal(self):
        for larger_team, other_team in (("red", "blue"), ("blue", "red")):
            with self.subTest(larger_team=larger_team):
                self.game.reset()
                for seat in (1, 3, 5):
                    self.game.add_player(f"Player {seat}", seat, larger_team)
                for seat in (2, 4):
                    self.game.add_player(f"Player {seat}", seat, other_team)
                self.game.start()
                self.assertEqual([team.score for team in self.game.teams.values()], [0, 0])
                self.advance(config.THINK_DURATION)
                self.assertEqual(self.game.eligible_seats(), (1, 2, 3, 4, 5))
                self.assertTrue(self.game.buzz(2))
                self.assertTrue(self.game.answer(judgment=False))
                self.assertEqual(self.game.eligible_seats(), (1, 3, 5))
                self.assertTrue(self.game.buzz(5))
                self.assertTrue(self.game.answer(judgment=True))
                self.assertEqual(self.game.teams[larger_team].score, 1)

    def test_addressing_pause_override_skip_and_reset(self):
        self.start()
        player = self.game.players[1]
        self.assertEqual(self.game.get_player_address(player), "Maya")
        self.game.addressing_mode = config.ADDRESSING_TEAM
        self.assertEqual(self.game.get_player_address(player), "Red team")
        self.assertTrue(self.game.pause())
        self.advance(config.GAME_DURATION)
        self.assertTrue(self.game.resume())
        self.advance(config.THINK_DURATION)
        self.game.buzz(1, override=True)
        self.game.answer(judgment=True)
        self.assertEqual(self.game.teams["red"].score, 1)
        question_id = self.game.round.question["id"]
        self.assertTrue(self.game.skip())
        self.assertNotEqual(self.game.round.question["id"], question_id)
        self.game.reset()
        self.assertEqual(self.game.state, GameState.INTRODUCE)
        self.assertFalse(self.game.players)
        self.assertEqual(self.game.teams["red"].score, 0)
