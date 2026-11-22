"""SpeechService threading and answer-window behavior, with fake audio."""

import queue
import time
import unittest

from speech.llm import LLMHelper
from speech.speech import ANSWER, DONT_KNOW, SpeechService

from .helpers import PromptListener, QuietTTS, wait_for

QUESTION = "A pizza shop sells 3 slices for 6 dollars. How much do 7 slices cost?"


def llm_returning(value):
    reply = "{\"answer\": %s}" % ("null" if value is None else value)
    return LLMHelper("http://x", "m", transport=lambda *a: {"choices": [{"message": {"content": reply}}]})


class ServiceTestCase(unittest.TestCase):
    def make(self, answers=(), chatter=(), names=(), *, tts=None, llm=None, **kw):
        self.answers = queue.Queue()
        self.tts = tts or QuietTTS()
        self.listener = PromptListener(answers, chatter, names)
        self.service = SpeechService(self.tts, self.listener, llm=llm, on_answer=self.answers.put,
                                     transcribe_budget=0.2, **kw)
        self.listener.service = self.service
        self.service.start()
        self.addCleanup(self.service.stop)
        return self.service

    def open_window(self, seat=1, timeout=2.5):
        self.service.handle_event("question", question=QUESTION, difficulty="easy")
        self.service.handle_event("state", state="JUDGE")
        self.service.handle_event("answer_open", seat=seat, timeout=timeout)

    def next_answer(self, timeout=3.0):
        return self.answers.get(timeout=timeout)


class Speaking(ServiceTestCase):
    def test_lines_are_spoken_in_order_without_blocking(self):
        service = self.make(tts=QuietTTS(seconds_per_word=0.02))
        started = time.monotonic()
        for line in ["one two three", "four five", "six"]:
            service.say(line)
        self.assertLess(time.monotonic() - started, 0.05)          # say() never blocks the game loop
        self.assertTrue(service.is_speaking)
        self.assertTrue(service.wait_until_quiet(timeout=2))
        self.assertEqual(self.tts.spoken, ["one two three", "four five", "six"])
        self.assertFalse(service.is_speaking)

    def test_math_is_made_speakable(self):
        service = self.make()
        service.handle_event("say", text="Three packs contain 4 × 3 = 12 pens.")
        service.wait_until_quiet(2)
        self.assertEqual(self.tts.spoken, ["Three packs contain 4 times 3 equals 12 pens."])

    def test_interrupt_drops_the_queue(self):
        service = self.make(tts=QuietTTS(seconds_per_word=0.5))
        for n in range(5):
            service.say(f"line {n}")
        time.sleep(0.05)
        service.interrupt()
        self.assertTrue(service.wait_until_quiet(timeout=1))
        self.assertLessEqual(len(self.tts.spoken), 1)

    def test_tts_failure_does_not_stop_the_service(self):
        class Broken(QuietTTS):
            def speak(self, text):
                raise OSError("no audio device")

        service = self.make(tts=Broken())
        from unittest import mock
        with mock.patch("builtins.print"):
            service.say("hello")
            self.assertTrue(service.wait_until_quiet(timeout=1))
            service.say("still alive")
            self.assertTrue(service.wait_until_quiet(timeout=1))

    def test_unstarted_service_speaks_inline(self):
        tts = QuietTTS()
        SpeechService(tts).say("hello")
        self.assertEqual(tts.spoken, ["hello"])


class AnswerWindow(ServiceTestCase):
    def test_clear_answer_is_delivered(self):
        self.make(answers=["I think it's fourteen dollars"])
        self.open_window(seat=3)
        heard = self.next_answer()
        self.assertEqual((heard.seat, heard.kind, heard.value, heard.source), (3, ANSWER, 14.0, "parser"))
        self.assertEqual(heard.text, "I think it's fourteen dollars")
        self.assertLess(heard.latency, 1.0)

    def test_waits_for_quizzy_to_finish_before_recording(self):
        service = self.make(answers=["14"], tts=QuietTTS(seconds_per_word=0.1))
        service.handle_event("say", text="Maya, what's your answer?")
        self.open_window(timeout=3)
        self.next_answer()
        self.assertFalse(any(speaking for _, _, speaking in self.listener.answer_calls()))

    def test_recording_leaves_time_to_transcribe(self):
        self.make(answers=["14"])
        self.open_window(timeout=2.5)
        self.next_answer()
        _, seconds, _ = self.listener.answer_calls()[0]
        self.assertLessEqual(seconds, 2.5 - 0.2 + 0.01)

    def test_repeat_then_answer(self):
        self.make(answers=["Repeat, please", "fourteen"])
        self.open_window()
        self.assertEqual(self.next_answer().value, 14)
        self.assertIn(QUESTION, self.tts.spoken)

    def test_dont_know(self):
        self.make(answers=["I don't know"])
        self.open_window()
        heard = self.next_answer()
        self.assertEqual((heard.kind, heard.value), (DONT_KNOW, None))

    def test_unclear_asks_for_just_the_number(self):
        self.make(answers=["is it 14 or 15?", "fifteen"])
        self.open_window(timeout=3)
        self.assertEqual(self.next_answer().value, 15)
        self.assertIn("Just the number, please.", self.tts.spoken)

    def test_unclear_goes_to_llm_when_available(self):
        self.make(answers=["it's 14 dollars for 7 slices"], llm=llm_returning(14))
        self.open_window()
        heard = self.next_answer()
        self.assertEqual((heard.value, heard.source), (14, "llm"))
        self.assertNotIn("Just the number, please.", self.tts.spoken)

    def test_llm_unsure_falls_back_to_asking(self):
        self.make(answers=["is it 14 or 15?", "15"], llm=llm_returning(None))
        self.open_window(timeout=3)
        self.assertEqual(self.next_answer().value, 15)
        self.assertIn("Just the number, please.", self.tts.spoken)

    def test_silence_delivers_nothing(self):
        self.make(answers=[])
        self.open_window(timeout=1.5)
        with self.assertRaises(queue.Empty):
            self.answers.get(timeout=1.8)

    def test_window_closing_cancels_listening(self):
        service = self.make()
        self.listener.delay = 1.0
        self.listener.add_answers("fourteen")
        self.open_window(timeout=3)
        self.assertTrue(wait_for(lambda: self.listener.answer_calls()))
        service.handle_event("state", state="STEAL")             # e.g. operator judged it
        with self.assertRaises(queue.Empty):
            self.answers.get(timeout=1.3)

    def test_reset_cancels_everything(self):
        service = self.make(tts=QuietTTS(seconds_per_word=0.3))
        self.listener.delay = 1.0
        self.listener.add_answers("fourteen")
        service.say("a long line that takes a while to say")
        self.open_window(timeout=4)
        service.handle_event("reset")
        self.assertTrue(service.wait_until_quiet(timeout=1))
        with self.assertRaises(queue.Empty):
            self.answers.get(timeout=1.5)


class BackgroundRepeat(ServiceTestCase):
    def test_repeat_while_thinking(self):
        service = self.make(chatter=["um what", "Can you repeat that please?"])
        service.handle_event("question", question=QUESTION, difficulty="easy")
        service.handle_event("state", state="THINK")
        self.assertTrue(wait_for(lambda: QUESTION in self.tts.spoken))

    def test_chatter_with_numbers_is_ignored(self):
        service = self.make(chatter=["I think it's 14", "no it's 12"])
        service.handle_event("question", question=QUESTION, difficulty="easy")
        service.handle_event("state", state="THINK")
        time.sleep(0.4)
        self.assertEqual(self.tts.spoken, [])
        self.assertTrue(self.answers.empty())

    def test_repeat_limit(self):
        service = self.make(max_repeats=2)
        service.handle_event("question", question=QUESTION, difficulty="easy")
        results = [service.repeat_question() for _ in range(4)]
        service.wait_until_quiet(2)
        self.assertEqual(results, [True, True, False, False])
        self.assertEqual(self.tts.spoken.count(QUESTION), 2)
        self.assertEqual(sum("all the repeats" in line for line in self.tts.spoken), 1)

    def test_new_problem_resets_the_limit(self):
        service = self.make(max_repeats=1)
        service.handle_event("question", question=QUESTION, difficulty="easy")
        self.assertTrue(service.repeat_question())
        service.handle_event("question", question="Next one?", difficulty="easy")
        self.assertTrue(service.repeat_question())

    def test_not_listening_outside_thinking_states(self):
        service = self.make(chatter=["repeat please"] * 5)
        service.handle_event("state", state="SCORE")
        time.sleep(0.3)
        self.assertEqual(self.listener.calls, [])

    def test_mini_question_becomes_the_repeat_target(self):
        mini = "How much do 5 slices cost?"
        service = self.make(question_provider=lambda: {"mini_question": mini})
        service.handle_event("question", question=QUESTION, difficulty="easy")
        service.handle_event("say", text=mini)
        service.repeat_question()
        service.wait_until_quiet(2)
        self.assertEqual(self.tts.spoken, [mini, mini])


class Introductions(ServiceTestCase):
    def test_ask_name(self):
        service = self.make(names=["Hi, my name is Maya"])
        self.assertEqual(service.ask_name(1), "Maya")
        service.wait_until_quiet(2)
        self.assertEqual(self.tts.spoken, ["Seat 1, what's your first name?", "Nice to meet you, Maya!"])

    def test_ask_name_retries_then_gives_up(self):
        service = self.make(names=["um", "Thank you."])
        self.assertIsNone(service.ask_name(2))
        service.wait_until_quiet(2)
        self.assertIn("Sorry, could you say just your first name?", self.tts.spoken)
        self.assertIn("No problem, my helper will type it in.", self.tts.spoken)

    def test_speak_only_service_ignores_answer_windows(self):
        service = SpeechService(QuietTTS()).start()
        self.addCleanup(service.stop)
        service.handle_event("answer_open", seat=1, timeout=2)
        self.assertIsNone(service._job)

    def test_speak_only_service_skips_names(self):
        tts = QuietTTS()
        service = SpeechService(tts).start()
        self.addCleanup(service.stop)
        self.assertIsNone(service.ask_name(1))


if __name__ == "__main__":
    unittest.main()
