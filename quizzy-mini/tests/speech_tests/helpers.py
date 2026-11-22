"""Fakes for speech tests: a listener that answers by prompt, and a quiet TTS."""

import threading
import time
from collections import deque

from speech.stt import NAME_PROMPT, NUMBER_PROMPT, REPEAT_PROMPT, Heard
from speech.tts import PrintTTS


class QuietTTS(PrintTTS):
    """PrintTTS without printing. seconds_per_word simulates speaking time."""

    def __init__(self, seconds_per_word: float = 0.0) -> None:
        super().__init__(seconds_per_word)
        self.lock = threading.Lock()

    def speak(self, text: str) -> None:
        with self.lock:
            self.spoken.append(text)
        self._stop.clear()
        if self.seconds_per_word:
            self._stop.wait(self.seconds_per_word * len(text.split()))


class PromptListener:
    """Replies according to what is being listened for.

    answers: lines for answer windows (NUMBER_PROMPT); a callable is evaluated at listen time.
    chatter: lines for background listening while players think (REPEAT_PROMPT).
    names:   lines for introductions (NAME_PROMPT).
    Running out means silence. Records whether Quizzy was talking when each recording began.
    """

    def __init__(self, answers=(), chatter=(), names=(), *, delay: float = 0.02, service=None) -> None:
        self.queues = {NUMBER_PROMPT: deque(answers), REPEAT_PROMPT: deque(chatter), NAME_PROMPT: deque(names)}
        self.delay = delay
        self.service = service
        self.calls: list[tuple[str, float, bool]] = []
        self.lock = threading.Lock()

    def add_answers(self, *lines) -> None:
        with self.lock:
            self.queues[NUMBER_PROMPT].extend(lines)

    def listen(self, max_seconds, prompt=None, cancel=None, start_timeout=None) -> Heard:
        speaking = bool(self.service and self.service.is_speaking)
        with self.lock:
            self.calls.append((prompt, max_seconds, speaking))
            queue = self.queues.get(prompt, deque())
            line = queue.popleft() if queue else None
        wait = self.delay if line is not None else min(max_seconds, start_timeout or 0.1, 0.1)
        end = time.monotonic() + wait
        while time.monotonic() < end:
            if cancel is not None and cancel.is_set():
                return Heard("")
            time.sleep(0.005)
        if callable(line):
            line = line()
        return Heard(line or "")

    def answer_calls(self):
        return [c for c in self.calls if c[0] == NUMBER_PROMPT]


def wait_for(predicate, timeout: float = 5.0, step: float = 0.01) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(step)
    return predicate()
