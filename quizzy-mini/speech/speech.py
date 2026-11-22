"""SpeechService: the one object the game talks to for voice in and voice out.

Output: every `say` event from GameController is queued and spoken on a worker thread, so
the game loop never blocks on audio.

Input: when the game opens an answer window (`answer_open`), the service waits for the
robot to finish "Maya, what's your answer?", records until the player stops talking,
transcribes with Whisper, parses the number, and hands a HeardAnswer to `on_answer`
(a thread-safe callable such as queue.Queue().put). The game loop then calls
game.answer(heard.value). This mirrors how vision delivers buzzes.

While players are thinking, it also listens for "Repeat, please" and re-reads the problem
(up to config.MAX_REPEATS times per problem).

    answers = queue.Queue()
    speech = SpeechService(make_tts(), make_listener(), on_answer=answers.put).start()
    game = create_game(speech=speech)        # main.py routes game events to speech.handle_event
    # game loop:
    #   heard = answers.get_nowait()
    #   game.answer(heard.value) if heard.kind == "answer" else game.answer(judgment=False)

Half duplex on purpose: the mic is never recording while Quizzy is talking, so the robot
cannot "hear" its own voice as an answer.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import config

from .answer_parser import Intent, extract_name, parse_answer
from .llm import LLMHelper
from .stt import NAME_PROMPT, NUMBER_PROMPT, REPEAT_PROMPT, Heard
from .tts import PrintTTS, TTSEngine, speakable

log = logging.getLogger(__name__)

ANSWER = "answer"
DONT_KNOW = "dont_know"


@dataclass(frozen=True)
class HeardAnswer:
    """What a player said in their answer window."""

    seat: int
    kind: str                 # "answer" (value is set) or "dont_know"
    value: float | None
    text: str                 # raw transcript, for logs and the operator
    confidence: float
    source: str               # "parser" or "llm"
    latency: float            # seconds from the window opening to this result


@dataclass
class _Job:
    kind: str                 # "answer" or "command"
    cancel: threading.Event
    seat: int | None = None
    deadline: float | None = None
    opened_at: float = 0.0


class SpeechService:
    def __init__(
        self,
        tts: TTSEngine | None = None,
        listener=None,
        *,
        llm: LLMHelper | None = None,
        on_answer: Callable[[HeardAnswer], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_repeats: int | None = None,
        repeat_states: tuple[str, ...] | None = None,
        min_confidence: float | None = None,
        transcribe_budget: float | None = None,
        question_provider: Callable[[], dict | None] | None = None,
    ) -> None:
        """`listener` needs listen(max_seconds, prompt, cancel) -> Heard (see stt.py).
        Without one the service only speaks; answers can still be typed in the operator console.
        `question_provider` returns the current question dict (e.g. lambda: game.round and
        game.round.question); it lets "repeat" re-read the mini-question in teaching mode."""
        self.tts = tts or PrintTTS()
        self.listener = listener
        self.llm = llm or LLMHelper.disabled()
        self.on_answer = on_answer or (lambda heard: None)
        self.clock = clock
        self.max_repeats = config.MAX_REPEATS if max_repeats is None else max_repeats
        self.repeat_states = tuple(config.REPEAT_LISTEN_STATES if repeat_states is None else repeat_states)
        self.min_confidence = config.ANSWER_MIN_CONFIDENCE if min_confidence is None else min_confidence
        self.transcribe_budget = config.STT_TRANSCRIBE_BUDGET if transcribe_budget is None else transcribe_budget
        self.question_provider = question_provider

        self._lines: queue.Queue = queue.Queue()
        self._pending = 0                       # lines queued or being spoken
        self._quiet = threading.Condition()
        self._job: _Job | None = None
        self._job_changed = threading.Condition()
        self._chunk_cancel = threading.Event()  # set when Quizzy starts talking mid-recording
        self._mic = threading.Lock()
        self._state_lock = threading.Lock()
        self._prompt: str | None = None         # the problem a "repeat" re-reads
        self._repeats = 0
        self._repeat_refused = False
        self._threads: list[threading.Thread] = []
        self._running = False

    # ------------------------------------------------------------ lifecycle

    def start(self) -> "SpeechService":
        if self._running:
            return self
        self._running = True
        self._threads = [threading.Thread(target=self._speak_loop, name="speech-tts", daemon=True)]
        if self.listener is not None:
            self._threads.append(threading.Thread(target=self._listen_loop, name="speech-stt", daemon=True))
        for thread in self._threads:
            thread.start()
        return self

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.interrupt()
        self._set_job(None)
        self._lines.put(None)
        with self._job_changed:
            self._job_changed.notify_all()
        for thread in self._threads:
            thread.join(timeout=2)
        close = getattr(self.listener, "close", None)
        if close:
            close()

    def __enter__(self) -> "SpeechService":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------ speaking

    def say(self, text: str) -> None:
        """Queue a line. Returns at once; lines are spoken in order."""
        text = (text or "").strip()
        if not text:
            return
        with self._quiet:
            self._pending += 1
        self._chunk_cancel.set()                 # stop any background "repeat" recording
        prepare = getattr(self.tts, "prepare", None)
        if prepare is not None:                  # cloud voice: start fetching audio now
            try:
                prepare(speakable(text))
            except Exception as error:
                log.debug("prepare failed: %s", error)
        self._lines.put(text)
        if not self._running:                    # not started: speak inline (handy in scripts)
            self._drain_inline()

    @property
    def is_speaking(self) -> bool:
        with self._quiet:
            return self._pending > 0

    def wait_until_quiet(self, timeout: float | None = None) -> bool:
        """Block until every queued line has been spoken. False on timeout."""
        with self._quiet:
            return self._quiet.wait_for(lambda: self._pending == 0, timeout)

    def interrupt(self) -> None:
        """Drop queued lines and cut off the current one."""
        dropped = 0
        while True:
            try:
                item = self._lines.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                dropped += 1
            else:
                self._lines.put(None)            # keep the stop signal
                break
        with self._quiet:
            self._pending = max(0, self._pending - dropped)
            self._quiet.notify_all()
        try:
            self.tts.stop()
        except Exception as error:
            log.debug("tts.stop failed: %s", error)

    def _speak_one(self, text: str) -> None:
        try:
            self.tts.speak(speakable(text))
        except Exception as error:               # audio failure must never stop the game
            log.warning("TTS failed (%s): %s", error, text)
            print(f"[Quizzy] {text}", flush=True)
        finally:
            with self._quiet:
                self._pending = max(0, self._pending - 1)
                self._quiet.notify_all()

    def _speak_loop(self) -> None:
        while True:
            text = self._lines.get()
            if text is None:
                return
            self._speak_one(text)

    def _drain_inline(self) -> None:
        while True:
            try:
                text = self._lines.get_nowait()
            except queue.Empty:
                return
            if text is not None:
                self._speak_one(text)

    # ------------------------------------------------------------ game events

    def handle_event(self, event: str, **data) -> None:
        """Route a GameController event. Safe to call from the game loop; never blocks."""
        if event == "say":
            text = data.get("text", "")
            self.say(text)
            if text and text == self._mini_question():
                self._set_prompt(text)
        elif event == "question":
            self._set_prompt(data.get("question"))
        elif event == "repeatable":              # e.g. the mini-question in teaching mode
            self._set_prompt(data.get("text"))
        elif event == "answer_open" and self.listener is not None:
            timeout = float(data.get("timeout", config.ANSWER_DURATION))
            now = self.clock()
            self._set_job(_Job(ANSWER, threading.Event(), seat=data.get("seat"),
                               deadline=now + timeout, opened_at=now))
        elif event == "state":
            state = data.get("state")
            if state == "JUDGE":
                return                           # answer_open follows right away
            if state in self.repeat_states and self.listener is not None:
                self._set_job(_Job("command", threading.Event()))
            else:
                self._set_job(None)
        elif event == "reset":
            self._set_job(None)
            self.interrupt()
            self._set_prompt(None)

    def _mini_question(self) -> str | None:
        if self.question_provider is None:
            return None
        try:
            question = self.question_provider()
        except Exception:
            return None
        return question.get("mini_question") if isinstance(question, dict) else None

    def _set_prompt(self, text: str | None) -> None:
        with self._state_lock:
            self._prompt = text
            self._repeats = 0
            self._repeat_refused = False

    def repeat_question(self) -> bool:
        """Re-read the current problem if repeats are left. Returns True if it was re-read."""
        with self._state_lock:
            prompt = self._prompt
            if not prompt:
                return False
            if self._repeats < self.max_repeats:
                self._repeats += 1
                line = prompt
            elif not self._repeat_refused:
                self._repeat_refused = True
                line = "That's all the repeats for this one. You've got this!"
            else:
                return False
        self.say(line)
        return line == prompt

    # ------------------------------------------------------------ listening

    def _set_job(self, job: _Job | None) -> None:
        with self._job_changed:
            if self._job is not None and self._job is not job:
                self._job.cancel.set()
            self._job = job
            self._job_changed.notify_all()

    def _current(self, job: _Job) -> bool:
        with self._job_changed:
            return self._running and self._job is job and not job.cancel.is_set()

    def _listen_loop(self) -> None:
        while self._running:
            with self._job_changed:
                self._job_changed.wait_for(lambda: not self._running or self._job is not None, timeout=0.5)
                job = self._job
            if job is None or not self._running:
                continue
            try:
                if job.kind == ANSWER:
                    self._run_answer(job)
                    with self._job_changed:
                        if self._job is job:
                            self._job = None
                else:
                    self._listen_for_repeat(job)
            except Exception:                    # keep listening even if one attempt crashes
                log.exception("Speech input failed")
                time.sleep(0.2)

    def _listen(self, seconds: float, prompt: str, cancel) -> Heard:
        started = time.monotonic()
        with self._mic:
            heard = self.listener.listen(seconds, prompt, cancel)
        if not heard.text and time.monotonic() - started < 0.2:
            cancel.wait(0.2)                     # a broken mic returns instantly: do not spin
        return heard

    def _run_answer(self, job: _Job) -> None:
        asked_for_number = False
        while self._current(job):
            if not self.wait_until_quiet(timeout=max(0.0, job.deadline - self.clock())):
                return
            remaining = job.deadline - self.clock() - self.transcribe_budget
            if remaining < 1.0 or not self._current(job):
                log.info("Seat %s: answer window closed with nothing usable heard", job.seat)
                return
            heard = self._listen(remaining, NUMBER_PROMPT, job.cancel)
            if not self._current(job):
                return
            result = parse_answer(heard.text)

            if result.intent is Intent.ANSWER and result.confidence >= self.min_confidence:
                self._deliver(job, ANSWER, result.value, heard.text, result.confidence, "parser")
                return
            if result.intent is Intent.REPEAT:
                self.repeat_question()
                continue
            if result.intent is Intent.DONT_KNOW:
                self._deliver(job, DONT_KNOW, None, heard.text, 1.0, "parser")
                return
            if result.intent in (Intent.UNCLEAR, Intent.ANSWER):   # unclear, or a low-confidence number
                value = self._ask_llm(heard.text, job)
                if value is not None:
                    self._deliver(job, ANSWER, value, heard.text, 0.6, "llm")
                    return
                if not asked_for_number:
                    asked_for_number = True
                    self.say("Just the number, please.")
                    continue
            # Silence, or still unclear after asking once: keep listening until the window closes.

    def _ask_llm(self, text: str, job: _Job) -> float | None:
        if not text or not self.llm.available:
            return None
        budget = job.deadline - self.clock() - 0.3
        if budget < 1.0:
            return None
        with self._state_lock:
            prompt = self._prompt
        return self.llm.interpret_answer(prompt, text, timeout=min(self.llm.timeout, budget))

    def _deliver(self, job: _Job, kind: str, value: float | None, text: str, confidence: float, source: str) -> None:
        if not self._current(job):
            return
        heard = HeardAnswer(job.seat, kind, value, text, confidence, source, self.clock() - job.opened_at)
        log.info("Seat %s answered %s (%s via %s, %.1fs)", job.seat, value if kind == ANSWER else kind,
                 repr(text), source, heard.latency)
        self.on_answer(heard)

    def _listen_for_repeat(self, job: _Job) -> None:
        """One short background recording while players think; react only to repeat requests."""
        if not self.wait_until_quiet(timeout=0.5) or not self._current(job):
            return
        self._chunk_cancel.clear()
        stop = _AnyEvent(job.cancel, self._chunk_cancel)
        heard = self._listen(4.0, REPEAT_PROMPT, stop)
        if not self._current(job) or self._chunk_cancel.is_set() or self.is_speaking:
            return                               # Quizzy talked over it: the clip is not the players
        if heard.text and parse_answer(heard.text).intent is Intent.REPEAT:
            log.info("Repeat requested: %r", heard.text)
            self.repeat_question()

    # ------------------------------------------------------------ introductions

    def ask_name(self, seat: int, *, attempts: int = 2, seconds: float = 6.0) -> str | None:
        """Ask the player in `seat` for their name and return it (None if not understood).

        Blocking; call it before game.start(), e.g. from an operator 'intro' command.
        """
        if self.listener is None:
            return None
        self.say(f"Seat {seat}, what's your first name?")
        for attempt in range(attempts):
            self.wait_until_quiet(timeout=15)
            heard = self._listen(seconds, NAME_PROMPT, threading.Event())
            name = extract_name(heard.text)
            if name:
                self.say(f"Nice to meet you, {name.split()[0]}!")
                return name
            if attempt + 1 < attempts:
                self.say("Sorry, could you say just your first name?")
        self.say("No problem, my helper will type it in.")
        return None


class _AnyEvent:
    """Looks like a threading.Event that is set when either of two events is set."""

    def __init__(self, a: threading.Event, b: threading.Event) -> None:
        self.a, self.b = a, b

    def is_set(self) -> bool:
        return self.a.is_set() or self.b.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        end = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            left = None if end is None else end - time.monotonic()
            if left is not None and left <= 0:
                return False
            self.a.wait(0.02 if left is None else min(0.02, left))
        return True


def build_speech_service(*, mini=None, on_answer: Callable[[HeardAnswer], None] | None = None,
                         listen: bool = True, question_provider=None) -> SpeechService:
    """Create a SpeechService from config.py (OpenAI voice, transcription and chat by default).

    Pass `mini` (a connected ReachyMini) to use the robot's speaker and microphones; without it
    the laptop's devices are used. Anything missing (no key, no network, no mic) degrades to a
    local voice / printed lines and keyboard answers instead of stopping the game.
    """
    from .stt import make_listener
    from .tts import make_tts

    c = lambda name, default: getattr(config, name, default)   # noqa: E731
    tts = make_tts(c("TTS_ENGINE", "openai"), voice=c("TTS_VOICE", None), rate=c("TTS_RATE", 165),
                   device=c("SPEAKER_DEVICE", None), mini=mini, fallback=c("TTS_FALLBACK", "auto"),
                   openai_model=c("TTS_OPENAI_MODEL", "gpt-4o-mini-tts"), openai_voice=c("TTS_OPENAI_VOICE", "coral"),
                   instructions=c("TTS_INSTRUCTIONS", None), timeout=c("TTS_TIMEOUT", 6.0))
    listener = None
    if listen:
        try:
            listener = make_listener(
                mini=mini, device=c("MIC_DEVICE", None), engine=c("STT_ENGINE", "openai"),
                openai_model=c("STT_OPENAI_MODEL", "gpt-4o-transcribe"), local_model=c("STT_LOCAL_MODEL", "base.en"),
                compute_type=c("STT_COMPUTE_TYPE", "int8"), timeout=c("STT_TIMEOUT", 2.5),
                min_rms=c("STT_MIN_SPEECH_RMS", 0.01), end_silence=c("STT_END_SILENCE", 0.8),
            )
            listener.transcriber.load()
            listener.transcriber.warm_up()
        except Exception as error:   # no mic or no sounddevice: keep the game playable
            log.warning("Speech input disabled (%s). Type answers in the operator console.", error)
            listener = None
    llm = LLMHelper.from_config(c("LLM_PROVIDER", "openai"), c("LLM_MODEL", None), c("LLM_TIMEOUT", 4.0))
    return SpeechService(tts, listener, llm=llm, on_answer=on_answer, question_provider=question_provider,
                         transcribe_budget=c("STT_TRANSCRIBE_BUDGET", 2.5))


def apply_answer(game, heard: HeardAnswer) -> bool:
    """Hand a HeardAnswer to GameController. Call from the game loop (same thread as update()).

    A clear number goes to game.answer(value), which judges it; "I don't know" counts as a
    miss right away so the other team can steal without waiting out the timer.
    Returns False if the game no longer accepts an answer (window closed, paused, ...).
    """
    current = getattr(getattr(game, "round", None), "player", None)
    if current is not None and heard.seat is not None and current.seat_id != heard.seat:
        return False                       # stale answer from an earlier window
    if heard.kind == ANSWER and heard.value is not None:
        return game.answer(heard.value)
    if heard.kind == DONT_KNOW:
        return game.answer(judgment=False)
    return False
