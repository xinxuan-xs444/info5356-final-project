"""Try Module 2 by hand. Run from the quizzy-mini folder.

    python -m speech.demo devices              list microphones and speakers
    python -m speech.demo mic                  record 5 s and play it back (no API needed)
    python -m speech.demo say "Hello, I'm Quizzy!"
    python -m speech.demo listen               say answers; see transcript and parsed number
    python -m speech.demo name                 the introduction flow
    python -m speech.demo round                one full voice round with teaching mode
    python -m speech.demo game                 the real game: type 'buzz 1' etc., answer by voice

Options:
    --robot     use the Reachy Mini speaker and microphones (needs the reachy-mini SDK)
    --type      type what the player "says" instead of using a microphone (any OS)
    --mic NAME  microphone name substring, e.g. --mic Reachy
    --offline   no OpenAI: local voice, local Whisper (if installed), scripted lines
                (this is also what happens automatically when the network drops)

OpenAI voice, transcription and the LLM need OPENAI_API_KEY in .env (copy .env.example).
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import queue
import random
import sys
import threading
import time

import config

from .answer_parser import format_number, parse_answer
from .speech import SpeechService, apply_answer, build_speech_service
from .stt import NUMBER_PROMPT, Heard, list_devices
from .teaching import TeachingCoach


class TypedListener:
    """Keyboard stand-in for the microphone."""

    def listen(self, max_seconds, prompt=None, cancel=None, start_timeout=None) -> Heard:
        try:
            return Heard(input(f"  (type what the player says, {max_seconds:.0f}s) > "))
        except EOFError:
            return Heard("")

    def close(self) -> None:
        pass


def build(args, mini, *, listen=True) -> SpeechService:
    if args.offline:
        config.TTS_ENGINE, config.STT_ENGINE, config.LLM_PROVIDER = config.TTS_FALLBACK, "local", "none"
    if args.mic:
        config.MIC_DEVICE = args.mic
    speech = build_speech_service(mini=mini, listen=listen and not args.type)
    if listen and args.type:
        speech.listener = TypedListener()
    if listen and speech.listener is None:
        print("No microphone or speech recognition available: try --type")
    return speech


def cmd_devices(args, mini) -> None:
    print(list_devices())
    print("\nmacOS voices: run `say -v '?'`. Set SPEAKER_DEVICE / MIC_DEVICE in config.py to a name substring.")


def cmd_mic(args, mini) -> None:
    """Microphone and speaker check without any API: record one utterance, report it, play it back."""
    import numpy as np

    from .stt import SoundDeviceRecorder
    from .tts import SoundDevicePlayer

    recorder = SoundDeviceRecorder(args.mic or config.MIC_DEVICE)
    print(f"Microphone: {recorder.name}. Say something now (up to 5 s)...", flush=True)
    clip = recorder.record(5.0)
    peak = float(np.max(np.abs(clip.samples))) if len(clip.samples) else 0.0
    print(f"  speech detected: {clip.speech_detected}, length {clip.duration:.1f}s, peak level {peak:.2f}")
    if not clip.speech_detected:
        print("  Nothing heard. Check System Settings > Privacy & Security > Microphone for Terminal.")
        return
    print("  Playing it back...", flush=True)
    SoundDevicePlayer(config.SPEAKER_DEVICE).play(clip.samples, 16000)


def cmd_say(args, mini) -> None:
    with build(args, mini, listen=False) as speech:
        speech.say(" ".join(args.text) or "Hi! I'm Quizzy, your math game host. 4 × 3 = 12!")
        speech.wait_until_quiet()


def cmd_listen(args, mini) -> None:
    with build(args, mini) as speech:
        if speech.listener is None:
            return
        print("Say an answer (Ctrl+C to stop). Try: 'seventeen and a half', 'repeat please', 'I don't know'.")
        count = 0
        while not args.count or count < args.count:
            count += 1
            start = time.monotonic()
            heard = speech.listener.listen(10, NUMBER_PROMPT)
            result = parse_answer(heard.text)
            print(f"  heard {heard.text!r} -> {result.intent.value}, value={result.value}, "
                  f"confidence={result.confidence:.2f} ({result.reason}); {time.monotonic() - start:.1f}s")


def cmd_name(args, mini) -> None:
    with build(args, mini) as speech:
        for seat in (1, 2):
            print(f"Seat {seat}: {speech.ask_name(seat)!r}")
        speech.wait_until_quiet()


def cmd_round(args, mini) -> None:
    from game.question_bank import QuestionBank

    question = random.choice(QuestionBank().get_questions())
    answers: queue.Queue = queue.Queue()
    window = max(config.ANSWER_DURATION, 15)     # the demo gives players a relaxed 15 s to answer
    with build(args, mini) as speech:
        if speech.listener is None:
            print("Stopping: speech recognition is not available (see the warning above).")
            return
        coach = TeachingCoach(speech.llm)
        speech.on_answer = answers.put
        speech.handle_event("question", question=question["question"])
        speech.say(question["question"])

        def ask(seat: int) -> float | None:
            speech.say(f"Seat {seat}, what's your answer?")
            speech.handle_event("answer_open", seat=seat, timeout=window)
            try:
                heard = answers.get(timeout=window + 1)
            except queue.Empty:
                speech.say("Time is up.")
                return None
            speech.handle_event("state", state="SCORE")
            return heard.value

        value = ask(1)
        if value is not None and abs(value - question["answer"]) < 1e-6:
            speech.say(coach.feedback_correct(question, "Seat 1"))
        else:
            speech.say(coach.feedback_incorrect())
            if value is not None:
                print(f"  (nudge: {coach.explain_wrong_answer(question, value)})")
            for line in coach.hint_lines(question):
                speech.say(line)
            value = ask(2)
            if value is not None and abs(value - question["answer"]) < 1e-6:
                speech.say(coach.feedback_correct(question, "Seat 2"))
            else:
                for line in coach.solution_lines(question) + coach.mini_question_lines(question):
                    speech.say(line)
                speech.handle_event("repeatable", text=question["mini_question"])
                mini_value = ask(1)
                correct = mini_value is not None and abs(mini_value - question["mini_answer"]) < 1e-6
                for line in coach.mini_feedback(question, correct, "Seat 1"):
                    speech.say(line)
        speech.wait_until_quiet()
    print(f"Answer was {format_number(question['answer'])} {question['unit']}.")


def cmd_game(args, mini) -> None:
    """The real GameController and operator console, with spoken questions and spoken answers."""
    from main import terminal_output
    from operator_ui.operator_console import HELP, OperatorConsole
    from game.game_controller import GameController

    answers: queue.Queue = queue.Queue()
    if args.type:
        print("In game mode, type answers as console commands: answer 14")
    speech = build(args, mini, listen=not args.type)
    speech.on_answer = answers.put
    speech.start()

    def emit(event, **data):
        terminal_output(event, **data)
        speech.handle_event(event, **data)

    game = GameController(emit=emit)
    speech.question_provider = lambda: game.round.question if game.round else None
    console = OperatorConsole(game)
    commands: queue.Queue = queue.Queue()

    def read() -> None:
        try:
            while True:
                line = input()
                commands.put(line)
                if line.strip().lower() == "quit":
                    return
        except EOFError:
            commands.put("quit")

    print("Quizzy Mini's Math Showdown (voice answers on)")
    print(HELP + " | intro SEAT red|blue")
    threading.Thread(target=read, daemon=True).start()
    try:
        while True:
            game.update()
            try:
                heard = answers.get_nowait()
                if not apply_answer(game, heard):
                    print(f"(late answer ignored: {heard.text!r})")
            except queue.Empty:
                pass
            try:
                command = commands.get(timeout=0.05)
            except queue.Empty:
                continue
            action, _, rest = command.strip().partition(" ")
            if action == "intro":            # intro SEAT COLOR: ask the player's name out loud
                seat, color = rest.split()
                name = speech.ask_name(int(seat)) or input(f"Name for seat {seat}? ")
                command = f"player {seat} {color} {name}"
            if not console.execute(command):
                return
    finally:
        speech.stop()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["devices", "mic", "say", "listen", "name", "round", "game"])
    parser.add_argument("text", nargs="*")
    parser.add_argument("--robot", action="store_true")
    parser.add_argument("--type", action="store_true")
    parser.add_argument("--mic")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--count", type=int, default=0, help="listen: stop after this many answers")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[%(asctime)s] %(message)s")

    with contextlib.ExitStack() as stack:
        mini = None
        if args.robot:
            from reachy_mini import ReachyMini

            mini = stack.enter_context(ReachyMini())
        try:
            globals()[f"cmd_{args.command}"](args, mini)
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    sys.exit(main())
