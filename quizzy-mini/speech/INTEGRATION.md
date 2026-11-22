# Wiring speech into the game

For whoever connects Module 2 once `feature/game-logic` and `feature/speech` are merged into `main`.
It is two small edits, about 25 lines, in files Module 1 owns. They are also in `integration.patch`:

```bash
git apply quizzy-mini/speech/integration.patch      # run from the repo root
```

This exact patch was applied to `feature/game-logic` and the whole test suite passed (game tests and speech tests).

## What already matches

| Game side (feature/game-logic) | Speech side |
|---|---|
| `say` event | spoken in order on a worker thread; `say()` never blocks the game loop |
| `question` event | becomes the text that "Repeat, please" re-reads |
| `state` event | THINK, BUZZ_OPEN, STEAL, HINT_THINK and TEACH: listens for "Repeat, please". Any other state cancels listening |
| `answer_open(seat, timeout)` | records that player's answer inside the game's window |
| `reset` event | stops talking and listening |
| `game.answer(value)` / `game.answer(judgment=False)` | called through `apply_answer(game, heard)` from the game loop |
| the game owns every timer | speech never times anything out on its own; silence just means no `HeardAnswer` |

Answers come back the same way vision buzzes do: `on_answer` is called from the speech thread,
so give it `queue.Queue().put`, and only the game loop touches the game.

## The patch

```diff
--- a/quizzy-mini/main.py
+++ b/quizzy-mini/main.py
@@ -1,6 +1,7 @@
 """Run the terminal simulation and route controller events to optional adapters."""
 
 import logging
+from queue import Queue
 
 from game.game_controller import GameController
 from operator_ui.operator_console import OperatorConsole
@@ -25,8 +26,8 @@
     """
     def output(event, **data):
         terminal_output(event, **data)
-        if speech is not None and event == "say":
-            speech.say(data["text"])
+        if speech is not None:
+            speech.handle_event(event, **data)
         if vision is not None:
             if event == "buzz_open":
                 vision.open_buzz(data["seats"])
@@ -44,15 +45,24 @@
         elif event == "reset":
             robot.reset()
 
-    return GameController(emit=output)
+    game = GameController(emit=output)
+    if speech is not None:
+        speech.question_provider = lambda: game.round.question if game.round else None
+    return game
 
 
 def main() -> None:
     logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
+    from speech import build_speech_service
+
+    answers = Queue()
+    speech = build_speech_service(on_answer=answers.put).start()   # later: mini=mini for the robot audio
     try:
-        OperatorConsole(create_game()).run()
+        OperatorConsole(create_game(speech=speech), answers=answers).run()
     except KeyboardInterrupt:
         print("\nGame console closed.")
+    finally:
+        speech.stop()
 
 
 if __name__ == "__main__":
--- a/quizzy-mini/operator_ui/operator_console.py
+++ b/quizzy-mini/operator_ui/operator_console.py
@@ -7,6 +7,7 @@
 
 import config
 from game.game_controller import GameController, GameState
+from speech import apply_answer
 
 logger = logging.getLogger(__name__)
 HELP = (
@@ -16,8 +17,9 @@
 
 
 class OperatorConsole:
-    def __init__(self, game: GameController) -> None:
+    def __init__(self, game: GameController, answers: Queue | None = None) -> None:
         self.game = game
+        self.answers = answers
 
     def run(self) -> None:
         commands = Queue()
@@ -37,6 +39,7 @@
         Thread(target=read_commands, daemon=True).start()
         while True:
             self.game.update()
+            self.handle_spoken_answer()
             try:
                 command = commands.get(timeout=0.05)
             except Empty:
@@ -44,6 +47,16 @@
             if not self.execute(command):
                 return
 
+    def handle_spoken_answer(self) -> None:
+        if self.answers is None:
+            return
+        try:
+            heard = self.answers.get_nowait()
+        except Empty:
+            return
+        if apply_answer(self.game, heard):
+            logger.info("Spoken answer: %r -> %s", heard.text, heard.value if heard.value is not None else heard.kind)
+
     def execute(self, command: str) -> bool:
         action, _, argument = command.strip().partition(" ")
         action = action.lower()
```

What it does:

1. `create_game()` forwards **every** event to `speech.handle_event()`, not just `say`. Speech needs
   `question`, `state` and `answer_open` too.
2. `speech.question_provider` lets "Repeat, please" re-read the mini-question during teaching mode.
3. `main()` builds the service from `config.py`. If the mic, Whisper or `sounddevice` is missing, it logs
   a warning and keeps going in speak-only mode, so the operator can still type `answer 14`.
4. The console loop hands spoken answers to the game. Manual `answer`, `correct` and `incorrect` keep working
   as overrides.

## With the robot connected

When Module 4 opens the robot (`with ReachyMini() as mini:`), pass the same object so audio uses the robot's
speaker and microphones:

```python
speech = build_speech_service(mini=mini, on_answer=answers.put).start()
```

Or keep the laptop audio and set `SPEAKER_DEVICE = "Reachy"` / `MIC_DEVICE = "Reachy"` in `config.py`.

## Optional: richer teaching lines

`GameController` already says the hint, the steps and the mini-question itself, and that works as is.
For more natural wording, Module 1 can swap its `say()` calls for `TeachingCoach` lines.
These come from the same `questions.json` data, so there are no new rules:

```python
from speech import TeachingCoach
coach = TeachingCoach()
for line in coach.solution_lines(question):   # "Let's solve it together.", "Step 1: ...", "So the answer is 14 dollars."
    self.say(line)
self.say(coach.feedback_correct(question, address))   # "That's right! 14 dollars. Nice work, Maya!"
coach.nudge(question, value)    # "Close! 2 is a step along the way." for a middle-step answer
```

## Still to agree as a team

- **Think time vs. reading time.** `next_question()` starts the 20 s THINK timer when the problem is *emitted*.
  Reading a hard problem aloud takes 6 to 9 s, so players effectively get about 12 s. One option: Module 1 starts
  the timer after `speech.wait_until_quiet()`, or adds the reading time
  (`time_to_say(text)` from `speech.tts` gives an estimate).
- **"Reads each problem twice."** The final proposal says Quizzy reads each problem twice. The game currently
  says it once (`test_question_is_said_once`), and players can ask for repeats. Speech supports either; it's
  Module 1's call.
- **Answer window.** The 10 s window includes Quizzy saying "Maya, what's your answer?" (about 1.5 s) and 1.5 s
  for Whisper, which leaves the player about 7 s to talk. That's plenty for a number. If the room is slow,
  raise `ANSWER_DURATION` or set `STT_MODEL = "tiny.en"`.
- **Pause.** `game.pause()` emits no event, so a player can still be heard while paused. The game then rejects
  the answer, and the console logs it as ignored. If we want, the game could emit `state` on pause.
