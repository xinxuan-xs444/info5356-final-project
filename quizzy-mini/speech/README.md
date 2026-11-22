# Module 2: Speech Interaction & LLM

Quizzy reads problems aloud, hears spoken answers, and teaches when both teams miss.

| File | What it does |
|---|---|
| `speech.py` | `SpeechService`: queues everything the game says, opens the mic when the game opens an answer window, delivers a `HeardAnswer`. Also listens for "Repeat, please" and asks players their names. |
| `answer_parser.py` | Spoken text to a number (`"seventeen and a half"` → 17.5, `"14, no wait, 15"` → 15), commands (repeat, I don't know) and names (`"I'm Maya"` → Maya). |
| `teaching.py` | `TeachingCoach`: feedback, hints, numbered worked solutions, mini-questions, and a nudge when a player answers with a middle step (e.g. the price of one slice). |
| `llm.py` | Optional helper for unclear answers and short explanations. Text only, hard timeout, scripted fallback. |
| `tts.py` / `stt.py` | Voices (macOS `say`, Reachy Mini speaker, pyttsx3) and listening (mic + endpointing + local Whisper). |
| `demo.py` | Try each piece by hand. |

How to wire it into the game: see [INTEGRATION.md](INTEGRATION.md).

## Setup (Mac recommended)

```bash
cd quizzy-mini
pip install -r ../requirements-speech.txt
python -m speech.demo devices          # find the mic / speaker names
python -m speech.demo say              # Quizzy should talk
python -m speech.demo listen           # say "seventeen and a half"; first run downloads Whisper (~140 MB)
```

macOS asks for microphone permission the first time. If nothing is heard, allow it for your terminal
under System Settings > Privacy & Security > Microphone, then restart the terminal.

**Using the robot's speaker and mics.** Either pass the connected robot (`--robot` in the demo,
`build_speech_service(mini=mini)` in code), or set `SPEAKER_DEVICE` / `MIC_DEVICE` in `config.py` to a
name substring from `python -m speech.demo devices` (e.g. `"Reachy"`).

**No mic or on Windows?** Everything still runs: `--type` lets you type what the player says, and the
operator console's `answer 14` always works.

## Try it

```bash
python -m speech.demo round            # one full round: answer, steal, hint, worked solution, mini-question
python -m speech.demo name             # "Seat 1, what's your first name?"
python -m speech.demo game             # the real GameController; type `player 1 red Maya`, `start`, `buzz 1`, then speak
python -m speech.demo game --robot     # same, through Reachy Mini
```

## How answers are handled

1. The game emits `answer_open` (seat, 10 s). Speech waits until Quizzy finishes "Maya, what's your answer?" so
   the mic never records the robot.
2. It records until the player has been quiet for 0.8 s, keeping 1.5 s at the end of the window for Whisper.
3. The parser reads the number. Then:
   - clear number → `HeardAnswer(kind="answer", value=14)`; the game judges it with `check_answer()`
   - "Repeat, please" → re-reads the problem (max 2 per problem) and keeps listening
   - "I don't know" / "pass" → `kind="dont_know"`, so the other team can steal right away
   - unclear (`"is it 14 or 15?"`) → asks the LLM if one is set up, otherwise says "Just the number, please." once
   - silence → nothing is sent; the game's own timer handles the timeout
4. The game loop calls `apply_answer(game, heard)`.

The parser and the LLM never decide right or wrong. They only extract the number.

## LLM (optional)

`config.LLM_PROVIDER = "auto"` uses [Ollama](https://ollama.com) if it is running on the laptop
(`ollama run llama3.2:3b`; nothing leaves the machine), else OpenAI if `OPENAI_API_KEY` is set, else
no LLM. The game works the same without it, with scripted lines. Calls time out after
`LLM_TIMEOUT` seconds, and the helper pauses itself for a minute after three failures. Use a
regular chat model, not a reasoning model.

## Tests

```bash
cd quizzy-mini
python -m unittest discover -s tests        # all modules, including tests/speech_tests
```

The speech tests use fake audio, so they run anywhere, including against the real `GameController`
(`tests/speech_tests/test_game_integration.py`).
Real audio must be checked by hand on the Mac with the demo commands above.

## Settings (`config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `TTS_ENGINE` | `"auto"` | `say` on macOS, the robot speaker if a robot is passed, `pyttsx3` elsewhere, else print |
| `TTS_VOICE`, `TTS_RATE` | Samantha, 165 | voice and words per minute (`say -v '?'` lists voices) |
| `SPEAKER_DEVICE`, `MIC_DEVICE` | None | device name substrings; None = system default |
| `STT_MODEL` | `base.en` | `tiny.en` is faster, `small.en` is more accurate |
| `STT_MIN_SPEECH_RMS` | 0.01 | raise it in a loud room if noise triggers recording |
| `STT_END_SILENCE` | 0.8 s | pause that ends an answer |
| `STT_TRANSCRIBE_BUDGET` | 1.5 s | time kept free at the end of the answer window for Whisper |
| `ANSWER_MIN_CONFIDENCE` | 0.5 | below this, a parsed number counts as unclear |
| `MAX_REPEATS` | 2 | "Repeat, please" requests per problem |
| `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TIMEOUT` | auto, None, 4 s | see above |
