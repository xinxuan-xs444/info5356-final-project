# Module 2: Speech Interaction & LLM

Quizzy reads problems aloud in a game-show voice, hears spoken answers, and teaches when both teams miss.
Voice, transcription and the LLM use the OpenAI API. When the network drops, every part has a fallback,
so the game never stalls.

| File | What it does |
|---|---|
| `speech.py` | `SpeechService`: queues everything the game says, opens the mic when the game opens an answer window, delivers a `HeardAnswer`. Also listens for "Repeat, please" and asks players their names. |
| `answer_parser.py` | Spoken text to a number (`"seventeen and a half"` → 17.5, `"14, no wait, 15"` → 15), commands (repeat, I don't know) and names (`"I'm Maya"` → Maya). Runs locally, no API. |
| `teaching.py` | `TeachingCoach`: feedback, hints, numbered worked solutions, mini-questions, and a nudge when a player answers with a middle step (e.g. the price of one slice). |
| `openai_api.py` | OpenAI speech-to-text (`gpt-4o-transcribe`), text-to-speech (`gpt-4o-mini-tts`) and chat. One key from `.env`, hard timeouts, automatic pause after repeated failures. |
| `llm.py` | Chat Completions (`gpt-4o-mini`) for unclear answers and short explanations, with scripted fallbacks. |
| `tts.py` / `stt.py` | Speakers and microphones (laptop, or Reachy Mini via its SDK), endpointing, local backup voices and Whisper. |
| `demo.py` | Try each piece by hand. |

How to wire it into the game: see [INTEGRATION.md](INTEGRATION.md).

## Setup (Mac recommended, Python 3.10+)

```bash
cp .env.example .env              # at the repo root; paste the team OPENAI_API_KEY. Never commit .env
pip install -r requirements-speech.txt
cd quizzy-mini
python -m speech.demo devices     # find the mic / speaker names
python -m speech.demo say         # Quizzy should talk in the OpenAI voice
python -m speech.demo listen      # say "seventeen and a half" and check the parsed number
```

macOS asks for microphone permission the first time. If nothing is heard, allow it for your terminal
under System Settings > Privacy & Security > Microphone, then restart the terminal.

**Using the robot's speaker and mics.** Either pass the connected robot (`--robot` in the demo,
`build_speech_service(mini=mini)` in code), or set `SPEAKER_DEVICE` / `MIC_DEVICE` in `config.py` to a
name substring from `python -m speech.demo devices` (e.g. `"Reachy"`).

**No mic or on Windows?** `--type` lets you type what the player says, and the operator console's
`answer 14` always works.

## Try it

```bash
python -m speech.demo round            # one full round: answer, steal, hint, worked solution, mini-question
python -m speech.demo name             # "Seat 1, what's your first name?"
python -m speech.demo game             # the real GameController; type `player 1 red Maya`, `start`, `buzz 1`, then speak
python -m speech.demo game --robot     # same, through Reachy Mini
python -m speech.demo round --offline  # rehearse the no-internet fallback
```

## How answers are handled

1. The game emits `answer_open` (seat, 10 s). Speech waits until Quizzy finishes "Maya, what's your answer?" so
   the mic never records the robot.
2. It records until the player has been quiet for 0.8 s, keeping 2.5 s at the end of the window for transcription.
3. The audio goes to `gpt-4o-transcribe`, and the local parser reads the number. Then:
   - clear number → `HeardAnswer(kind="answer", value=14)`; the game judges it with `check_answer()`
   - "Repeat, please" → re-reads the problem (max 2 per problem) and keeps listening
   - "I don't know" / "pass" → `kind="dont_know"`, so the other team can steal right away
   - unclear (`"is it 14 or 15?"`) → asks `gpt-4o-mini` for the committed number, otherwise says "Just the number, please." once
   - silence → nothing is sent; the game's own timer handles the timeout
4. The game loop calls `apply_answer(game, heard)`.

The parser and the LLM never decide right or wrong. They only extract the number.

## When OpenAI is slow or offline

| Part | Fallback |
|---|---|
| Voice | the same line in a local voice (macOS `say`, the robot speaker, or pyttsx3); if no local voice exists, the line is printed for the operator to read out |
| Listening | local Whisper if `faster-whisper` is installed; otherwise the console shows **SPEECH RECOGNITION OFFLINE**, and the operator types `answer 14` (Wizard-of-Oz) |
| LLM | scripted lines from `questions.json` ("Just the number, please.", hints, worked steps) |

Each call has a timeout (`TTS_TIMEOUT`, `STT_TIMEOUT`, `LLM_TIMEOUT`). After 3 failures in a row a service
is skipped for 60 s, so later lines go straight to the fallback instead of waiting each time.

Lines are fetched ahead while the previous one is being spoken, and repeated lines ("Buzz open!") are
cached, so the cloud voice adds little delay.

## Notes for the team

- **Privacy:** OpenAI receives players' answer audio and the text Quizzy speaks, which includes first names in
  the name condition. No video is sent. Mention this in the study consent. `STT_ENGINE = "local"` keeps
  answer audio on the laptop.
- **Model lifetime:** OpenAI lists `tts-1`, `tts-1-hd` and `gpt-4o-mini-tts` as shutting down on
  January 6, 2027. That is after our study, and the fallback voice takes over automatically if it ever fails.
- **Cost:** a 5-minute game is a few cents (short clips plus short lines).

## Tests

```bash
cd quizzy-mini
python -m unittest discover -s tests        # all modules, including tests/speech_tests
```

The speech tests use fake audio and a fake network: no key needed, no API calls, no charges. They cover
the request format sent to OpenAI, every fallback, and play against the real `GameController`
(`tests/speech_tests/test_game_integration.py`). Real audio and real API calls must be checked by hand on
the Mac with the demo commands above.

## Settings (`config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `TTS_ENGINE` | `"openai"` | cloud voice with a local fallback; or local only: `say`, `reachy`, `pyttsx3`, `print` |
| `TTS_OPENAI_MODEL`, `TTS_OPENAI_VOICE` | `gpt-4o-mini-tts`, `coral` | `tts-1` is faster but has no style control |
| `TTS_INSTRUCTIONS` | None | voice style; None = cheerful game-show host for 11–13 year olds |
| `TTS_TIMEOUT` | 6 s | then that line uses the local voice |
| `TTS_FALLBACK`, `TTS_VOICE`, `TTS_RATE` | auto, Samantha, 165 | the local voice |
| `SPEAKER_DEVICE`, `MIC_DEVICE` | None | device name substrings; None = system default |
| `STT_ENGINE`, `STT_OPENAI_MODEL` | `openai`, `gpt-4o-transcribe` | or `gpt-4o-mini-transcribe` / `whisper-1`; `local` = offline Whisper |
| `STT_TIMEOUT` | 2.5 s | wait for a transcript |
| `STT_MIN_SPEECH_RMS` | 0.01 | raise it in a loud room if noise triggers recording |
| `STT_END_SILENCE` | 0.8 s | pause that ends an answer |
| `STT_TRANSCRIBE_BUDGET` | 2.5 s | time kept free at the end of the answer window for transcription |
| `ANSWER_MIN_CONFIDENCE` | 0.5 | below this, a parsed number counts as unclear |
| `MAX_REPEATS` | 2 | "Repeat, please" requests per problem |
| `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TIMEOUT` | `openai`, `gpt-4o-mini`, 4 s | `none` turns the LLM off |
