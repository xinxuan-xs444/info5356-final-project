"""Application configuration for Quizzy Mini."""
# All code files should read parameters from config.py.

GAME_DURATION = 300      # 5 min
THINK_DURATION = 20      # 20 sec to think after the question is read
ANSWER_DURATION = 10     # 10 sec to answer after raising hand
HINT_THINK_DURATION = 5  # 5 sec to think after the hint
HINT_BUZZ_DURATION = 15  # 15 sec waiting for a buzz

WIN_SCORE = 5
MAX_PLAYERS = 4
MIN_PLAYERS = 3

QUESTION_COUNT = 20



# Experimental conditions
ADDRESSING_NAME = "name"
ADDRESSING_TEAM = "team"

# --- Module 4: Robot Motion & Scoreboard ---
# All angles in degrees. Calibrate with `python -m calibration seats` / `python -m calibration antennas`
# before a real session; these are placeholders until then.

# Seat -> (yaw, pitch) for the head to look at that seat. Direction (which physical side is
# Seat 1) must match game logic's seat numbering and vision.SeatMap(seat_ids=...) -- still
# "to agree as a team" per vision/INTEGRATION.md, so treat these as provisional.
SEAT_HEAD_ANGLES = {
    1: (-30.0, -5.0),
    2: (-10.0, -5.0),
    3: (10.0, -5.0),
    4: (30.0, -5.0),
}
HEAD_NEUTRAL_ANGLE = (0.0, -5.0)   # also the pose vision's camera assumes when a buzz window opens
HEAD_MOVE_DURATION = 0.6           # seconds for a look_at_seat / look_neutral move

# Hard safety clamps, independent of calibration above -- never send the head or antennas
# past these regardless of what a bad calibration value says.
HEAD_SAFE_YAW_RANGE = (-45.0, 45.0)
HEAD_SAFE_PITCH_RANGE = (-20.0, 20.0)
ANTENNA_SAFE_RANGE = (-90.0, 150.0)
MAX_MOVE_SPEED_DEG_S = 180.0

# Antenna angle for each score 0..WIN_SCORE (index = score), one needle per team.
ANTENNA_SCORE_ANGLES = (-60.0, -24.0, 12.0, 48.0, 84.0, 120.0)
ANTENNA_WIGGLE_DEGREES = 20.0
ANTENNA_WIGGLE_CYCLES = 2

IDLE_ANIMATION_INTERVAL = 12.0   # seconds with no game "state" event before an idle animation plays

# --- Module 2: speech, answer parsing and LLM (read by speech/*) ---
# Needs OPENAI_API_KEY in a .env file at the repo root (copy .env.example). Without a key, or
# when the network drops, everything falls back: local voice, keyboard answers, scripted lines.

# Voice (text-to-speech)
TTS_ENGINE = "openai"          # "openai" (cloud + local fallback) or a local-only: "say", "reachy", "pyttsx3", "print"
TTS_OPENAI_MODEL = "gpt-4o-mini-tts"   # or "tts-1" (faster, no style control) / "tts-1-hd"
TTS_OPENAI_VOICE = "coral"     # alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, ...
TTS_INSTRUCTIONS = None        # None = the built-in game-show host style (speech/openai_api.py)
TTS_TIMEOUT = 6.0              # seconds before a line falls back to the local voice
TTS_FALLBACK = "auto"          # local voice: "auto" (say on macOS, pyttsx3 elsewhere), "say", "pyttsx3", "print"
TTS_VOICE = "Samantha"         # macOS `say` voice for the fallback; list with: say -v '?'
TTS_RATE = 150                 # words per minute for the fallback voice (a bit slow, for kids)
TTS_LINE_PAUSE = 0.3           # short pause between consecutive lines
SPEAKER_DEVICE = None          # None = default, or a name substring such as "Reachy"
MIC_DEVICE = None              # list devices with: python -m speech.demo devices

# Listening (speech-to-text)
STT_ENGINE = "openai"          # "openai" (cloud; local Whisper as backup if installed) or "local"
STT_OPENAI_MODEL = "gpt-4o-transcribe"   # or "gpt-4o-mini-transcribe" / "whisper-1"
STT_TIMEOUT = 2.5              # seconds to wait for a transcript before giving up on it
STT_LOCAL_MODEL = "base.en"    # offline backup (pip install faster-whisper)
STT_COMPUTE_TYPE = "int8"
STT_MIN_SPEECH_RMS = 0.01      # quietest level that still counts as speech (0-1 scale)
STT_END_SILENCE = 1.2          # seconds of silence that ends an answer (room for "um... fourteen")
ENCOURAGE_AFTER = 4.0          # quiet this long in an answer window -> "Take your time." (once)
STT_TRANSCRIBE_BUDGET = 2.5    # seconds kept free at the end of an answer window for transcription

ANSWER_MIN_CONFIDENCE = 0.5    # parser confidence below this is treated as "unclear"
MAX_REPEATS = 2                # "Repeat, please" requests honored per problem
REPEAT_LISTEN_STATES = ("THINK", "BUZZ_OPEN", "STEAL", "HINT_THINK", "TEACH")

# LLM (Chat Completions): unclear answers and short explanations, text only
LLM_PROVIDER = "openai"        # "openai" or "none"
LLM_MODEL = "gpt-4o-mini"
LLM_TIMEOUT = 4.0              # seconds; on timeout the scripted line is used
