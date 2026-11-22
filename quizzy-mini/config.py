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

# --- Module 2: speech, answer parsing and LLM (read by speech/*) ---
# Audio devices. None = system default. A name substring such as "Reachy" also works.
TTS_ENGINE = "auto"            # "auto", "say" (macOS), "reachy", "pyttsx3" or "print"
TTS_VOICE = "Samantha"         # macOS voice; list them with: say -v '?'
TTS_RATE = 165                 # words per minute; a bit slower than default for kids
SPEAKER_DEVICE = None          # e.g. "Reachy Mini Audio"; list with: python -m speech.demo devices
MIC_DEVICE = None              # e.g. "Reachy"

STT_MODEL = "base.en"          # faster-whisper model: tiny.en, base.en, small.en
STT_COMPUTE_TYPE = "int8"      # int8 runs well on a Mac CPU
STT_MIN_SPEECH_RMS = 0.01      # quietest level that still counts as speech (0-1 scale)
STT_END_SILENCE = 0.8          # seconds of silence that ends an utterance
STT_TRANSCRIBE_BUDGET = 1.5    # seconds kept free at the end of an answer window for Whisper

ANSWER_MIN_CONFIDENCE = 0.5    # parser confidence below this is treated as "unclear"
MAX_REPEATS = 2                # "Repeat, please" requests honored per problem
REPEAT_LISTEN_STATES = ("THINK", "BUZZ_OPEN", "STEAL", "HINT_THINK", "TEACH")

# LLM helper: text only, used for unclear answers and short explanations.
# "auto" uses Ollama if it is running locally, else OpenAI if OPENAI_API_KEY is set, else off.
LLM_PROVIDER = "auto"          # "auto", "ollama", "openai" or "none"
LLM_MODEL = None               # None = provider default (see speech/llm.py)
LLM_TIMEOUT = 4.0              # seconds; on timeout the scripted fallback is used
