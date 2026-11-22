"""Module 2: speech interaction, answer parsing, teaching mode and the optional LLM helper.

Start here: SpeechService (speech.py). See README.md and INTEGRATION.md in this folder.
"""

from .answer_parser import Intent, ParseResult, extract_name, format_number, match_name, parse_answer
from .llm import LLMHelper
from .speech import HeardAnswer, SpeechService, apply_answer, build_speech_service
from .teaching import TeachingCoach

__all__ = [
    "HeardAnswer", "Intent", "LLMHelper", "ParseResult", "SpeechService", "TeachingCoach",
    "apply_answer", "build_speech_service", "extract_name", "format_number", "match_name", "parse_answer",
]
