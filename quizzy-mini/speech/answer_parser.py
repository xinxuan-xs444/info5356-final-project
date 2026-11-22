"""Turn what a player said into a number, a command, or a name.

Pure Python with no audio or network dependencies, so it is easy to test.

    parse_answer("I think it's twelve")          -> value 12.0, intent ANSWER
    parse_answer("seventeen and a half")         -> value 17.5
    parse_answer("um 14, no wait, 15")            -> value 15.0 (the correction wins)
    parse_answer("is it 14 or 15?")               -> value None, reason "ambiguous"
    parse_answer("can you repeat please")         -> intent REPEAT
    parse_answer("I don't know")                  -> intent DONT_KNOW
    extract_name("Hi, my name is maya chen")       -> "Maya Chen"

The parser never decides whether an answer is right. It only extracts the number;
GameController.check_answer() judges it.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "Intent",
    "ParseResult",
    "parse_answer",
    "parse_numbers",
    "detect_intent",
    "extract_name",
    "match_name",
    "format_number",
]


class Intent(Enum):
    ANSWER = "answer"          # a number was found
    REPEAT = "repeat"          # "Repeat, please", "say that again", ...
    DONT_KNOW = "dont_know"    # "I don't know", "pass", ...
    UNCLEAR = "unclear"        # speech, but no single number (e.g. "14 or 15?")
    NONE = "none"              # silence or a known Whisper hallucination


@dataclass(frozen=True)
class ParseResult:
    text: str
    intent: Intent
    value: float | None = None
    confidence: float = 0.0
    candidates: tuple[float, ...] = field(default_factory=tuple)
    reason: str = ""

    @property
    def has_value(self) -> bool:
        return self.value is not None


# ---------------------------------------------------------------- vocabulary

_UNITS = {
    "zero": 0, "oh": 0, "nought": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
_DENOMINATORS = {
    "half": 2, "halves": 2, "third": 3, "thirds": 3, "quarter": 4, "quarters": 4,
    "fourth": 4, "fourths": 4, "fifth": 5, "fifths": 5, "sixth": 6, "sixths": 6,
    "seventh": 7, "sevenths": 7, "eighth": 8, "eighths": 8, "ninth": 9, "ninths": 9,
    "tenth": 10, "tenths": 10,
}
_UNICODE_FRACTIONS = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3}

# Whisper sometimes writes a lone number as a homophone ("For." for "Four.").
# Only used when nothing else in a very short utterance is a number.
_HOMOPHONES = {"to": 2, "too": 2, "for": 4, "fore": 4, "won": 1, "ate": 8, "free": 3, "tree": 3}

_CORRECTION_CUES = {"no", "wait", "actually", "sorry", "mean", "instead", "change", "correction", "oops"}
_ANSWER_CUES = {"is", "equals", "equal", "=", "answer", "so", "makes", "get", "got", "be", "it's", "its"}

_REPEAT_PATTERNS = [
    r"\brepeat\b", r"\bsay (?:that|it|the question) again\b", r"\bone more time\b",
    r"\bagain,? please\b", r"\bwhat(?: was| is|'s) the question\b", r"\bpardon\b",
    r"\bcome again\b", r"\bdidn'?t (?:hear|catch)\b", r"\bread (?:it|that) again\b",
]
_DONT_KNOW_PATTERNS = [
    r"\bi (?:do ?n[o']?t|dont) know\b", r"\bdunno\b", r"\bno idea\b", r"\bi give up\b",
    r"^\s*(?:pass|skip)\b", r"\bwe pass\b", r"\bi pass\b", r"\bnot sure\b", r"\bno clue\b",
]
# Phrases Whisper is known to invent on silence or noise.
_HALLUCINATIONS = {
    "", "you", "thank you", "thanks", "thank you very much", "thanks for watching",
    "thank you for watching", "bye", "okay", "ok", "uh", "um", "hmm", "mm", "oh",
    "subtitles by the amara org community", "please subscribe", "so", "music", "applause",
}

_TOKEN_RE = re.compile(
    r"\d+/\d+"                      # 3/4
    r"|\d+(?:,\d{3})+(?:\.\d+)?"      # 1,200 or 1,200.5
    r"|\d*\.\d+|\d+"                 # 17.5, .5, 14
    r"|[½¼¾⅓⅔]"
    r"|[a-z]+(?:'[a-z]+)?"           # words, with contractions like it's
    r"|[=/]"
)


def _tokenize(text: str) -> list[str]:
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)        # twenty-four -> twenty four
    text = re.sub(r"(?<=\d)\s*/\s*(?=\d)", "/", text)          # 3 / 4 -> 3/4
    text = re.sub(r"(?<![\w.])-(?=\d)", " minus ", text)       # -5 -> minus 5
    return _TOKEN_RE.findall(text)


def _is_digit_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:,\d{3})*(?:\.\d+)?|\d*\.\d+", token))


def _digit_value(token: str) -> float:
    return float(token.replace(",", ""))


# ---------------------------------------------------------------- number spans

@dataclass(frozen=True)
class _Span:
    value: float
    start: int
    end: int          # exclusive token index


def _read_word_integer(tokens: list[str], i: int) -> tuple[int, int] | None:
    """Read 'two hundred and fifty', 'twenty four', 'a hundred'. Returns (value, next index)."""
    total = 0
    current = 0
    last = None          # kind of the previous number word
    j = i
    consumed_any = False
    while j < len(tokens):
        tok = tokens[j]
        nxt = tokens[j + 1] if j + 1 < len(tokens) else None
        if tok in ("a", "an") and not consumed_any and nxt in _SCALES:
            current, last, j, consumed_any = 1, "unit", j + 1, True
            continue
        if tok == "and" and last == "scale" and nxt is not None and (nxt in _UNITS or nxt in _TEENS or nxt in _TENS):
            j += 1
            continue
        if tok in _UNITS and tok != "oh":
            if last in ("unit", "teen"):
                break
            current += _UNITS[tok]
            last = "unit"
        elif tok in _TEENS:
            if last in ("unit", "teen", "tens"):
                break
            current += _TEENS[tok]
            last = "teen"
        elif tok in _TENS:
            if last in ("unit", "teen", "tens"):
                break
            current += _TENS[tok]
            last = "tens"
        elif tok in _SCALES and consumed_any:
            scale = _SCALES[tok]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
            last = "scale"
        else:
            break
        consumed_any = True
        j += 1
    if not consumed_any:
        return None
    return total + current, j


def _read_decimal_words(tokens: list[str], j: int) -> tuple[str, int]:
    """After 'point': 'five' -> '5', 'two five' -> '25', 'twenty five' -> '25'."""
    digits = ""
    while j < len(tokens):
        tok = tokens[j]
        if tok in _UNITS:
            digits += str(_UNITS[tok])
            j += 1
        elif tok in _TEENS or tok in _TENS:
            read = _read_word_integer(tokens, j)
            if read is None:
                break
            digits += str(read[0])
            j = read[1]
        elif re.fullmatch(r"\d+", tok):
            digits += tok
            j += 1
        else:
            break
    return digits, j


def _read_fraction(tokens: list[str], j: int) -> tuple[float, int] | None:
    """Read 'a half', 'one half', 'three quarters', 'two thirds' starting at j."""
    if j >= len(tokens):
        return None
    tok = tokens[j]
    if tok in ("a", "an", "one") and j + 1 < len(tokens) and tokens[j + 1] in _DENOMINATORS:
        return 1 / _DENOMINATORS[tokens[j + 1]], j + 2
    if tok in _UNITS and tok not in ("zero", "oh") and j + 1 < len(tokens) and tokens[j + 1] in _DENOMINATORS:
        return _UNITS[tok] / _DENOMINATORS[tokens[j + 1]], j + 2
    if tok in _UNICODE_FRACTIONS:
        return _UNICODE_FRACTIONS[tok], j + 1
    return None


def _read_number(tokens: list[str], i: int) -> _Span | None:
    sign = 1
    j = i
    if tokens[j] in ("minus", "negative") and j + 1 < len(tokens):
        sign, j = -1, j + 1

    tok = tokens[j]
    value: float | None = None

    # "point five" with no whole part -> 0.5
    if tok == "point" and j + 1 < len(tokens):
        digits, k = _read_decimal_words(tokens, j + 1)
        if digits:
            return _Span(sign * float(f"0.{digits}"), i, k)

    # A plain fraction with no whole part: "a half", "three quarters", "½".
    fraction = _read_fraction(tokens, j)
    if fraction is not None and not (tok in _UNITS and j + 1 < len(tokens) and tokens[j + 1] in _SCALES):
        value, j = fraction
        return _Span(sign * value, i, j)

    if _is_digit_token(tok):
        value = _digit_value(tok)
        j += 1
        if j < len(tokens) and tokens[j] in _SCALES:                 # "2 hundred"
            value *= _SCALES[tokens[j]]
            j += 1
        if j < len(tokens) and re.fullmatch(r"\d+/\d+", tokens[j]):  # "1 1/2"
            num, den = tokens[j].split("/")
            if int(den):
                value += int(num) / int(den)
                j += 1
    elif re.fullmatch(r"\d+/\d+", tok):
        num, den = tok.split("/")
        if not int(den):
            return None
        value, j = int(num) / int(den), j + 1
    else:
        read = _read_word_integer(tokens, j)
        if read is None:
            return None
        value, j = float(read[0]), read[1]
        if j + 1 < len(tokens) and tokens[j] == "point":
            digits, k = _read_decimal_words(tokens, j + 1)
            if digits:
                value = float(f"{int(value)}.{digits}")
                j = k

    # "seventeen and a half", "1 and a quarter", "2 and three quarters", "3½"
    if j < len(tokens) and tokens[j] == "and":
        fraction = _read_fraction(tokens, j + 1)
        if fraction is not None:
            value += fraction[0]
            j = fraction[1]
    elif j < len(tokens) and tokens[j] in _UNICODE_FRACTIONS:
        value += _UNICODE_FRACTIONS[tokens[j]]
        j += 1
    # "point five" with a digit whole part written by Whisper as words is handled above;
    # "14 point 5" (digits) is rare but cheap to support.
    if j + 1 < len(tokens) and tokens[j] == "point" and re.fullmatch(r"\d+", tokens[j + 1]) and float(value).is_integer():
        value = float(f"{int(value)}.{tokens[j + 1]}")
        j += 2

    return _Span(sign * value, i, j)


def _spans(tokens: list[str]) -> list[_Span]:
    spans: list[_Span] = []
    i = 0
    while i < len(tokens):
        span = _read_number(tokens, i)
        if span is None or span.end <= i:
            i += 1
            continue
        if math.isfinite(span.value):
            spans.append(span)
        i = span.end
    return spans


def parse_numbers(text: str) -> list[float]:
    """Every number mentioned, in order. 'three slices for 6 dollars' -> [3.0, 6.0]."""
    return [span.value for span in _spans(_tokenize(text))]


# ---------------------------------------------------------------- intents

def _normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("’", "'")).split())


def is_hallucination(text: str) -> bool:
    return _normalized(text) in _HALLUCINATIONS


def detect_intent(text: str) -> Intent | None:
    """REPEAT or DONT_KNOW if the phrase matches, otherwise None."""
    norm = _normalized(text)
    if any(re.search(pattern, norm) for pattern in _REPEAT_PATTERNS):
        return Intent.REPEAT
    if any(re.search(pattern, norm) for pattern in _DONT_KNOW_PATTERNS):
        return Intent.DONT_KNOW
    return None


def parse_answer(text: str | None) -> ParseResult:
    """Parse one spoken answer. See the module docstring for examples."""
    text = (text or "").strip()
    if is_hallucination(text):
        return ParseResult(text, Intent.NONE, reason="silence")

    intent = detect_intent(text)
    if intent is Intent.REPEAT:
        # A repeat request often quotes numbers from the problem; never read them as an answer.
        return ParseResult(text, Intent.REPEAT, confidence=1.0, reason="repeat request")

    tokens = _tokenize(text)
    spans = _spans(tokens)
    if not spans:
        if intent is Intent.DONT_KNOW:
            return ParseResult(text, Intent.DONT_KNOW, confidence=1.0, reason="gave up")
        content = [t for t in tokens if t not in ("um", "uh", "the", "it's", "its", "is", "answer", "it", "i", "think")]
        if content and len(content) <= 2 and content[-1] in _HOMOPHONES:
            value = float(_HOMOPHONES[content[-1]])
            return ParseResult(text, Intent.ANSWER, value, 0.5, (value,), "homophone")
        return ParseResult(text, Intent.UNCLEAR, reason="no number")

    values = tuple(span.value for span in spans)
    hedged = intent is Intent.DONT_KNOW      # "not sure, maybe 14"

    def result(value: float, confidence: float, reason: str) -> ParseResult:
        if hedged:
            confidence = min(confidence, 0.7)
        return ParseResult(text, Intent.ANSWER, value, confidence, values, reason)

    if len(set(values)) == 1:
        return result(values[0], 0.95, "single number")

    # "14 or 15?" is a guess between two numbers: let the LLM or the player resolve it.
    first_end = spans[0].end
    if "or" in tokens[first_end:spans[-1].start]:
        return ParseResult(text, Intent.UNCLEAR, None, 0.0, values, "ambiguous")

    # "14, no wait, 15" -> the number after the last correction cue.
    cue_positions = [k for k, tok in enumerate(tokens) if tok in _CORRECTION_CUES and k >= first_end]
    if cue_positions:
        after = [span for span in spans if span.start > cue_positions[-1]]
        if after and len({span.value for span in after}) == 1:
            return result(after[-1].value, 0.8, "correction")

    # "7 times 2 is 14", "so it's 14" -> the number right after the last answer cue.
    cued = [span for span in spans
            if any(tokens[k] in _ANSWER_CUES for k in range(max(0, span.start - 2), span.start))]
    if cued and cued[-1] is spans[-1]:
        return result(cued[-1].value, 0.75, "answer cue")

    return ParseResult(text, Intent.UNCLEAR, None, 0.0, values, "ambiguous")


# ---------------------------------------------------------------- names

_NAME_PATTERNS = [
    r"\bmy name is ([a-z][a-z' -]*)",
    r"\bmy name's ([a-z][a-z' -]*)",
    r"\bname is ([a-z][a-z' -]*)",
    r"\bcall me ([a-z][a-z' -]*)",
    r"\bi am ([a-z][a-z' -]*)",
    r"\bi'm ([a-z][a-z' -]*)",
    r"\bthis is ([a-z][a-z' -]*)",
    r"\bit's ([a-z][a-z' -]*)",
    r"\bit is ([a-z][a-z' -]*)",
]
_NOT_NAMES = {
    "not", "here", "ready", "sorry", "fine", "good", "okay", "ok", "yes", "no", "a", "an",
    "the", "so", "um", "uh", "and", "going", "gonna", "on", "in", "at", "your", "my",
    "excited", "happy", "red", "blue", "team", "seat", "hi", "hello", "hey", "thank", "thanks",
    "you", "me", "it", "is", "name", "just", "really", "very", "like", "from", "playing",
}
_GREETINGS = {"hi", "hello", "hey", "um", "uh", "so", "yeah", "yes", "okay", "ok", "well", "oh"}


def _clean_name_words(words: list[str], limit: int = 2) -> str | None:
    picked: list[str] = []
    for word in words:
        word = word.strip("'- ")
        if not word or word in _NOT_NAMES or not re.fullmatch(r"[a-z][a-z'-]*", word):
            break
        picked.append(word)
        if len(picked) == limit:
            break
    if not picked:
        return None
    return " ".join(w[:1].upper() + w[1:] for w in picked)


def extract_name(text: str | None) -> str | None:
    """Pull a first name (optionally with surname) out of an introduction."""
    norm = " ".join(re.sub(r"[^a-z' -]+", " ", (text or "").lower().replace("’", "'")).split())
    if not norm or is_hallucination(norm):
        return None
    for pattern in _NAME_PATTERNS:
        match = re.search(pattern, norm)
        if match:
            return _clean_name_words(match.group(1).split())
    words = [w for w in norm.split() if w not in _GREETINGS]
    if 1 <= len(words) <= 2:               # just "Maya" or "Maya Chen"
        return _clean_name_words(words)
    return None


def match_name(text: str, roster: list[str], cutoff: float = 0.75) -> str | None:
    """Fuzzy-match a heard name against known players ('Mia' -> 'Maya' only if close enough)."""
    heard = extract_name(text) or text.strip()
    if not heard or not roster:
        return None
    lookup = {name.lower(): name for name in roster}
    first_names = {name.split()[0].lower(): name for name in roster}
    for candidate in (heard.lower(), heard.split()[0].lower()):
        if candidate in lookup:
            return lookup[candidate]
        if candidate in first_names:
            return first_names[candidate]
    close = difflib.get_close_matches(heard.split()[0].lower(), list(first_names), n=1, cutoff=cutoff)
    return first_names[close[0]] if close else None


# ---------------------------------------------------------------- formatting

def format_number(value: float) -> str:
    """14.0 -> '14', 17.5 -> '17.5', 0.3333 -> '0.33'."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")
