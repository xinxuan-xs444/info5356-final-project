"""Optional LLM helper. Text in, text out; never audio, never judging.

Two jobs, both with scripted fallbacks in the callers:
  1. interpret_answer(): pull the final number out of a messy transcript
     ("it's 14 dollars for 7 slices") when the rule-based parser is unsure.
  2. explain(): one or two kid-friendly sentences about a wrong answer.

Works with any OpenAI-compatible chat endpoint:
  * Ollama on the laptop (nothing leaves the machine): `ollama run llama3.2:3b`
  * OpenAI: set OPENAI_API_KEY in the environment.

Every call has a hard timeout. After repeated failures the helper switches itself off for a
minute so a dead network never slows the game down. Uses only the standard library.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request
from typing import Callable

log = logging.getLogger(__name__)

PROVIDERS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3.2:3b", "key_env": None},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY"},
}

_INTERPRET_SYSTEM = (
    "You extract a middle-school student's final numeric answer from a speech transcript. "
    "The transcript may contain filler words, units, working steps, or self-corrections. "
    "Reply with JSON only: {\"answer\": <number>} for the single final answer the student commits to, "
    "or {\"answer\": null} if they did not commit to exactly one number "
    "(for example 'is it 14 or 15?'). Do not solve the problem yourself and do not judge correctness."
)
_EXPLAIN_SYSTEM = (
    "You are Quizzy, a friendly robot game host for 11-13 year olds practicing ratios. "
    "In at most two short sentences (under 35 words total), explain kindly why the student's answer "
    "does not fit and point to the key step. Plain words, no lists, no markdown, no emojis. "
    "Do not mention the student's name."
)


class LLMHelper:
    def __init__(
        self,
        base_url: str | None,
        model: str | None,
        *,
        api_key: str | None = None,
        timeout: float = 4.0,
        transport: Callable[[str, dict, dict, float], dict] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._transport = transport or _http_post_json
        self._clock = clock
        self._failures = 0
        self._disabled_until = 0.0

    # ------------------------------------------------------------ setup

    @classmethod
    def disabled(cls) -> "LLMHelper":
        return cls(None, None)

    @classmethod
    def from_config(cls, provider: str = "auto", model: str | None = None, timeout: float = 4.0) -> "LLMHelper":
        provider = (provider or "none").lower()
        if provider == "auto":
            if _ollama_running(PROVIDERS["ollama"]["base_url"]):
                provider = "ollama"
            elif os.environ.get("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "none"
        if provider not in PROVIDERS:
            if provider != "none":
                log.warning("Unknown LLM provider %r; LLM features are off", provider)
            return cls.disabled()
        spec = PROVIDERS[provider]
        key = os.environ.get(spec["key_env"]) if spec["key_env"] else None
        if spec["key_env"] and not key:
            log.warning("%s is not set; LLM features are off", spec["key_env"])
            return cls.disabled()
        helper = cls(spec["base_url"], model or os.environ.get("LLM_MODEL") or spec["model"],
                     api_key=key, timeout=timeout)
        log.info("LLM helper: %s (%s)", provider, helper.model)
        return helper

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.model) and self._clock() >= self._disabled_until

    # ------------------------------------------------------------ jobs

    def interpret_answer(self, question: str | None, transcript: str, *, timeout: float | None = None) -> float | None:
        """The number the student committed to, or None. Never raises."""
        if not transcript.strip():
            return None
        user = f"Problem: {question or '(not given)'}\nTranscript: {transcript.strip()}"
        reply = self._chat(_INTERPRET_SYSTEM, user, max_tokens=20, timeout=timeout)
        if reply is None:
            return None
        return _parse_answer_json(reply)

    def explain(self, question: dict, student_answer: float | None) -> str | None:
        """A short explanation of the mistake, or None. Never raises."""
        steps = " ".join(question.get("solution", []))
        user = (
            f"Problem: {question.get('question')}\n"
            f"Correct answer: {question.get('answer')} {question.get('unit', '')}\n"
            f"Worked solution: {steps}\n"
            f"Student answered: {student_answer if student_answer is not None else 'nothing clear'}"
        )
        reply = self._chat(_EXPLAIN_SYSTEM, user, max_tokens=80)
        if not reply:
            return None
        reply = " ".join(reply.replace("*", "").split())
        sentences = re.split(r"(?<=[.!?])\s+", reply)
        reply = " ".join(sentences[:2]).strip()
        return reply if 0 < len(reply) <= 300 else None

    # ------------------------------------------------------------ plumbing

    def _chat(self, system: str, user: str, *, max_tokens: int, timeout: float | None = None) -> str | None:
        if not self.available:
            return None
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        started = self._clock()
        try:
            data = self._transport(f"{self.base_url}/chat/completions", body, headers,
                                   self.timeout if timeout is None else timeout)
            text = data["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("no text in reply")
        except Exception as error:  # network, HTTP, JSON or shape problems: all fall back
            self._failures += 1
            log.warning("LLM call failed (%s); using scripted fallback", error)
            if self._failures >= 3:
                self._disabled_until = self._clock() + 60
                self._failures = 0
                log.warning("LLM paused for 60 s after repeated failures")
            return None
        self._failures = 0
        log.debug("LLM reply in %.2fs: %r", self._clock() - started, text)
        return text.strip()


def _parse_answer_json(reply: str) -> float | None:
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0)).get("answer")
    except (json.JSONDecodeError, AttributeError):
        return None
    if isinstance(value, str):
        try:
            value = float(value.replace(",", ""))
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _http_post_json(url: str, body: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _ollama_running(base_url: str) -> bool:
    root = base_url.rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=0.5) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False
