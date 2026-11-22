"""OpenAI API access for Module 2: speech-to-text, text-to-speech, and chat.

One OPENAI_API_KEY, read from the environment or from a shared .env file (repo root or
quizzy-mini/). Standard library only: no `openai` package needed, and every call has
a hard timeout so a dropped connection can never stall the game.

  OpenAITranscriber  POST /v1/audio/transcriptions  (gpt-4o-transcribe, whisper-1, ...)
  OpenAITTS          POST /v1/audio/speech          (gpt-4o-mini-tts, tts-1, ...)
  chat_completion()  POST /v1/chat/completions      (gpt-4o-mini), used by llm.py

Breaker: after 3 failures in a row a service is skipped for 60 s, so the game goes
straight to its fallback (local voice, keyboard input, or a scripted line) instead of
waiting for a timeout on every line.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

API_BASE = "https://api.openai.com/v1"
TTS_SAMPLE_RATE = 24000       # response_format "pcm": 24 kHz, 16-bit signed little-endian, mono


class APIUnavailable(RuntimeError):
    """No key, no network, a timeout, an HTTP error, or the breaker is open."""


# ---------------------------------------------------------------- key and .env

def load_env(paths: list[Path] | None = None) -> None:
    """Read KEY=value lines from .env files into os.environ (existing values win)."""
    here = Path(__file__).resolve()
    candidates = paths or [here.parents[2] / ".env", here.parents[1] / ".env", Path.cwd() / ".env"]
    for path in candidates:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def api_key() -> str | None:
    load_env()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return key or None


# ---------------------------------------------------------------- breaker and HTTP

class Breaker:
    def __init__(self, name: str, failures: int = 3, cooldown: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.name, self.limit, self.cooldown, self.clock = name, failures, cooldown, clock
        self.failures = 0
        self.open_until = 0.0
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        return self.clock() >= self.open_until

    def success(self) -> None:
        with self._lock:
            self.failures = 0

    def failure(self, error: Exception) -> None:
        with self._lock:
            self.failures += 1
            log.warning("OpenAI %s failed (%s)", self.name, error)
            if self.failures >= self.limit:
                self.failures = 0
                self.open_until = self.clock() + self.cooldown
                log.warning("OpenAI %s paused for %.0f s; using the fallback", self.name, self.cooldown)


Transport = Callable[[str, bytes, dict, float], bytes]


def http_post(url: str, body: bytes, headers: dict, timeout: float) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:300]
        raise APIUnavailable(f"HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise APIUnavailable(str(error)) from error


class OpenAIClient:
    """Shared key, base URL and transport. `transport` is swappable for tests."""

    def __init__(self, key: str | None = None, *, base_url: str = API_BASE, transport: Transport | None = None) -> None:
        self.key = key if key is not None else api_key()
        self.base_url = base_url.rstrip("/")
        self.transport = transport or http_post

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def post(self, path: str, body: bytes, content_type: str, timeout: float) -> bytes:
        if not self.key:
            raise APIUnavailable("OPENAI_API_KEY is not set (put it in .env)")
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": content_type}
        return self.transport(f"{self.base_url}{path}", body, headers, timeout)

    def post_json(self, path: str, payload: dict, timeout: float) -> bytes:
        return self.post(path, json.dumps(payload).encode(), "application/json", timeout)


def multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = f"----quizzy{uuid.uuid4().hex}"
    out = io.BytesIO()
    for name, value in fields.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, (filename, data, mime) in files.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                  f"Content-Type: {mime}\r\n\r\n".encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def wav_bytes(samples, rate: int) -> bytes:
    """float32 mono samples -> 16-bit PCM WAV file bytes."""
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return out.getvalue()


# ---------------------------------------------------------------- speech to text

class OpenAITranscriber:
    """Cloud transcription. transcribe() raises APIUnavailable on any failure."""

    def __init__(self, client: OpenAIClient | None = None, *, model: str = "gpt-4o-transcribe",
                 language: str = "en", timeout: float = 2.5) -> None:
        self.client = client or OpenAIClient()
        self.model, self.language, self.timeout = model, language, timeout
        self.breaker = Breaker("speech-to-text")

    @property
    def available(self) -> bool:
        return self.client.configured and self.breaker.closed

    def transcribe(self, clip, prompt: str | None = None) -> str:
        from .stt import SAMPLE_RATE, is_prompt_echo

        if not clip.speech_detected or clip.duration < 0.2:
            return ""
        if not self.breaker.closed:
            raise APIUnavailable("speech-to-text paused after repeated failures")
        fields = {"model": self.model, "language": self.language, "response_format": "json", "temperature": "0"}
        if prompt:
            fields["prompt"] = prompt
        body, content_type = multipart(fields, {"file": ("answer.wav", wav_bytes(clip.samples, SAMPLE_RATE), "audio/wav")})
        started = time.monotonic()
        try:
            reply = json.loads(self.client.post("/audio/transcriptions", body, content_type, self.timeout))
            text = reply.get("text")
            if not isinstance(text, str):
                raise APIUnavailable(f"unexpected reply: {str(reply)[:120]}")
        except (APIUnavailable, ValueError) as error:
            self.breaker.failure(error)
            raise APIUnavailable(str(error)) from error
        self.breaker.success()
        log.debug("Transcribed in %.2fs: %r", time.monotonic() - started, text)
        text = text.strip()
        return "" if is_prompt_echo(text, prompt) else text


# ---------------------------------------------------------------- text to speech

HOST_INSTRUCTIONS = (
    "You are Quizzy, a cheerful robot game-show host for 11 to 13 year olds. "
    "Speak warmly and with energy, clearly and a little slower than normal so numbers are easy to catch."
)


class OpenAITTS:
    """Cloud voice. synthesize() returns (float32 samples, rate); speak() also plays them.

    prepare(text) starts synthesis in the background, so a line is usually ready by the time
    the previous one finishes. Recent lines are cached ("Buzz open!" is fetched once).
    """

    def __init__(self, player, client: OpenAIClient | None = None, *, model: str = "gpt-4o-mini-tts",
                 voice: str = "coral", instructions: str | None = HOST_INSTRUCTIONS, speed: float = 1.0,
                 timeout: float = 6.0, cache_size: int = 64) -> None:
        self.player = player
        self.client = client or OpenAIClient()
        self.model, self.voice, self.instructions, self.speed, self.timeout = model, voice, instructions, speed, timeout
        self.breaker = Breaker("text-to-speech")
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts-fetch")
        self._cache: "OrderedDict[str, Future]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.client.configured and self.breaker.closed

    def _payload(self, text: str) -> dict:
        payload = {"model": self.model, "input": text, "voice": self.voice, "response_format": "pcm"}
        if self.instructions and not self.model.startswith("tts-1"):
            payload["instructions"] = self.instructions          # tts-1 / tts-1-hd do not accept it
        if self.speed != 1.0:
            payload["speed"] = self.speed
        return payload

    def _fetch(self, text: str):
        import numpy as np

        audio = self.client.post_json("/audio/speech", self._payload(text), self.timeout)
        if len(audio) < 2:
            raise APIUnavailable("empty audio")
        samples = np.frombuffer(audio[: len(audio) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        return samples, TTS_SAMPLE_RATE

    def prepare(self, text: str) -> None:
        if not self.available:
            return
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return
            self._cache[text] = self._pool.submit(self._fetch, text)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def synthesize(self, text: str):
        if not self.breaker.closed:
            raise APIUnavailable("text-to-speech paused after repeated failures")
        self.prepare(text)
        with self._lock:
            future = self._cache.get(text)
        if future is None:                    # breaker opened between the two calls
            raise APIUnavailable("text-to-speech unavailable")
        try:
            result = future.result(timeout=self.timeout + 0.5)
        except Exception as error:
            with self._lock:
                self._cache.pop(text, None)   # do not cache failures
            self.breaker.failure(error)
            raise APIUnavailable(str(error)) from error
        self.breaker.success()
        return result

    def speak(self, text: str) -> None:
        samples, rate = self.synthesize(text)
        self.player.play(samples, rate)

    def stop(self) -> None:
        self.player.stop()


# ---------------------------------------------------------------- chat

def chat_completion(client: OpenAIClient, payload: dict, timeout: float) -> str:
    """Returns the reply text or raises APIUnavailable."""
    try:
        data = json.loads(client.post_json("/chat/completions", payload, timeout))
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise APIUnavailable(f"unexpected chat reply ({error})") from error
    if not isinstance(text, str):
        raise APIUnavailable("no text in chat reply")
    return text
