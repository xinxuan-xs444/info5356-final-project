"""Text-to-speech engines. Each one has speak(text) (blocks until done) and stop().

  OpenAITTS      OpenAI gpt-4o-mini-tts game-show voice (openai_api.py), played by a player below
  MacSayTTS      macOS `say`; can route to the Reachy Mini speaker with audio_device="Reachy"
  ReachyMiniTTS  synthesizes with `say`, plays through the Reachy Mini SDK (mini.media)
  Pyttsx3TTS     Windows / Linux voice (pip install pyttsx3)
  PrintTTS       prints the line, so the operator can read it out if all audio fails
  FallbackTTS    tries the OpenAI voice, falls back to a local one line by line

Players: SoundDevicePlayer (any speaker, by name) and ReachyMiniPlayer (robot speaker via SDK).

make_tts() builds the chain from config.py. speakable() rewrites math symbols so the
voice reads "4 × 3 = 12" as "4 times 3 equals 12".
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading

import wave
from typing import Protocol

log = logging.getLogger(__name__)


class TTSEngine(Protocol):
    def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...


_SYMBOLS = [
    (r"(?<=\d)\s*[×xX*]\s*(?=\d)", " times "),
    (r"×", " times "),
    (r"÷", " divided by "),
    (r"(?<=\d)\s*/\s*(?=\d)", " over "),
    (r"(?<=\d):(?=\d)", " to "),                  # 3:5, but not "Step 1: 18 ..."
    (r"=", " equals "),
    (r"−", " minus "),
    (r"(?<=\d)\s*\+\s*(?=\d)", " plus "),
    (r"(?<=\d)\s*-\s*(?=\d)", " minus "),
    (r"%", " percent"),
    (r"→", " then "),
    (r"\$\s*(\d[\d,]*(?:\.\d+)?)", r"\1 dollars"),
]


def speakable(text: str) -> str:
    for pattern, replacement in _SYMBOLS:
        text = re.sub(pattern, replacement, text)
    return " ".join(text.split())


class SoundDevicePlayer:
    """Plays float32 audio on a sounddevice output (None = default, or a name substring like "Reachy")."""

    def __init__(self, device: int | str | None = None) -> None:
        import sounddevice as sd

        from .stt import find_device

        self._sd = sd
        self.device = find_device(device, kind="output") if device is not None else None
        self.rate = int(sd.query_devices(self.device, "output")["default_samplerate"])
        self._stop = threading.Event()

    def play(self, samples, rate: int) -> None:
        from .stt import resample

        self._stop.clear()
        audio = resample(samples, rate, self.rate)
        block = max(1, self.rate // 20)
        with self._sd.OutputStream(device=self.device, samplerate=self.rate, channels=1, dtype="float32") as stream:
            for start in range(0, len(audio), block):
                if self._stop.is_set():
                    return
                stream.write(audio[start:start + block].reshape(-1, 1))

    def stop(self) -> None:
        self._stop.set()


class ReachyMiniPlayer:
    """Plays float32 audio through the robot speaker (`mini` is a connected ReachyMini)."""

    def __init__(self, mini) -> None:
        self.media = mini.media
        self._stop = threading.Event()
        self._started = False

    def play(self, samples, rate: int) -> None:
        from .stt import resample

        self._stop.clear()
        out_rate = self.media.get_output_audio_samplerate()
        if out_rate and out_rate > 0 and out_rate != rate:
            samples, rate = resample(samples, rate, out_rate), out_rate
        if not self._started:
            self.media.start_playing()
            self._started = True
        self.media.push_audio_sample(samples)
        # push_audio_sample returns at once; wait for playback (interruptible) plus a small tail.
        self._stop.wait(len(samples) / rate + 0.15)

    def stop(self) -> None:
        self._stop.set()
        try:
            self.media.clear_player()
        except Exception:  # older SDKs: nothing queued to clear
            pass


class FallbackTTS:
    """Speak with `primary` (the OpenAI voice); if it fails, say the same line with `fallback`."""

    def __init__(self, primary, fallback) -> None:
        self.primary, self.fallback = primary, fallback
        self._current = None

    def prepare(self, text: str) -> None:
        prepare = getattr(self.primary, "prepare", None)
        if prepare is not None:
            prepare(text)

    def speak(self, text: str) -> None:
        if getattr(self.primary, "available", True):
            try:
                self._current = self.primary
                self.primary.speak(text)
                return
            except Exception as error:   # timeout, network, HTTP error: use the local voice for this line
                log.info("Cloud voice unavailable (%s); using the local voice", error)
        self._current = self.fallback
        self.fallback.speak(text)

    def stop(self) -> None:
        for engine in (self.primary, self.fallback):
            try:
                engine.stop()
            except Exception:
                pass


class PrintTTS:
    """Prints instead of speaking. seconds_per_word > 0 simulates speaking time."""

    def __init__(self, seconds_per_word: float = 0.0, prefix: str = "[Quizzy] ") -> None:
        self.seconds_per_word = seconds_per_word
        self.prefix = prefix
        self.spoken: list[str] = []
        self._stop = threading.Event()

    def speak(self, text: str) -> None:
        self._stop.clear()
        self.spoken.append(text)
        print(f"{self.prefix}{text}", flush=True)
        if self.seconds_per_word:
            self._stop.wait(self.seconds_per_word * len(text.split()))

    def stop(self) -> None:
        self._stop.set()


class MacSayTTS:
    """macOS built-in voices. audio_device is an output device name prefix, e.g. "Reachy"."""

    def __init__(self, voice: str | None = "Samantha", rate: int = 165, audio_device: str | None = None) -> None:
        if not shutil.which("say"):
            raise RuntimeError("`say` not found: MacSayTTS needs macOS")
        self.voice, self.rate, self.audio_device = voice, rate, audio_device
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def command(self) -> list[str]:
        cmd = ["say", "-r", str(self.rate), "-f", "-"]
        if self.voice:
            cmd += ["-v", self.voice]
        if self.audio_device:
            cmd += ["-a", self.audio_device]
        return cmd

    def speak(self, text: str) -> None:
        with self._lock:
            self._proc = subprocess.Popen(self.command(), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            proc = self._proc
        try:
            _, err = proc.communicate(text.encode("utf-8"))
            if proc.returncode not in (0, -15) and err:   # -15: stopped by us
                log.warning("say failed: %s", err.decode(errors="replace").strip())
        finally:
            with self._lock:
                self._proc = None

    def stop(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()


class ReachyMiniTTS:
    """Speaks through the robot's own speaker using the Reachy Mini SDK.

    `mini` is a connected reachy_mini.ReachyMini (share it with vision and motion).
    Synthesis uses macOS `say` to a 16 kHz WAV, then the samples are pushed to mini.media.
    """

    SAMPLE_RATE = 16000

    def __init__(self, mini, voice: str | None = "Samantha", rate: int = 165) -> None:
        if not shutil.which("say"):
            raise RuntimeError("ReachyMiniTTS synthesizes with macOS `say`; use MacSayTTS(audio_device=...) elsewhere")
        self.player = ReachyMiniPlayer(mini)
        self.voice, self.rate = voice, rate

    def _synthesize(self, text: str):
        import numpy as np

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            cmd = ["say", "-r", str(self.rate), "-o", path, "--file-format=WAVE",
                   f"--data-format=LEI16@{self.SAMPLE_RATE}", "-f", "-"]
            if self.voice:
                cmd += ["-v", self.voice]
            subprocess.run(cmd, input=text.encode("utf-8"), check=True, capture_output=True)
            with wave.open(path, "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                rate = wav.getframerate()
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return samples, rate
        finally:
            os.remove(path)

    def speak(self, text: str) -> None:
        samples, rate = self._synthesize(text)
        self.player.play(samples, rate)

    def stop(self) -> None:
        self.player.stop()


class Pyttsx3TTS:
    """Cross-platform fallback. The engine lives on the thread that first speaks."""

    def __init__(self, rate: int = 165, voice: str | None = None) -> None:
        import pyttsx3  # noqa: F401  (fail early if missing)

        self.rate, self.voice = rate, voice
        self._engine = None

    def speak(self, text: str) -> None:
        import pyttsx3

        if self._engine is None:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            if self.voice:
                self._engine.setProperty("voice", self.voice)
        self._engine.say(text)
        self._engine.runAndWait()

    def stop(self) -> None:
        if self._engine is not None:
            self._engine.stop()


def make_local_tts(kind: str = "auto", *, voice: str | None = None, rate: int = 165,
                   device: str | None = None, mini=None) -> TTSEngine:
    """An offline voice: `say` (macOS, robot speaker if `mini`), pyttsx3, or print."""
    kind = (kind or "auto").lower()
    is_mac = platform.system() == "Darwin" and shutil.which("say") is not None
    try:
        if kind == "reachy" or (kind == "auto" and mini is not None and is_mac and device is None):
            return ReachyMiniTTS(mini, voice=voice, rate=rate)
        if kind in ("say", "auto") and is_mac:
            return MacSayTTS(voice=voice, rate=rate, audio_device=device)
        if kind in ("pyttsx3", "auto"):
            return Pyttsx3TTS(rate=rate)
    except Exception as error:  # missing package, no robot, not a Mac
        log.warning("Local voice %r unavailable (%s); printing lines instead", kind, error)
    if kind not in ("print", "auto"):
        log.warning("Unknown or unavailable local voice %r; printing lines instead", kind)
    return PrintTTS()


def make_tts(kind: str = "openai", *, voice: str | None = None, rate: int = 165, device: str | None = None,
             mini=None, fallback: str = "auto", openai_model: str = "gpt-4o-mini-tts",
             openai_voice: str = "coral", instructions: str | None = None, timeout: float = 6.0) -> TTSEngine:
    """Build the voice from config.TTS_ENGINE.

    "openai": the cloud voice with a local voice as fallback (the default).
    "say" / "reachy" / "pyttsx3" / "print" / "auto": local voices only.
    """
    kind = (kind or "openai").lower()
    local = make_local_tts(fallback if kind == "openai" else kind, voice=voice, rate=rate, device=device, mini=mini)
    if kind != "openai":
        return local
    from .openai_api import HOST_INSTRUCTIONS, OpenAIClient, OpenAITTS

    client = OpenAIClient()
    if not client.configured:
        log.warning("OPENAI_API_KEY not set (see .env.example); using the local voice")
        return local
    try:
        player = ReachyMiniPlayer(mini) if mini is not None and device is None else SoundDevicePlayer(device)
    except Exception as error:   # sounddevice missing or no such speaker
        log.warning("Cannot play cloud audio (%s); using the local voice", error)
        return local
    cloud = OpenAITTS(player, client, model=openai_model, voice=openai_voice,
                      instructions=instructions if instructions is not None else HOST_INSTRUCTIONS, timeout=timeout)
    return FallbackTTS(cloud, local)


def time_to_say(text: str, rate_wpm: int = 165) -> float:
    """Rough speaking time in seconds, for planning timers."""
    return 60.0 * len(text.split()) / max(rate_wpm, 1)


__all__ = ["TTSEngine", "PrintTTS", "MacSayTTS", "ReachyMiniTTS", "Pyttsx3TTS", "FallbackTTS", "SoundDevicePlayer",
           "ReachyMiniPlayer", "make_tts", "make_local_tts", "speakable", "time_to_say"]

