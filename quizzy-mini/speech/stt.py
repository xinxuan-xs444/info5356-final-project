"""Speech-to-text: microphone capture with simple endpointing, then transcription.

  SoundDeviceRecorder  any mic (laptop, USB, or the Reachy Mini mic array as a USB device)
  ReachyMiniRecorder   the robot mic through the Reachy Mini SDK (mini.media)
  OpenAITranscriber    OpenAI gpt-4o-transcribe / whisper-1 (openai_api.py), the default
  WhisperSTT           local faster-whisper, optional offline backup
  FallbackTranscriber  cloud first, local Whisper if installed, else "" (operator types the answer)
  Listener             recorder + transcriber: listen(seconds) -> text
  ScriptedListener     returns prepared lines; for tests and keyboard demos

Recording stops early once the player has spoken and then been quiet for
config.STT_END_SILENCE seconds, so a quick "fourteen!" is transcribed right away
instead of waiting out the whole answer window.

numpy, sounddevice and faster-whisper are imported lazily, so the rest of the package
(and its tests) works without them.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000          # what Whisper expects
CHUNK_SECONDS = 0.03

# Whisper prompts nudge spelling and vocabulary. They describe the speech rather than quote it,
# because Whisper sometimes "hears" a quoted prompt sentence in noise.
NUMBER_PROMPT = "A student answers a math question with a number, like twelve dollars or 7.5 hours."
REPEAT_PROMPT = "Students talk with each other about a math problem."
NAME_PROMPT = "A student says their first name."


@dataclass
class AudioClip:
    samples: "object"          # numpy float32 mono at SAMPLE_RATE
    speech_detected: bool

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE


class Recorder(Protocol):
    def record(self, max_seconds: float, cancel: threading.Event | None = None) -> AudioClip: ...
    def close(self) -> None: ...


class Transcriber(Protocol):
    def transcribe(self, clip: AudioClip, prompt: str | None = None) -> str: ...


# ---------------------------------------------------------------- endpointing

class Endpointer:
    """Energy-based voice activity detection, fed one chunk at a time.

    The first `calibration` seconds measure the room (players rarely answer within 150 ms of
    the robot finishing). Speech starts when the level stays above the threshold for
    `start_seconds`; the clip ends after `end_silence` seconds below it. The threshold is
    2.5 times the noise floor, never lower than `min_rms`.
    """

    MAX_NOISE = 0.04

    def __init__(self, *, min_rms: float = 0.01, end_silence: float = 0.8, start_seconds: float = 0.09,
                 preroll: float = 0.3, max_speech: float = 8.0, calibration: float = 0.15) -> None:
        self.min_rms = min_rms
        self.end_silence = end_silence
        self.start_seconds = start_seconds
        self.max_speech = max_speech
        self.noise = min_rms / 3
        self.started = False
        self.loud_run = 0.0
        self.quiet_run = 0.0
        self.speech_time = 0.0
        self.calibration = calibration
        self._calib: list[float] = []
        self._elapsed = 0.0
        self._pre = deque(maxlen=max(1, int(preroll / CHUNK_SECONDS)))
        self.chunks: list = []

    @property
    def threshold(self) -> float:
        return max(self.min_rms, self.noise * 2.5)

    def feed(self, chunk, seconds: float) -> bool:
        """Add a mono float chunk. Returns True when the utterance is complete."""
        import numpy as np

        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        self._elapsed += seconds
        if not self.started and self._elapsed <= self.calibration:
            self._calib.append(rms)
            # Capped, so a player who starts talking instantly does not make us deaf.
            self.noise = min(max(self.noise, sum(self._calib) / len(self._calib)), self.MAX_NOISE)
            self._pre.append(chunk)
            return False
        loud = rms >= self.threshold
        if not self.started:
            self._pre.append(chunk)
            if loud:
                self.loud_run += seconds
                if self.loud_run >= self.start_seconds:
                    self.started = True
                    self.chunks.extend(self._pre)
                    self._pre.clear()
            else:
                self.loud_run = 0.0
                self.noise = min(0.95 * self.noise + 0.05 * rms, self.MAX_NOISE)   # keep learning the room
            return False
        self.chunks.append(chunk)
        self.speech_time += seconds
        self.quiet_run = 0.0 if loud else self.quiet_run + seconds
        return self.quiet_run >= self.end_silence or self.speech_time >= self.max_speech

    def clip(self):
        import numpy as np

        if not self.chunks:
            return AudioClip(np.zeros(0, dtype=np.float32), False)
        return AudioClip(np.concatenate(self.chunks).astype(np.float32), self.started)


def resample(samples, rate_in: int, rate_out: int = SAMPLE_RATE):
    import numpy as np

    if rate_in == rate_out or len(samples) == 0:
        return samples.astype(np.float32)
    if rate_in % rate_out == 0:                      # 48 kHz -> 16 kHz: average groups (a gentle low-pass)
        step = rate_in // rate_out
        usable = len(samples) - len(samples) % step
        return samples[:usable].reshape(-1, step).mean(axis=1).astype(np.float32)
    n_out = int(round(len(samples) * rate_out / rate_in))
    x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def _to_mono(block):
    return block.mean(axis=1) if getattr(block, "ndim", 1) == 2 else block


# ---------------------------------------------------------------- recorders

class SoundDeviceRecorder:
    """Records from a sounddevice input. `device` is an index, a name substring, or None."""

    def __init__(self, device: int | str | None = None, *, endpointer: Callable[[], Endpointer] | None = None) -> None:
        import sounddevice as sd

        self._sd = sd
        self.device = find_device(device, kind="input") if device is not None else None
        info = sd.query_devices(self.device, "input")
        self.rate = int(info["default_samplerate"])
        self.name = info["name"]
        self._make_endpointer = endpointer or Endpointer
        log.info("Microphone: %s (%d Hz)", self.name, self.rate)

    def record(self, max_seconds: float, cancel: threading.Event | None = None) -> AudioClip:
        import queue

        blocks: "queue.Queue" = queue.Queue()
        endpointer = self._make_endpointer()
        block_size = max(1, int(self.rate * CHUNK_SECONDS))

        def callback(indata, frames, time_info, status):   # runs on the audio thread
            blocks.put(indata[:, 0].copy())

        deadline = time.monotonic() + max(0.0, max_seconds)
        with self._sd.InputStream(device=self.device, channels=1, samplerate=self.rate,
                                  blocksize=block_size, dtype="float32", callback=callback):
            while time.monotonic() < deadline and not (cancel and cancel.is_set()):
                try:
                    block = blocks.get(timeout=0.1)
                except queue.Empty:
                    continue
                if endpointer.feed(block, len(block) / self.rate):
                    break
        clip = endpointer.clip()
        return AudioClip(resample(clip.samples, self.rate), clip.speech_detected)

    def close(self) -> None:
        pass


class ReachyMiniRecorder:
    """Records from the robot's microphones via the SDK (16 kHz stereo float32 chunks)."""

    def __init__(self, mini, *, endpointer: Callable[[], Endpointer] | None = None) -> None:
        self.media = mini.media
        rate = int(self.media.get_input_audio_samplerate() or 0)
        self.rate = rate if rate > 0 else SAMPLE_RATE
        self._make_endpointer = endpointer or Endpointer
        self.media.start_recording()

    def record(self, max_seconds: float, cancel: threading.Event | None = None) -> AudioClip:
        while self.media.get_audio_sample() is not None:     # drop audio buffered while not listening
            pass
        endpointer = self._make_endpointer()
        deadline = time.monotonic() + max(0.0, max_seconds)
        while time.monotonic() < deadline and not (cancel and cancel.is_set()):
            block = self.media.get_audio_sample()
            if block is None:
                time.sleep(0.01)
                continue
            mono = _to_mono(block)
            if endpointer.feed(mono, len(mono) / self.rate):
                break
        clip = endpointer.clip()
        return AudioClip(resample(clip.samples, self.rate), clip.speech_detected)

    def close(self) -> None:
        try:
            self.media.stop_recording()
        except Exception:
            pass


def find_device(name_or_index: int | str, kind: str = "input") -> int:
    """Resolve 'Reachy' to the first matching device index."""
    import sounddevice as sd

    if isinstance(name_or_index, int):
        return name_or_index
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for index, device in enumerate(sd.query_devices()):
        if name_or_index.lower() in device["name"].lower() and device[key] > 0:
            return index
    raise RuntimeError(f"No {kind} device matching {name_or_index!r}. Run: python -m speech.demo devices")


def list_devices() -> str:
    import sounddevice as sd

    return str(sd.query_devices())


# ---------------------------------------------------------------- whisper

class WhisperSTT:
    """Local Whisper. Downloads the model on first use (base.en is about 140 MB)."""

    def __init__(self, model: str = "base.en", *, compute_type: str = "int8", language: str = "en") -> None:
        self.model_name = model
        self.compute_type = compute_type
        self.language = language
        self._model = None
        self._backend = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return True

    def load(self) -> "WhisperSTT":
        with self._lock:
            if self._model is not None:
                return self
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(self.model_name, device="cpu", compute_type=self.compute_type)
                self._backend = "faster-whisper"
            except ImportError:
                import whisper  # openai-whisper

                self._model = whisper.load_model(self.model_name)
                self._backend = "openai-whisper"
            log.info("Whisper ready: %s via %s", self.model_name, self._backend)
        return self

    def warm_up(self) -> None:
        """Run once on silence so the first real answer is not slowed by lazy setup."""
        import numpy as np

        self.transcribe(AudioClip(np.zeros(SAMPLE_RATE // 2, dtype=np.float32), True))

    def transcribe(self, clip: AudioClip, prompt: str | None = None) -> str:
        if not clip.speech_detected or clip.duration < 0.2:
            return ""
        self.load()
        with self._lock:
            if self._backend == "faster-whisper":
                segments, _ = self._model.transcribe(
                    clip.samples, language=self.language, beam_size=5, temperature=0.0,
                    condition_on_previous_text=False, initial_prompt=prompt,
                    vad_filter=False, without_timestamps=True,
                )
                parts = [s.text for s in segments if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
                text = " ".join(p.strip() for p in parts).strip()
            else:
                result = self._model.transcribe(
                    clip.samples, language=self.language, fp16=False, temperature=0.0,
                    condition_on_previous_text=False, initial_prompt=prompt,
                )
                text = str(result.get("text", "")).strip()
        return "" if is_prompt_echo(text, prompt) else text


def is_prompt_echo(text: str, prompt: str | None) -> bool:
    """True when Whisper just repeated (part of) its prompt, a known failure on noise."""
    if not text or not prompt:
        return False
    word = r"[a-z0-9]+(?:[.'][a-z0-9]+)*"          # keeps 7.5 and it's, drops sentence periods
    words = re.findall(word, text.lower())
    return len(words) >= 3 and f" {' '.join(words)} " in f" {' '.join(re.findall(word, prompt.lower()))} "


class FallbackTranscriber:
    """Try the cloud first, then local Whisper (if installed). If both fail, return "" and warn
    the operator to type the answer: the game's own timer keeps running, so nothing stalls."""

    def __init__(self, primary: Transcriber, fallback: Transcriber | None = None) -> None:
        self.primary, self.fallback = primary, fallback
        self._warned_at = -1e9

    def transcribe(self, clip: AudioClip, prompt: str | None = None) -> str:
        if not clip.speech_detected or clip.duration < 0.2:
            return ""
        errors = []
        for engine in (self.primary, self.fallback):
            if engine is None or not getattr(engine, "available", True):
                continue
            try:
                return engine.transcribe(clip, prompt)
            except Exception as error:   # network, timeout, model problems
                errors.append(f"{type(engine).__name__}: {error}")
        if time.monotonic() - self._warned_at > 10:
            self._warned_at = time.monotonic()
            log.warning("SPEECH RECOGNITION OFFLINE (%s). Type the answer in the console, e.g. `answer 14`.",
                        "; ".join(errors) or "no engine available")
        return ""

    def load(self) -> "FallbackTranscriber":
        load = getattr(self.fallback, "load", None)
        if load is not None:
            try:
                load()
            except Exception as error:
                log.info("Local Whisper backup not available (%s)", error)
                self.fallback = None
        return self

    def warm_up(self) -> None:
        warm = getattr(self.fallback, "warm_up", None)
        if warm is not None:
            warm()


# ---------------------------------------------------------------- listeners

@dataclass(frozen=True)
class Heard:
    text: str
    audio_seconds: float = 0.0
    transcribe_seconds: float = 0.0


class Listener:
    def __init__(self, recorder: Recorder, transcriber: Transcriber) -> None:
        self.recorder = recorder
        self.transcriber = transcriber

    def listen(self, max_seconds: float, prompt: str | None = None,
               cancel: threading.Event | None = None) -> Heard:
        clip = self.recorder.record(max_seconds, cancel)
        if cancel is not None and cancel.is_set():
            return Heard("", clip.duration)
        started = time.monotonic()
        try:
            text = self.transcriber.transcribe(clip, prompt)
        except Exception as error:   # model or audio failure: behave like silence
            log.warning("Transcription failed: %s", error)
            text = ""
        heard = Heard(text, clip.duration, time.monotonic() - started)
        if text:
            log.info("Heard %r (%.1fs audio, %.2fs to transcribe)", text, heard.audio_seconds, heard.transcribe_seconds)
        return heard

    def close(self) -> None:
        self.recorder.close()


class ScriptedListener:
    """Plays back prepared transcripts in order. A None entry (or running out) means nothing was heard."""

    def __init__(self, lines: Iterable[str | None] = (), *, delay: float = 0.0) -> None:
        self.lines = deque(lines)
        self.delay = delay
        self.calls: list[tuple[float, str | None]] = []
        self._lock = threading.Lock()

    def add(self, *lines: str | None) -> None:
        with self._lock:
            self.lines.extend(lines)

    def listen(self, max_seconds: float, prompt: str | None = None,
               cancel: threading.Event | None = None) -> Heard:
        with self._lock:
            line = self.lines.popleft() if self.lines else None
            self.calls.append((max_seconds, prompt))
        wait = self.delay if line is not None else min(max_seconds, max(self.delay, 0.05))
        if cancel is not None:
            if cancel.wait(wait):
                return Heard("")
        elif wait:
            time.sleep(wait)
        return Heard(line or "")

    def close(self) -> None:
        pass


def make_listener(*, mini=None, device: str | int | None = None, engine: str = "openai",
                  openai_model: str = "gpt-4o-transcribe", local_model: str = "base.en",
                  compute_type: str = "int8", timeout: float = 2.5, min_rms: float = 0.01,
                  end_silence: float = 0.8) -> Listener:
    """Mic (robot SDK if `mini`, else sounddevice) + transcriber chosen by `engine`:
    "openai" (cloud, local Whisper as backup if installed) or "local" (offline Whisper only)."""
    def endpointer() -> Endpointer:
        return Endpointer(min_rms=min_rms, end_silence=end_silence)

    if mini is not None and device is None:
        recorder: Recorder = ReachyMiniRecorder(mini, endpointer=endpointer)
    else:
        recorder = SoundDeviceRecorder(device, endpointer=endpointer)

    local = WhisperSTT(local_model, compute_type=compute_type)
    if (engine or "openai").lower() == "local":
        return Listener(recorder, local)

    from .openai_api import OpenAITranscriber

    cloud = OpenAITranscriber(model=openai_model, timeout=timeout)
    if not cloud.client.configured:
        log.warning("OPENAI_API_KEY not set (see .env.example); trying local Whisper")
    try:
        import faster_whisper  # noqa: F401
        backup: Transcriber | None = local
    except ImportError:
        backup = None
    return Listener(recorder, FallbackTranscriber(cloud, backup))
