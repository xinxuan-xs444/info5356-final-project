"""OpenAI speech-to-text, text-to-speech and fallbacks, with a fake network (no key, no calls)."""

import io
import json
import os
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from speech.openai_api import (
    APIUnavailable, OpenAIClient, OpenAITranscriber, OpenAITTS, TTS_SAMPLE_RATE, chat_completion,
    load_env, multipart,
)
from speech.speech import SpeechService
from speech.stt import NUMBER_PROMPT, AudioClip, FallbackTranscriber
from speech.tts import FallbackTTS, PrintTTS, make_tts

from .helpers import QuietTTS


def speech_clip(seconds=1.0):
    t = np.arange(int(seconds * 16000)) / 16000
    return AudioClip((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), True)


class FakeNet:
    """Records requests; replies with `reply` (bytes or callable) or raises `error`."""

    def __init__(self, reply=b"{}", error=None):
        self.reply, self.error = reply, error
        self.requests = []
        self.lock = threading.Lock()

    def __call__(self, url, body, headers, timeout):
        with self.lock:
            self.requests.append({"url": url, "body": body, "headers": headers, "timeout": timeout})
        if self.error:
            raise self.error
        return self.reply(url, body) if callable(self.reply) else self.reply


def form_fields(body: bytes) -> dict:
    fields = {}
    for part in body.split(b"--")[1:]:
        if b'name="' not in part:
            continue
        name = part.split(b'name="', 1)[1].split(b'"', 1)[0].decode()
        value = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
        fields[name] = value
    return fields


class EnvFile(unittest.TestCase):
    def test_load_env_reads_key_without_overriding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text('# team key\nexport OPENAI_API_KEY="sk-from-file"\nOTHER=1\n')
            with mock.patch.dict(os.environ, {}, clear=True):
                load_env([path])
                self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-from-file")
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-already-set"}, clear=True):
                load_env([path])
                self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-already-set")

    def test_env_example_exists_and_env_is_ignored(self):
        root = Path(__file__).parents[3]
        self.assertIn("OPENAI_API_KEY=", (root / ".env.example").read_text())
        self.assertIn(".env", (root / ".gitignore").read_text().split())

    def test_no_key_fails_fast(self):
        client = OpenAIClient(key="", transport=FakeNet())
        with self.assertRaises(APIUnavailable):
            client.post_json("/chat/completions", {}, 1)
        self.assertEqual(client.transport.requests, [])


class Transcription(unittest.TestCase):
    def test_request_shape_and_text(self):
        net = FakeNet(json.dumps({"text": " Fourteen dollars. "}).encode())
        stt = OpenAITranscriber(OpenAIClient("sk-test", transport=net), timeout=2.5)
        self.assertEqual(stt.transcribe(speech_clip(), NUMBER_PROMPT), "Fourteen dollars.")
        request = net.requests[0]
        self.assertEqual(request["url"], "https://api.openai.com/v1/audio/transcriptions")
        self.assertEqual(request["headers"]["Authorization"], "Bearer sk-test")
        self.assertTrue(request["headers"]["Content-Type"].startswith("multipart/form-data; boundary="))
        self.assertEqual(request["timeout"], 2.5)
        fields = form_fields(request["body"])
        self.assertEqual(fields["model"], b"gpt-4o-transcribe")
        self.assertEqual(fields["language"], b"en")
        self.assertEqual(fields["response_format"], b"json")
        self.assertEqual(fields["prompt"], NUMBER_PROMPT.encode())
        with wave.open(io.BytesIO(fields["file"])) as wav:          # a real 16 kHz mono WAV
            self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (16000, 1, 2))
            self.assertEqual(wav.getnframes(), 16000)

    def test_silence_is_not_sent(self):
        net = FakeNet()
        stt = OpenAITranscriber(OpenAIClient("sk-test", transport=net))
        self.assertEqual(stt.transcribe(AudioClip(np.zeros(16000, np.float32), False)), "")
        self.assertEqual(net.requests, [])

    def test_prompt_echo_is_dropped(self):
        net = FakeNet(json.dumps({"text": "A student answers a math question with a number."}).encode())
        stt = OpenAITranscriber(OpenAIClient("sk-test", transport=net))
        self.assertEqual(stt.transcribe(speech_clip(), NUMBER_PROMPT), "")

    def test_failures_raise_and_open_the_breaker(self):
        net = FakeNet(error=APIUnavailable("timed out"))
        stt = OpenAITranscriber(OpenAIClient("sk-test", transport=net))
        for _ in range(3):
            with self.assertRaises(APIUnavailable):
                stt.transcribe(speech_clip())
        self.assertFalse(stt.available)
        with self.assertRaises(APIUnavailable):
            stt.transcribe(speech_clip())
        self.assertEqual(len(net.requests), 3)                       # breaker open: no 4th call

    def test_garbage_reply_is_a_failure(self):
        stt = OpenAITranscriber(OpenAIClient("sk-test", transport=FakeNet(b"<html>oops</html>")))
        with self.assertRaises(APIUnavailable):
            stt.transcribe(speech_clip())


class FallbackListening(unittest.TestCase):
    class Engine:
        def __init__(self, text=None, error=None, available=True):
            self.text, self.error, self.available, self.calls = text, error, available, 0

        def transcribe(self, clip, prompt=None):
            self.calls += 1
            if self.error:
                raise self.error
            return self.text

    def test_cloud_first(self):
        cloud, local = self.Engine("fourteen"), self.Engine("forty")
        self.assertEqual(FallbackTranscriber(cloud, local).transcribe(speech_clip()), "fourteen")
        self.assertEqual(local.calls, 0)

    def test_local_whisper_when_cloud_fails(self):
        cloud, local = self.Engine(error=APIUnavailable("offline")), self.Engine("fourteen")
        self.assertEqual(FallbackTranscriber(cloud, local).transcribe(speech_clip()), "fourteen")

    def test_everything_down_means_keyboard(self):
        engine = FallbackTranscriber(self.Engine(error=APIUnavailable("offline")), None)
        with self.assertLogs("speech.stt", level="WARNING") as logs:
            self.assertEqual(engine.transcribe(speech_clip()), "")
        self.assertIn("answer 14", logs.output[0])

    def test_unavailable_cloud_is_skipped(self):
        cloud, local = self.Engine("x", available=False), self.Engine("fourteen")
        self.assertEqual(FallbackTranscriber(cloud, local).transcribe(speech_clip()), "fourteen")
        self.assertEqual(cloud.calls, 0)


class FakePlayer:
    def __init__(self):
        self.played, self.stopped = [], 0

    def play(self, samples, rate):
        self.played.append((samples, rate))

    def stop(self):
        self.stopped += 1


class Voice(unittest.TestCase):
    def pcm(self, n=2400):
        return (np.ones(n) * 8192).astype("<i2").tobytes()

    def test_request_shape_and_audio(self):
        net = FakeNet(self.pcm())
        player = FakePlayer()
        tts = OpenAITTS(player, OpenAIClient("sk-test", transport=net), timeout=6)
        tts.speak("Hello, I'm Quizzy!")
        payload = json.loads(net.requests[0]["body"])
        self.assertEqual(net.requests[0]["url"], "https://api.openai.com/v1/audio/speech")
        self.assertEqual(payload["model"], "gpt-4o-mini-tts")
        self.assertEqual(payload["voice"], "coral")
        self.assertEqual(payload["response_format"], "pcm")
        self.assertEqual(payload["input"], "Hello, I'm Quizzy!")
        self.assertIn("game-show host", payload["instructions"])
        samples, rate = player.played[0]
        self.assertEqual(rate, TTS_SAMPLE_RATE)
        self.assertEqual(len(samples), 2400)
        self.assertAlmostEqual(float(samples[0]), 0.25)

    def test_tts1_gets_no_instructions(self):
        net = FakeNet(self.pcm())
        OpenAITTS(FakePlayer(), OpenAIClient("sk-test", transport=net), model="tts-1").speak("hi")
        self.assertNotIn("instructions", json.loads(net.requests[0]["body"]))

    def test_prepare_fetches_ahead_and_caches_repeats(self):
        net = FakeNet(self.pcm())
        tts = OpenAITTS(FakePlayer(), OpenAIClient("sk-test", transport=net))
        tts.prepare("Buzz open!")
        for _ in range(3):
            tts.speak("Buzz open!")
        self.assertEqual(len(net.requests), 1)

    def test_failure_raises_is_not_cached_and_opens_breaker(self):
        net = FakeNet(error=APIUnavailable("timed out"))
        tts = OpenAITTS(FakePlayer(), OpenAIClient("sk-test", transport=net))
        for _ in range(3):
            with self.assertRaises(APIUnavailable):
                tts.speak("Buzz open!")
        self.assertEqual(len(net.requests), 3)                       # retried, not served from cache
        self.assertFalse(tts.available)

    def test_fallback_voice_says_the_line_when_cloud_fails(self):
        cloud = OpenAITTS(FakePlayer(), OpenAIClient("sk-test", transport=FakeNet(error=APIUnavailable("down"))))
        local = QuietTTS()
        voice = FallbackTTS(cloud, local)
        voice.speak("Correct!")
        self.assertEqual(local.spoken, ["Correct!"])

    def test_fallback_skips_cloud_while_breaker_is_open(self):
        net = FakeNet(error=APIUnavailable("down"))
        cloud = OpenAITTS(FakePlayer(), OpenAIClient("sk-test", transport=net))
        local = QuietTTS()
        voice = FallbackTTS(cloud, local)
        for n in range(6):
            voice.speak(f"line {n}")
        self.assertEqual(len(local.spoken), 6)
        self.assertEqual(len(net.requests), 3)

    def test_service_prefetches_queued_lines(self):
        net = FakeNet(self.pcm())
        cloud = OpenAITTS(FakePlayer(), OpenAIClient("sk-test", transport=net))
        with mock.patch.object(cloud, "prepare", wraps=cloud.prepare) as prepare:
            service = SpeechService(FallbackTTS(cloud, QuietTTS())).start()
            self.addCleanup(service.stop)
            service.say("6 ÷ 2 = 3")
            service.wait_until_quiet(3)
        prepare.assert_any_call("6 divided by 2 equals 3")

    def test_make_tts_without_key_uses_local_voice(self):
        with mock.patch("speech.openai_api.load_env"), mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch("speech.tts.platform.system", return_value="Linux"), \
             mock.patch.dict("sys.modules", {"pyttsx3": None}):
            self.assertIsInstance(make_tts("openai"), PrintTTS)

    def test_make_tts_with_key_builds_cloud_voice_with_fallback(self):
        fake_sd = mock.MagicMock()
        fake_sd.query_devices.return_value = {"default_samplerate": 48000.0}
        with mock.patch("speech.openai_api.load_env"), \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd, "pyttsx3": None}), \
             mock.patch("speech.tts.platform.system", return_value="Linux"):
            voice = make_tts("openai", openai_voice="nova")
        self.assertIsInstance(voice, FallbackTTS)
        self.assertIsInstance(voice.primary, OpenAITTS)
        self.assertEqual(voice.primary.voice, "nova")
        self.assertIsInstance(voice.fallback, PrintTTS)


class Chat(unittest.TestCase):
    def test_reply_text_and_errors(self):
        good = OpenAIClient("sk-test", transport=FakeNet(json.dumps(
            {"choices": [{"message": {"content": "{\"answer\": 14}"}}]}).encode()))
        self.assertEqual(chat_completion(good, {"model": "gpt-4o-mini"}, 4), '{"answer": 14}')
        bad = OpenAIClient("sk-test", transport=FakeNet(b'{"error": "x"}'))
        with self.assertRaises(APIUnavailable):
            chat_completion(bad, {}, 4)

    def test_multipart_encoding(self):
        body, content_type = multipart({"model": "m"}, {"file": ("a.wav", b"RIFF", "audio/wav")})
        boundary = content_type.split("boundary=")[1]
        self.assertTrue(body.startswith(f"--{boundary}\r\n".encode()))
        self.assertTrue(body.endswith(f"--{boundary}--\r\n".encode()))
        self.assertEqual(form_fields(body), {"model": b"m", "file": b"RIFF"})


if __name__ == "__main__":
    unittest.main()
