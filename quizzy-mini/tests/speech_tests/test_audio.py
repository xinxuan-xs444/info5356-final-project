"""TTS text cleanup, endpointing, resampling and recorder/transcriber plumbing (no real devices)."""

import threading
import unittest
from unittest import mock

import numpy as np

from speech import stt
from speech.stt import AudioClip, Endpointer, Listener, ScriptedListener, WhisperSTT, resample
from speech.tts import MacSayTTS, PrintTTS, make_tts, speakable

RATE = stt.SAMPLE_RATE


def tone(seconds, level=0.2, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds, level=0.001, rate=RATE):
    return (level * np.random.default_rng(0).standard_normal(int(seconds * rate))).astype(np.float32)


def feed_all(endpointer, audio, rate=RATE):
    step = int(rate * stt.CHUNK_SECONDS)
    for start in range(0, len(audio), step):
        chunk = audio[start:start + step]
        if endpointer.feed(chunk, len(chunk) / rate):
            return start + step
    return None


class SpeakableText(unittest.TestCase):
    def test_math_symbols(self):
        self.assertEqual(speakable("Three packs contain 4 × 3 = 12 pens."),
                         "Three packs contain 4 times 3 equals 12 pens.")
        self.assertEqual(speakable("6 ÷ 2 = 3 batches."), "6 divided by 2 equals 3 batches.")
        self.assertEqual(speakable("120 − 30 = 90"), "120 minus 30 equals 90")
        self.assertEqual(speakable("2 + 3 = 5 parts"), "2 plus 3 equals 5 parts")
        self.assertEqual(speakable("$14 and 25%"), "14 dollars and 25 percent")
        self.assertEqual(speakable("a ratio of 3:5"), "a ratio of 3 to 5")
        self.assertEqual(speakable("A 750-gram trail mix"), "A 750-gram trail mix")

    def test_every_bank_line_becomes_plain_words(self):
        import json
        from pathlib import Path
        questions = json.loads((Path(__file__).parents[2] / "game" / "questions.json").read_text())
        for q in questions:
            for line in [q["question"], q["hint"], q["mini_question"], *q["solution"]]:
                self.assertNotRegex(speakable(line), r"[×÷=−+]", line)


class Engines(unittest.TestCase):
    def test_print_tts_records_lines(self):
        tts = PrintTTS()
        with mock.patch("builtins.print"):
            tts.speak("hello")
        self.assertEqual(tts.spoken, ["hello"])

    def test_print_tts_stop_cuts_simulated_speech(self):
        tts = PrintTTS(seconds_per_word=10)
        timer = threading.Timer(0.05, tts.stop)
        timer.start()
        with mock.patch("builtins.print"):
            tts.speak("a b c")      # would take 30 s without stop()
        timer.join()

    def test_make_tts_falls_back_to_print(self):
        with mock.patch("speech.tts.platform.system", return_value="Linux"), \
             mock.patch.dict("sys.modules", {"pyttsx3": None}):
            self.assertIsInstance(make_tts("auto"), PrintTTS)
            self.assertIsInstance(make_tts("say"), PrintTTS)
        self.assertIsInstance(make_tts("print"), PrintTTS)

    def test_say_command_routes_to_device(self):
        with mock.patch("speech.tts.shutil.which", return_value="/usr/bin/say"):
            tts = MacSayTTS(voice="Samantha", rate=150, audio_device="Reachy")
        self.assertEqual(tts.command(), ["say", "-r", "150", "-f", "-", "-v", "Samantha", "-a", "Reachy"])


class Endpointing(unittest.TestCase):
    def test_stops_after_trailing_silence(self):
        ep = Endpointer(min_rms=0.01, end_silence=0.5)
        audio = np.concatenate([silence(0.5), tone(1.0), silence(2.0)])
        stop_at = feed_all(ep, audio)
        self.assertIsNotNone(stop_at)
        self.assertAlmostEqual(stop_at / RATE, 2.0, delta=0.1)     # 0.5 lead + 1.0 speech + 0.5 tail
        clip = ep.clip()
        self.assertTrue(clip.speech_detected)
        self.assertGreater(clip.duration, 1.0)                      # includes pre-roll and tail

    def test_silence_only_is_not_speech(self):
        ep = Endpointer(min_rms=0.01)
        self.assertIsNone(feed_all(ep, silence(2.0)))
        clip = ep.clip()
        self.assertFalse(clip.speech_detected)
        self.assertEqual(clip.duration, 0)

    def test_short_click_is_ignored(self):
        ep = Endpointer(min_rms=0.01)
        feed_all(ep, np.concatenate([silence(0.3), tone(0.03), silence(1.0)]))
        self.assertFalse(ep.clip().speech_detected)

    def test_threshold_adapts_to_noisy_room(self):
        ep = Endpointer(min_rms=0.01)
        feed_all(ep, silence(1.0, level=0.02))     # fan noise above min_rms
        self.assertGreater(ep.threshold, 0.02)
        self.assertFalse(ep.started)

    def test_player_who_answers_instantly_is_still_heard(self):
        ep = Endpointer(min_rms=0.01, end_silence=0.5)
        self.assertIsNotNone(feed_all(ep, np.concatenate([tone(1.0), silence(1.0)])))
        self.assertTrue(ep.clip().speech_detected)

    def test_max_speech_caps_long_talk(self):
        ep = Endpointer(min_rms=0.01, max_speech=1.0)
        stop_at = feed_all(ep, tone(5.0))
        self.assertAlmostEqual(stop_at / RATE, 1.25, delta=0.1)   # calibration + start + 1 s


class Resampling(unittest.TestCase):
    def test_integer_and_fractional_ratios(self):
        self.assertEqual(len(resample(np.ones(48000, dtype=np.float32), 48000)), 16000)
        self.assertEqual(len(resample(np.ones(44100, dtype=np.float32), 44100)), 16000)
        same = np.ones(100, dtype=np.float32)
        self.assertEqual(len(resample(same, RATE)), 100)


class ListenerPlumbing(unittest.TestCase):
    def test_listener_returns_text_and_survives_transcriber_errors(self):
        class Rec:
            def record(self, seconds, cancel=None):
                return AudioClip(tone(1.0), True)

            def close(self):
                pass

        class Good:
            def transcribe(self, clip, prompt=None):
                return "fourteen"

        class Broken:
            def transcribe(self, clip, prompt=None):
                raise RuntimeError("model crashed")

        self.assertEqual(Listener(Rec(), Good()).listen(3).text, "fourteen")
        self.assertEqual(Listener(Rec(), Broken()).listen(3).text, "")

    def test_whisper_skips_silence_without_loading_a_model(self):
        whisper = WhisperSTT("tiny.en")
        self.assertEqual(whisper.transcribe(AudioClip(np.zeros(16000, dtype=np.float32), False)), "")
        self.assertIsNone(whisper._model)

    def test_prompt_echo_is_dropped(self):
        from speech.stt import NUMBER_PROMPT, is_prompt_echo
        self.assertTrue(is_prompt_echo("A student answers a math question.", NUMBER_PROMPT))
        self.assertFalse(is_prompt_echo("twelve dollars", NUMBER_PROMPT))       # a real short answer
        self.assertFalse(is_prompt_echo("I think it's fourteen", NUMBER_PROMPT))
        self.assertFalse(is_prompt_echo("fourteen", None))

    def test_scripted_listener(self):
        listener = ScriptedListener(["fourteen", None])
        self.assertEqual(listener.listen(5, "prompt").text, "fourteen")
        self.assertEqual(listener.listen(5).text, "")
        self.assertEqual(listener.listen(5).text, "")
        cancel = threading.Event()
        cancel.set()
        listener.add("never")
        self.assertEqual(listener.listen(5, cancel=cancel).text, "")


if __name__ == "__main__":
    unittest.main()


class FakeDevices(unittest.TestCase):
    """Exercise the real recorder / robot code paths against fake hardware."""

    def test_sounddevice_recorder_records_resamples_and_stops_at_silence(self):
        audio = np.concatenate([silence(0.3, rate=48000), tone(0.8, rate=48000), silence(3.0, rate=48000)])

        class FakeStream:
            def __init__(self, device, channels, samplerate, blocksize, dtype, callback):
                self.callback, self.blocksize = callback, blocksize
                self.stop = threading.Event()

            def __enter__(self):
                def run():
                    for start in range(0, len(audio), self.blocksize):
                        if self.stop.is_set():
                            return
                        self.callback(audio[start:start + self.blocksize].reshape(-1, 1), self.blocksize, None, None)
                threading.Thread(target=run, daemon=True).start()
                return self

            def __exit__(self, *exc):
                self.stop.set()

        fake_sd = mock.MagicMock()
        fake_sd.query_devices.side_effect = lambda *a, **k: (
            {"name": "Reachy Mini Audio", "default_samplerate": 48000.0, "max_input_channels": 2}
            if a else [{"name": "MacBook Mic", "max_input_channels": 1}, {"name": "Reachy Mini Audio", "max_input_channels": 2}])
        fake_sd.InputStream = FakeStream
        with mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            recorder = stt.SoundDeviceRecorder("reachy")
            self.assertEqual(recorder.device, 1)
            clip = recorder.record(5.0)
        self.assertTrue(clip.speech_detected)
        self.assertGreater(clip.duration, 0.8)
        self.assertLess(clip.duration, 2.0)          # stopped at the pause, not after 5 s
        self.assertEqual(clip.samples.dtype, np.float32)

    def test_reachy_recorder_uses_sdk_chunks(self):
        stereo = np.stack([tone(1.0), tone(1.0)], axis=1)
        chunks = [stereo[i:i + 512] for i in range(0, len(stereo), 512)] + [np.zeros((512, 2), np.float32)] * 60
        media = mock.MagicMock()
        media.get_input_audio_samplerate.return_value = 16000
        feed = iter([None] + chunks)                 # first None ends the stale-audio drain
        media.get_audio_sample.side_effect = lambda: next(feed, None)
        recorder = stt.ReachyMiniRecorder(mock.MagicMock(media=media))
        media.start_recording.assert_called_once()
        clip = recorder.record(3.0)
        self.assertTrue(clip.speech_detected)
        self.assertAlmostEqual(clip.duration, 1.9, delta=0.4)
        recorder.close()
        media.stop_recording.assert_called_once()

    def test_reachy_tts_pushes_synthesized_audio(self):
        import wave

        def fake_say(cmd, input, check, capture_output):
            path = cmd[cmd.index("-o") + 1]
            with wave.open(path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes((np.ones(1600) * 1000).astype(np.int16).tobytes())
            self.assertEqual(input, b"Hello there")

        media = mock.MagicMock()
        media.get_output_audio_samplerate.return_value = 16000
        from speech.tts import ReachyMiniTTS
        with mock.patch("speech.tts.shutil.which", return_value="/usr/bin/say"), \
             mock.patch("speech.tts.subprocess.run", side_effect=fake_say):
            tts = ReachyMiniTTS(mock.MagicMock(media=media))
            tts.speak("Hello there")
        media.start_playing.assert_called_once()
        pushed = media.push_audio_sample.call_args[0][0]
        self.assertEqual(len(pushed), 1600)
        self.assertAlmostEqual(float(pushed[0]), 1000 / 32768, places=4)
        tts.stop()
        media.clear_player.assert_called_once()
