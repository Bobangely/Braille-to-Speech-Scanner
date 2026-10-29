"""WAV cache, Edge/Piper fallback and playback helpers of the finger reader's speech."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from speech_cache import CachedWavSpeaker


class HybridSpeakerTests(unittest.TestCase):
    """Fixed phrases come from the clear online voice's cache; everything else from Piper."""

    class Fake:
        def __init__(self, can):
            self.can, self.prepared, self.spoken, self.stopped = set(can), [], [], 0

        def prepare_many(self, texts, lang='thai'):
            self.prepared.append(list(texts))
            return {t for t in texts if t in self.can}

        def speak(self, text, lang='thai'):
            self.spoken.append(text)
            return True

        def stop(self):
            self.stopped += 1

    def test_fixed_phrases_use_the_online_voice_and_words_use_piper(self):
        from speech_cache import HybridSpeaker
        online, piper = self.Fake({'กอไก่'}), self.Fake({'กอไก่', 'ไก่'})
        speaker = HybridSpeaker(online, piper, fixed={'กอไก่'})
        self.assertEqual(speaker.prepare_many(['กอไก่', 'ไก่']), {'กอไก่', 'ไก่'})
        self.assertEqual((online.prepared, piper.prepared), ([['กอไก่']], [['ไก่']]))
        speaker.speak('กอไก่'), speaker.speak('ไก่')
        self.assertEqual((online.spoken, piper.spoken), (['กอไก่'], ['ไก่']))

    def test_without_fixed_set_every_text_tries_the_online_voice_first(self):
        from speech_cache import HybridSpeaker
        online, piper = self.Fake({'ไก่'}), self.Fake({'นก'})
        speaker = HybridSpeaker(online, piper, fixed=None)
        self.assertEqual(speaker.prepare_many(['ไก่', 'นก']), {'ไก่', 'นก'})
        self.assertEqual((online.prepared, piper.prepared), ([['ไก่', 'นก']], [['นก']]))

    def test_piper_is_synthesized_only_when_chosen(self):
        from speech_cache import HybridSpeaker
        online, piper = self.Fake({'ไก่'}), self.Fake({'ไก่'})
        speaker = HybridSpeaker(online, piper, fixed=None)
        speaker.prepare_many(['ไก่'])
        self.assertEqual(piper.prepared, [[]])  # Edge had it: no Piper work
        speaker.engine = 'piper'
        speaker.prepare_many(['ไก่'])
        self.assertEqual(piper.prepared[-1], ['ไก่'])

    def test_piper_stands_in_when_the_online_voice_crashes(self):
        from speech_cache import HybridSpeaker
        online, piper = self.Fake(()), self.Fake({'กอไก่'})
        online.prepare_many = mock.Mock(side_effect=RuntimeError('edge-tts env broken'))
        speaker = HybridSpeaker(online, piper, fixed=None)
        with self.assertLogs('speech_cache', 'WARNING'):
            self.assertEqual(speaker.prepare_many(['กอไก่']), {'กอไก่'})
        speaker.speak('กอไก่')
        self.assertEqual(piper.spoken, ['กอไก่'])

    def test_piper_stands_in_when_the_online_voice_has_no_audio(self):
        from speech_cache import HybridSpeaker
        online, piper = self.Fake(()), self.Fake({'กอไก่'})
        speaker = HybridSpeaker(online, piper, fixed={'กอไก่'})
        self.assertEqual(speaker.prepare_many(['กอไก่']), {'กอไก่'})
        speaker.speak('กอไก่')
        self.assertEqual((online.spoken, piper.spoken), ([], ['กอไก่']))


class CachedWavSpeakerTests(unittest.TestCase):
    def test_missing_tts_model_fails_at_startup(self):
        with tempfile.TemporaryDirectory() as cache:
            with self.assertRaises(FileNotFoundError):
                CachedWavSpeaker(Path(cache) / 'missing.onnx', Path(cache))

    def speaker(self, cache, batches, played, fail=()):
        model = Path(cache) / 'thai.onnx'
        model.write_bytes(b'')

        def synthesize(model, jobs):
            batches.append([text for text, _ in jobs])
            for text, output in jobs:
                if text not in fail:
                    Path(output).write_bytes(b'RIFF')

        return CachedWavSpeaker(model, Path(cache), synthesize=synthesize, play=played.append)

    def test_prepare_many_synthesizes_missing_texts_in_one_batch(self):
        with tempfile.TemporaryDirectory() as cache:
            batches, played = [], []
            speaker = self.speaker(cache, batches, played)
            self.assertEqual(speaker.prepare_many(['ไก่', 'มา'], 'thai'), {'ไก่', 'มา'})
            self.assertEqual(speaker.prepare_many(['ไก่', 'ดี'], 'thai'), {'ไก่', 'ดี'})
            self.assertEqual(batches, [['ไก่', 'มา'], ['ดี']])

    def test_prepared_text_is_played_from_cache(self):
        with tempfile.TemporaryDirectory() as cache:
            batches, played = [], []
            speaker = self.speaker(cache, batches, played)
            speaker.prepare_many(['ไก่'], 'thai')
            self.assertTrue(speaker.speak('ไก่', 'thai'))
            speaker.stop()
            self.assertTrue(str(played[0]).endswith('.wav'))
            self.assertIsNone(played[1])

    def test_speak_never_synthesizes_in_the_frame_loop(self):
        with tempfile.TemporaryDirectory() as cache:
            batches, played = [], []
            speaker = self.speaker(cache, batches, played)
            with self.assertLogs('speech_cache', 'WARNING'):
                self.assertFalse(speaker.speak('ไม่เคยเตรียม', 'thai'))
            self.assertEqual((batches, played), ([], []))

    def test_names_are_sent_to_tts_respelled_but_played_by_name(self):
        with tempfile.TemporaryDirectory() as cache:
            batches, played = [], []
            speaker = self.speaker(cache, batches, played)
            self.assertEqual(speaker.prepare_many(['งองู', 'ไก่ งองู'], 'thai'), {'งองู', 'ไก่ งองู'})
            self.assertEqual(batches, [['งอ งู', 'ไก่ งอ งู']])
            self.assertTrue(speaker.speak('งองู', 'thai'))

    def test_edge_voice_gets_names_as_written(self):
        with tempfile.TemporaryDirectory() as cache:
            batches = []

            def synthesize(model, jobs):
                batches.append([text for text, _ in jobs])

            CachedWavSpeaker(None, Path(cache), backend='edge', synthesize=synthesize).prepare_many(['งองู', 'สระอิ'])
            self.assertEqual(batches, [['งองู', 'สะ ละ อิ']])  # Edge has its own, shorter respelling list

    def test_leading_silence_is_trimmed_after_synthesis(self):
        import wave
        from speech_cache import trim_leading_silence
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.wav'
            samples = np.concatenate([np.zeros(16000), np.full(8000, 9000), np.zeros(4000)]).astype(np.int16)
            with wave.open(str(path), 'wb') as out:
                out.setnchannels(1), out.setsampwidth(2), out.setframerate(16000)
                out.writeframes(samples.tobytes())
            trim_leading_silence(path, keep_s=0.02)
            with wave.open(str(path)) as trimmed:
                self.assertEqual(trimmed.getnframes(), 320 + 8000 + 4000)  # 1 s of silence -> 20 ms

    def test_volume_scales_a_cached_copy(self):
        import wave
        from speech_cache import at_volume
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.wav'
            with wave.open(str(path), 'wb') as out:
                out.setnchannels(1), out.setsampwidth(2), out.setframerate(16000)
                out.writeframes(np.full(100, 10000, np.int16).tobytes())
            self.assertEqual(at_volume(path, 100), path)
            quiet = at_volume(path, 50)
            with wave.open(str(quiet)) as half:
                self.assertEqual(np.frombuffer(half.readframes(100), np.int16)[0], 5000)

    def test_failed_texts_are_not_reported_ready(self):
        with tempfile.TemporaryDirectory() as cache:
            batches, played = [], []
            speaker = self.speaker(cache, batches, played, fail={'มา'})
            self.assertEqual(speaker.prepare_many(['ไก่', 'มา'], 'thai'), {'ไก่'})

    def test_blank_texts_are_never_sent_to_the_voice(self):
        # Piper writes no audio for '' and the whole batch used to fail (wave.Error).
        with tempfile.TemporaryDirectory() as cache:
            batches, played = [], []
            speaker = self.speaker(cache, batches, played)
            self.assertEqual(speaker.prepare_many(['', '  ', 'ไก่'], 'thai'), {'ไก่'})
            self.assertEqual(batches, [['ไก่']])

    def test_one_failing_text_does_not_stop_the_batch_in_the_voice_process(self):
        from speech_cache import BATCH_SCRIPT
        self.assertIn('except Exception', BATCH_SCRIPT)  # per text, inside the Piper/MMS process


if __name__ == '__main__':
    unittest.main()
