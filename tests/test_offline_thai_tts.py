import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

from piper_tts import PiperSpeech, synthesize_wav


class OfflineThaiSpeechTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mms = self.root / 'mms'
        self.mms.mkdir()
        for name in ('config.json', 'tokenizer_config.json', 'vocab.json', 'model.safetensors'):
            (self.mms / name).write_bytes(b'test')
        self.english = self.root / 'english.onnx'
        self.english.write_bytes(b'test')
        Path(str(self.english) + '.json').write_text('{}')

    def test_backend_is_only_applied_to_thai_jobs(self):
        speaker = PiperSpeech({'thai': self.mms, 'english': self.english}, thai_backend='mms')
        self.addCleanup(speaker.stop)
        jobs = []

        def start(*args, **kwargs):
            jobs.append(json.load(kwargs['stdin']))
            proc = Mock()
            proc.poll.return_value = 0
            return proc

        with patch('piper_tts.subprocess.Popen', side_effect=start):
            speaker.speak('สวัสดี', 'thai')
            speaker.speak('Hello', 'english')
        self.assertEqual([job['backend'] for job in jobs], ['mms', 'piper'])
        self.assertEqual([job['text'] for job in jobs], ['สวัสดี', 'Hello'])

    def test_missing_local_weights_fail_before_starting_process(self):
        (self.mms / 'model.safetensors').unlink()
        with patch('piper_tts.subprocess.Popen') as start:
            with self.assertRaises(FileNotFoundError):
                PiperSpeech({'thai': self.mms}, thai_backend='mms')
            start.assert_not_called()

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            PiperSpeech({'english': self.english}, thai_backend='cloud')

    def test_mms_uses_local_files_cpu_and_writes_pcm16(self):
        torch, transformers = Mock(), Mock()
        tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
        tokenizer.return_value = {'input_ids': 'test'}
        model = transformers.VitsModel.from_pretrained.return_value
        model.eval.return_value = model
        model.config.sampling_rate = 16000
        tensor = Mock()
        tensor.cpu.return_value.numpy.return_value = np.array([-2., -.5, 0., .5, 2.], dtype=np.float32)
        model.return_value.waveform = [tensor]
        from contextlib import nullcontext
        torch.inference_mode.return_value = nullcontext()
        output = self.root / 'speech.wav'
        with patch.dict('sys.modules', {'torch': torch, 'transformers': transformers}), \
                patch('socket.create_connection', side_effect=AssertionError('network')):
            metrics = synthesize_wav(self.mms, 'ป่า ป้า ป๊า ป๋า', output, backend='mms')
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with(str(self.mms), local_files_only=True)
        transformers.VitsModel.from_pretrained.assert_called_once_with(str(self.mms), local_files_only=True)
        tokenizer.assert_called_once_with('ป่า ป้า ป๊า ป๋า', return_tensors='pt')
        with wave.open(str(output), 'rb') as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 16000))
            samples = np.frombuffer(wav.readframes(5), dtype='<i2')
        self.assertEqual(samples.tolist(), [-32767, -16383, 0, 16383, 32767])
        self.assertGreater(metrics['audio_seconds'], 0)
