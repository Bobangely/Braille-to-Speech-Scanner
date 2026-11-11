from contextlib import nullcontext
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

from piper_tts import synthesize_wav


class OfflineThaiSpeechTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mms = self.root / 'mms'
        self.mms.mkdir()
        for name in ('config.json', 'tokenizer_config.json', 'vocab.json', 'model.safetensors'):
            (self.mms / name).write_bytes(b'test')

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            synthesize_wav(self.mms, 'ป่า', self.root / 'speech.wav', backend='cloud')

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

    def test_edge_expands_thai_speech_units_before_sending(self):
        sent = []

        class Communicate:
            def __init__(self, text, **kwargs):
                sent.append(text)

            async def save(self, path):
                Path(path).write_bytes(b'mp3')

        def write_wav(path, samples, rate, subtype):
            with wave.open(str(path), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(b'\0\0' * 100)

        soundfile = types.SimpleNamespace(read=lambda *args, **kwargs: ([], 24000), write=write_wav)
        output = self.root / 'speech.wav'
        with patch.dict('sys.modules', {'edge_tts': types.SimpleNamespace(Communicate=Communicate),
                                        'soundfile': soundfile}):
            metrics = synthesize_wav('', 'ฃ', output, backend='edge')
            synthesize_wav('', 'ฃ', output, backend='edge', voice='en-US-JennyNeural')
        self.assertEqual(sent, ['ขอขวด', 'ฃ'])  # แปลงเฉพาะเสียงไทย
        self.assertGreater(metrics['audio_seconds'], 0)
