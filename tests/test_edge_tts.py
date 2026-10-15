import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch
import wave

import numpy as np

from camera_reader import RealTimeBrailleScanner
from decoder import decode_cells, decode_cells_verbose
from piper_tts import PiperSpeech, synthesize_wav
from tests.test_punctuation_recovery import logical_cells
from tests.test_thai_standard import CASES
from tests.test_thai_word_pipeline import PATTERNS
from tests.test_thai_speech_names import EXPECTED as CONSONANT_NAMES
from tts import prepare_thai_speech_text


FIXTURES = Path(__file__).parent / 'fixtures'


class EdgeSpeechTests(unittest.TestCase):
    def test_edge_needs_no_local_models_and_selects_voice_by_language(self):
        speaker = PiperSpeech({}, backend='edge')
        self.addCleanup(speaker.stop)
        jobs = []

        def start(*args, **kwargs):
            jobs.append(json.load(kwargs['stdin']))
            process = Mock()
            process.poll.return_value = 0
            return process

        with patch('piper_tts.subprocess.Popen', side_effect=start):
            speaker.speak('สวัสดี', 'thai')
            speaker.speak('Hello', 'english')
        self.assertEqual([job['voice'] for job in jobs], ['th-TH-NiwatNeural', 'en-US-JennyNeural'])
        self.assertTrue(all(job['backend'] == 'edge' for job in jobs))
        with self.assertRaises(ValueError):
            speaker.speak('test', 'unknown')

    def test_edge_preserves_text_converts_mp3_and_cleans_intermediate_file(self):
        edge, soundfile = Mock(), Mock()
        edge.Communicate.return_value.save = AsyncMock()
        soundfile.read.return_value = (np.zeros(2400, dtype=np.int16), 24000)

        def write(output, data, rate, **kwargs):
            with wave.open(str(output), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(data.tobytes())

        soundfile.write.side_effect = write
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'speech.wav'
            with patch.dict('sys.modules', {'edge_tts': edge, 'soundfile': soundfile}):
                result = synthesize_wav('', 'คำที่มีใช้: ฐาน ฐิติ', output, backend='edge', voice='th-TH-PremwadeeNeural')
            edge.Communicate.assert_called_once_with('คำที่มีใช้: ฐาน ฐิติ', voice='th-TH-PremwadeeNeural', rate='+10%')
            self.assertEqual(result['audio_seconds'], .1)
            self.assertFalse(output.with_suffix('.mp3').exists())
            edge.Communicate.reset_mock()
            with patch.dict('sys.modules', {'edge_tts': edge, 'soundfile': soundfile}):
                synthesize_wav('', 'ฃ พยัญชนะตัวที่ 3', output, backend='edge', voice='th-TH-PremwadeeNeural')
            edge.Communicate.assert_called_once_with('ขอขวด พยัญชนะตัวที่ สาม', voice='th-TH-PremwadeeNeural', rate='+10%')

    def test_network_failure_is_an_error_without_another_voice_fallback(self):
        edge = Mock()
        async def fail_after_partial_download(filename):
            Path(filename).write_bytes(b'partial mp3')
            raise RuntimeError('network unavailable')
        edge.Communicate.return_value.save = AsyncMock(side_effect=fail_after_partial_download)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict('sys.modules', {'edge_tts': edge, 'soundfile': Mock(), 'piper': Mock()}):
            with self.assertRaisesRegex(RuntimeError, 'network unavailable'):
                synthesize_wav('', 'เกาะขอน', Path(directory)/'speech.wav', backend='edge', voice='th-TH-PremwadeeNeural')
            self.assertFalse((Path(directory)/'speech.mp3').exists())

    def test_existing_symbol_speech_label_does_not_change_confirmed_text(self):
        scanner = RealTimeBrailleScanner(res_preset='480p', tts_backend='edge')
        self.addCleanup(scanner._speech.stop)
        original = decode_cells(logical_cells(['1245', '16', '5', '126', '13', '16']), 'thai')
        self.assertEqual(original, 'กา(ขา')
        scanner._confirmed_text = original
        jobs = []
        def start(*args, **kwargs):
            jobs.append(json.load(kwargs['stdin']))
            process = Mock()
            process.poll.return_value = 0
            return process
        with patch('piper_tts.subprocess.Popen', side_effect=start):
            scanner._handle_key(ord('s'), None, None)
        self.assertEqual(jobs[0]['text'], 'กา วงเล็บเปิด ขา')
        self.assertEqual(scanner._confirmed_text, original)

    def test_reference_patterns_reach_edge_job_without_dropped_repeated_or_reordered_text(self):
        recorded = json.loads((FIXTURES/'punctuation_cells.json').read_text(encoding='utf-8'))
        book = json.loads((FIXTURES/'thai_g1_book_cases.json').read_text(encoding='utf-8'))
        cases = [(recorded['expected_text'], recorded['cells'])]
        hello = json.loads((FIXTURES/'thai_hello_cells.json').read_text(encoding='utf-8'))
        cases.append((hello['expected_text'], hello['cells']))
        cases += [(case['expected'], logical_cells(case['raw_patterns'])) for case in book]
        cases += [(text, logical_cells(patterns)) for text, patterns in CASES + PATTERNS]
        cases.append(('123', logical_cells(['3456', '1', '12', '14'])))
        scanner = RealTimeBrailleScanner(res_preset='480p', tts_backend='edge')
        self.addCleanup(scanner._speech.stop)
        context = scanner._context_generation
        for expected, cells in cases:
            with self.subTest(expected=expected):
                original = deepcopy(cells)
                decoded = decode_cells(cells, 'thai')
                self.assertEqual(decoded, expected)
                self.assertFalse(any(token.get('warning') for token in decode_cells_verbose(cells, 'thai')))
                spoken = dict(CONSONANT_NAMES).get(expected, expected)
                self.assertEqual(prepare_thai_speech_text(decoded), spoken)
                scanner._confirmed_text = decoded
                process = Mock()
                process.poll.return_value = 0
                jobs = []
                def capture(*args, **kwargs):
                    jobs.append(json.load(kwargs['stdin']))
                    return process
                with patch('piper_tts.subprocess.Popen', side_effect=capture):
                    scanner._handle_key(ord('s'), None, None)
                self.assertEqual(jobs[0]['text'], spoken)
                self.assertEqual(jobs[0]['voice'], 'th-TH-NiwatNeural')
                self.assertEqual(cells, original)
                self.assertEqual(scanner._context_generation, context)
