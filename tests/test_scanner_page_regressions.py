"""Replay visible screenshot patterns; these are not raw camera detections."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from camera_reader import RealTimeBrailleScanner
from decoder import decode_cells, decode_cells_verbose
from tests.test_punctuation_recovery import logical_cells
from tests.test_yolo_cell_stream import page
from tts import prepare_thai_speech_text
from yolo_detector import YOLOBrailleDetector


class ScannerPageRegressions(unittest.TestCase):
    def test_screenshot_second_line_and_dashboard_preserve_closing_parenthesis(self):
        case = json.loads((Path(__file__).parent/'fixtures/obsolete_consonant_line.json').read_text(encoding='utf-8'))
        cells = logical_cells(case['patterns'])
        original = deepcopy(cells)
        text = decode_cells(cells, 'thai')
        self.assertEqual(text, case['expected'])
        tokens = decode_cells_verbose(cells, 'thai')
        self.assertEqual(tokens[-2]['char'], ')')
        self.assertTrue(tokens[-1]['consumed'])
        self.assertFalse(any(t['warning'] for t in tokens))
        self.assertEqual(cells, original)
        scanner = RealTimeBrailleScanner(res_preset='480p')
        scanner._confirmed_text = text
        scanner._ui = Mock()
        scanner._compose_dashboard(None, {}, text, True)
        self.assertEqual(scanner._ui.compose.call_args.args[1]['text'], text)
        self.assertEqual(prepare_thai_speech_text(text), case['speech_expected'])

    def test_colored_detection_preserves_closed_parenthesis_after_tone_and_coda(self):
        image, _, patterns = page([['126', '123', '256', '2456', '5', '345']])
        for color, source in [('blue', image), ('red', image[:, :, ::-1].copy())]:
            with self.subTest(color=color):
                cells, _ = YOLOBrailleDetector().detect(source, lang='thai', dot_color=color)
                self.assertEqual([c['dots'] for c in cells], patterns)
                self.assertEqual(decode_cells(cells, 'thai'), 'แล้ว)')

    def test_closing_parenthesis_does_not_consume_across_boundaries(self):
        for change in ({'line_id': 1}, {'reading_x': 900}, {'row_ambiguous': True}):
            cells = logical_cells(['5', '345'])
            cells[-1].update(change)
            self.assertNotIn(')', decode_cells(cells, 'thai'))
            self.assertFalse(decode_cells_verbose(cells, 'thai')[-1]['consumed'])
        self.assertEqual(decode_cells(logical_cells(['345']), 'thai'), 'ั')

    def test_edge_names_isolated_consonants_without_changing_words(self):
        from edge_stream import prepare_edge_text
        source = 'ฃ พยัญชนะตัวที่ 3\n(ฃ) ข ค ฅ กฃ ฃา ฃ่ Aฃ'
        expected = 'ขอขวด พยัญชนะตัวที่ สาม\n(ขอขวด) ขอไข่ คอควาย คอคน กฃ ฃา ฃ่ Aฃ'
        self.assertEqual(prepare_edge_text(source, 'th-TH-PremwadeeNeural'), expected)
        self.assertEqual(prepare_edge_text(source, 'en-US-JennyNeural'), source)
        self.assertEqual(prepare_thai_speech_text(source), expected.replace('(ขอขวด)', ' วงเล็บเปิด ขอขวด วงเล็บปิด '))
