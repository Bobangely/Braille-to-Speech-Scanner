"""English Grade 1 mapping and the recorded diag_063 cell stream."""

import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from camera_reader import AsyncBrailleWorker, RealTimeBrailleScanner
from config import CHAR_TO_BRAILLE
from decoder import decode_cells, decode_cells_verbose
from line_confirmation import LineConfirmation, reading_lines
from tests.test_camera_recovery import await_condition


def cells(*patterns):
    return [dict(dots=set(map(int, pattern))) for pattern in patterns]


class EnglishBrailleTests(unittest.TestCase):
    def test_alphabet(self):
        patterns = '1 12 14 145 15 124 1245 125 24 245 13 123 134 1345 135 1234 12345 1235 234 2345 136 1236 2456 1346 13456 1356'
        self.assertEqual(decode_cells(cells(*patterns.split())), 'abcdefghijklmnopqrstuvwxyz')

    def test_grade_one_punctuation_and_reverse_mapping(self):
        for pattern, char in [('2', ','), ('23', ';'), ('25', ':'),
                              ('235', '!'), ('236', '?'), ('256', '.'),
                              ('3', "'"), ('36', '-')]:
            with self.subTest(pattern=pattern):
                self.assertEqual(decode_cells(cells(pattern)), char)
                self.assertIsNone(decode_cells_verbose(cells(pattern))[0]['warning'])
                self.assertEqual(CHAR_TO_BRAILLE[char], frozenset(map(int, pattern)))

    def test_single_capital_and_numbers_with_punctuation(self):
        self.assertEqual(decode_cells(cells('6', '1', '12', '2', '', '3456', '1', '2', '12', '256', '14')), 'Ab, 1,2.3')
        self.assertEqual(decode_cells(cells('6', '', '1')), 'a')

    def test_explicit_and_geometric_spaces_and_line_break(self):
        stream = cells('1', '', '', '12', '14', '145')
        stream[4]['break_before'] = True
        for cell in stream:
            cell['line_id'] = 0
        stream[-1]['line_id'] = 1
        self.assertEqual(decode_cells(stream), 'a b c\nd')
        stream = [dict(dots={1}, reading_x=0, cell_pitch=10),
                  dict(dots={1, 2}, reading_x=20, cell_pitch=10)]
        self.assertEqual(decode_cells(stream), 'a b')

    def test_unread_cells_are_not_spaces_or_guessed_letters(self):
        for extra, warning in [(dict(crop_status='empty'), 'empty_crop'),
                               (dict(geometry_status='invalid'), 'invalid_cell_grid'),
                               (dict(row_ambiguous=True), 'ambiguous_row_grid')]:
            stream = [dict(dots=set(), **extra)]
            self.assertEqual(decode_cells(stream), '\ufffd')
            self.assertEqual(decode_cells_verbose(stream)[0]['warning'], warning)
        self.assertEqual(decode_cells(cells('123456')), '[1,2,3,4,5,6]')
        self.assertEqual(decode_cells_verbose(cells('123456'))[0]['warning'], 'unknown_pattern')

    def test_recorded_cells_keep_geometry_text_and_confirm_all_lines(self):
        stream = json.loads((Path(__file__).parent / 'fixtures' / 'english_diag_063.json').read_text())
        text = decode_cells(stream, 'english')
        self.assertEqual(text, 'One day, Jupiter called\nin al the animals in\nheaven for a meeting')
        tokens = decode_cells_verbose(stream, 'english')
        lines = reading_lines(stream, tokens, text)
        confirmation = LineConfirmation(required=2)
        for result_id in (1, 2):
            confirmation.update(dict(result_id=result_id, lang='english', status='ok', line_results=lines))
        self.assertEqual([line['confirmed'] for line in confirmation.lines], [True] * 3)

    def test_language_button_reaches_worker_decoder_and_returns_to_thai(self):
        class Detector:
            def detect(self, image, lang='thai'):
                observed.append(lang)
                return cells('1245'), dict(dots=[])

        observed = []
        scanner = RealTimeBrailleScanner(lang='thai', res_preset='480p')
        frame = np.zeros((240, 320, 3), np.uint8)
        scanner._compose_dashboard(frame, {})
        worker = AsyncBrailleWorker(Detector())
        worker.start()
        try:
            for frame_id, language, expected in [(1, 'english', 'g'), (2, 'thai', '\u0e01')]:
                scanner.history.append('previous language')
                box, _ = next(item for item in scanner._ui.buttons if item[1] == ord('l'))
                scanner._on_mouse(cv2.EVENT_LBUTTONDOWN, (box[0]+box[2])/2, (box[1]+box[3])/2, 0, None)
                scanner._handle_key(scanner._ui_pending_key, None, None)
                self.assertEqual(scanner.lang, language)
                self.assertFalse(scanner.history)
                worker.submit_frame(np.zeros((20, 20, 3), np.uint8), lang=scanner.lang, frame_id=frame_id)
                result = await_condition(lambda: r if (r := worker.get_latest_results())['frame_id'] == frame_id else None)
                self.assertEqual(result['lang'], language)
                self.assertEqual(result['decoded_text'], expected)
            self.assertEqual(observed, ['english', 'thai'])
        finally:
            worker.stop()
