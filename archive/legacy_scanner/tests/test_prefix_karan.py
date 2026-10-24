"""Prefix 356 uses complete adjacent pairs, without blocking medial letters."""

from copy import deepcopy
import time
import unittest
from unittest.mock import Mock

import numpy as np

from camera_reader import AsyncBrailleWorker, RealTimeBrailleScanner, _uncertainty_notice
from decoder import decode_cells, decode_cells_verbose
from tests.test_punctuation_recovery import logical_cells
from tests.test_thai_standard import render
from yolo_detector import YOLOBrailleDetector


# Independent cell transcription of วัฒนธรรม (not model-generated labels).
CULTURE = ['2456', '345', '36', '23456', '1345', '356', '23456', '1235', '1235', '134']


class PrefixKaranTests(unittest.TestCase):
    def test_complete_prefix_pairs_work_at_start_and_inside_words(self):
        cases = [
            (CULTURE, 'วัฒนธรรม'),
            (['356', '23456', '1345', '16'], 'ธนา'),
            (['135', '356', '23456', '12', '1236', '16', '13456'], 'อธิบาย'),
            (['234', '12', '23456', '356', '23456', '12'], 'สิทธิ'),
            (['356', '13'], 'ฃ'),
            (['1245', '356', '13'], 'กฃ'),
        ]
        for patterns, expected in cases:
            with self.subTest(expected=expected):
                cells = logical_cells(patterns)
                original = deepcopy(cells)
                self.assertEqual(decode_cells(cells, 'thai'), expected)
                self.assertFalse(any(t['warning'] for t in decode_cells_verbose(cells, 'thai')))
                self.assertEqual(cells, original)

    def test_standalone_karan_and_nonmatching_following_cell_are_preserved(self):
        for tail, expected in [([], 'ศักดิ์'), (['1256', '16', '134'], 'ศักดิ์ตาม')]:
            cells = logical_cells(['6', '234', '345', '1245', '145', '12', '356'] + tail)
            self.assertEqual(decode_cells(cells, 'thai'), expected)
            self.assertEqual(decode_cells_verbose(cells, 'thai')[6]['char'], '์')

    def test_pair_cannot_cross_blank_gap_line_or_unreadable_cell(self):
        prefix = ['6', '234', '345', '1245', '145', '12', '356']
        for change in [dict(line_id=1), dict(reading_x=2000),
                       dict(row_ambiguous=True), dict(dots=frozenset(), crop_status='empty')]:
            with self.subTest(change=change):
                cells = logical_cells(prefix + ['23456'])
                cells[-1].update(change)
                tokens = decode_cells_verbose(cells, 'thai')
                self.assertEqual(tokens[6]['char'], '์')
                self.assertFalse(tokens[7]['consumed'])
                self.assertNotIn('ธ', decode_cells(cells, 'thai'))
                if change.get('row_ambiguous') or change.get('crop_status'):
                    self.assertTrue(tokens[7]['warning'])
        self.assertEqual(decode_cells(logical_cells(prefix + ['', '23456']), 'thai'), 'ศักดิ์ ท')

    def test_true_overlap_has_explicit_longest_pair_precedence(self):
        # จันทร์ + ทา and จันทรธา have identical dots without a boundary.
        # This decoder has no dictionary: a complete pair wins; this is not
        # proof of the intended word. A physical gap preserves the karan.
        patterns = ['245', '345', '1345', '23456', '1235', '356', '23456', '16']
        cells = logical_cells(patterns)
        self.assertEqual(decode_cells(cells, 'thai'), 'จันทรธา')
        self.assertFalse(any(t['warning'] for t in decode_cells_verbose(cells, 'thai')))
        for cell in cells[6:]:
            cell['reading_x'] += 95
        self.assertEqual(decode_cells(cells, 'thai'), 'จันทร์ ทา')

    def test_fifty_five_cells_keep_dots_order_and_all_other_tokens(self):
        # Relocate the target to logical C45. No production rule depends on it.
        cells = logical_cells(['1245'] * 39 + CULTURE + ['16'] * 6)
        for i, cell in enumerate(cells):
            cell['track_id'] = i + 101
            cell['line_id'] = 0 if i < 20 else 1 if i < 35 else 2
            cell['reading_x'] = (i - [0, 20, 35][cell['line_id']]) * 95
        original = deepcopy(cells)
        tokens = decode_cells_verbose(cells, 'thai')
        self.assertEqual(len(tokens), 55)
        self.assertEqual(tokens[44]['char'], 'ธ')
        self.assertEqual((tokens[44]['cell_start'], tokens[44]['cell_end']), (44, 45))
        self.assertTrue(tokens[45]['consumed'])
        self.assertFalse(any(t['warning'] for t in tokens))
        self.assertEqual([t['char'] for t in tokens[:39]], ['ก'] * 39)
        self.assertEqual([t['char'] for t in tokens[39:44]], ['ว', 'ั', 'ฒ', '', 'น'])
        self.assertEqual([t['char'] for t in tokens[46:]], ['ร', 'ร', 'ม'] + ['า'] * 6)
        self.assertEqual(cells, original)

    def test_color_detection_and_pairing_agree_with_decoder(self):
        image = render(CULTURE)
        cells, debug = YOLOBrailleDetector().detect(image, lang='thai')
        self.assertEqual([c['dots'] for c in cells], [frozenset(map(int, p)) for p in CULTURE])
        self.assertEqual(decode_cells(cells, 'thai'), 'วัฒนธรรม')
        tokens = decode_cells_verbose(cells, 'thai')
        self.assertFalse(any(t['warning'] for t in tokens))
        self.assertEqual(cells[5]['symbol'], tokens[5]['char'])
        self.assertTrue(cells[6]['symbol_continuation'])
        self.assertTrue(tokens[6]['consumed'])
        self.assertFalse(debug.get('grid_pending'))

    def test_warning_number_matches_grid_track_id_not_list_index(self):
        result = dict(cells=[dict(track_id=46)], verbose_results=[
            dict(dots=[3, 5, 6], warning='ambiguous_row_grid', consumed=False)])
        self.assertEqual(_uncertainty_notice(result), 'C46 [356]: ambiguous_row_grid')
        result['cells'] = []
        self.assertEqual(_uncertainty_notice(result), 'C1 [356]: ambiguous_row_grid')

    def test_worker_and_confirmation_recover_without_bypassing_error_gate(self):
        cells = logical_cells(CULTURE)

        class Detector:
            def detect(self, image, lang='thai'):
                current = deepcopy(cells)
                if image[0, 0, 0] == 0:
                    current[5]['row_ambiguous'] = True
                return current, dict(dots=[], grid_pending=False)

        scanner = RealTimeBrailleScanner(res_preset='480p')
        worker = AsyncBrailleWorker(Detector(), default_lang='thai')
        scanner.ai_worker = worker
        scanner._preview = Mock(motion_valid=True)
        scanner._preview.render.side_effect = lambda frame, *args: frame.copy()
        scanner._draw_top_hud = Mock()
        scanner._draw_notice = Mock()
        worker.start()
        thread = worker.thread
        try:
            for frame_id in range(1, 8):
                frame = np.full((120, 320, 3), 0 if frame_id == 1 else 255, np.uint8)
                scanner._render_frame(frame, frame_id)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    result = worker.get_latest_results()
                    if result['frame_id'] == frame_id:
                        break
                    time.sleep(.002)
                else:
                    self.fail('Worker stopped publishing frames')
                for _ in range(3):
                    scanner._render_frame(frame, frame_id)
                self.assertIs(worker.thread, thread)
                self.assertTrue(thread.is_alive())
                self.assertIsNone(result['error'])
                self.assertEqual(result['status'], 'uncertain' if frame_id == 1 else 'ok')
                self.assertEqual(len(scanner.history), frame_id - 1)
                display = scanner._preview.render.call_args.args[2]
                self.assertEqual(display['decoded_text'], 'วัฒนธรรม' if frame_id == 7 else '')
            self.assertTrue(scanner._draw_top_hud.call_args.args[2])
        finally:
            self.assertTrue(worker.stop())


if __name__ == '__main__':
    unittest.main()
