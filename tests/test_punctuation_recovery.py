"""Book punctuation must decode before the existing live confirmation gate."""

from copy import deepcopy
import json
from pathlib import Path
import time
import unittest
from unittest.mock import Mock

import numpy as np

from camera_reader import AsyncBrailleWorker, RealTimeBrailleScanner, _uncertainty_notice
from decoder import decode_cells, decode_cells_verbose
from tests.test_yolo_cell_stream import page
from yolo_cell_stream import pair_markers
from yolo_detector import YOLOBrailleDetector


def logical_cells(patterns):
    return [dict(dots=frozenset(map(int, pattern)), line_id=0, reading_x=95*i,
                 cell_pitch=95, adjacency_limit=145, x=95*i, y=0, center=(95*i, 0),
                 line_cell_index=i, column_ambiguous=False, row_ambiguous=False)
            for i, pattern in enumerate(patterns)]


def recorded_case():
    return json.loads((Path(__file__).parent/'fixtures/punctuation_cells.json').read_text(encoding='utf-8'))


class PunctuationRecoveryTests(unittest.TestCase):
    def test_prefixed_punctuation_and_single_cell_vowels_are_distinct(self):
        # Independent reference: Liblouis th-g1.utb, punctuation rules 84-89.
        cases = [('2', ',', 'ๆ'), ('23', ';', 'ี'), ('25', ':', 'ู'),
                 ('256', '.', '้'), ('235', '!', None), ('236', '?', '๋')]
        for suffix, punctuation, single in cases:
            with self.subTest(suffix=suffix):
                cells = logical_cells(['1245', '456', suffix, '1245'])
                self.assertEqual(decode_cells(cells, 'thai'), 'ก'+punctuation+'ก')
                tokens = decode_cells_verbose(cells, 'thai')
                self.assertEqual(tokens[1]['char'], punctuation)
                self.assertEqual(tokens[1]['cell_end'], 2)
                self.assertTrue(tokens[2]['consumed'])
                self.assertEqual(tokens[2]['char'], '')
                self.assertFalse(any(t['warning'] for t in tokens))
                if single is not None:
                    self.assertEqual(decode_cells(logical_cells([suffix]), 'thai'), single)

    def test_saved_twenty_two_cells_assemble_without_replacement_character(self):
        case = recorded_case()
        self.assertEqual(len(case['cells']), 22)
        self.assertEqual(decode_cells(case['cells'], 'thai'), case['expected_text'])
        tokens = decode_cells_verbose(case['cells'], 'thai')
        self.assertEqual(tokens[11]['char'], ':')
        self.assertTrue(tokens[12]['consumed'])
        self.assertFalse(any(t['warning'] for t in tokens))

    def test_stream_pairs_punctuation_without_changing_dots(self):
        cells = logical_cells(['1245', '456', '25', '1245'])
        original = deepcopy(cells)
        groups = list(pair_markers(cells, 'thai'))
        self.assertEqual([len(group) for group in groups], [1, 2, 1])
        self.assertEqual(groups[1][0]['symbol'], ':')
        self.assertTrue(groups[1][1]['symbol_continuation'])
        self.assertEqual([c['dots'] for group in groups for c in group], [c['dots'] for c in original])
        self.assertEqual([len(group) for group in pair_markers(cells, 'english')], [1]*4)

    def test_prefix_cannot_consume_across_gap_line_or_failed_crop(self):
        for change in (dict(line_id=1), dict(reading_x=900), dict(row_ambiguous=True),
                       dict(dots=frozenset(), crop_status='empty')):
            with self.subTest(change=change):
                cells = logical_cells(['456', '25'])
                cells[1].update(change)
                self.assertNotIn(':', decode_cells(cells, 'thai'))
                tokens = decode_cells_verbose(cells, 'thai')
                self.assertEqual(tokens[0]['warning'], 'unknown_or_incomplete_symbol')
                self.assertFalse(tokens[1]['consumed'])
        cells = logical_cells(['456', '', '25'])
        self.assertNotIn(':', decode_cells(cells, 'thai'))

    def test_incomplete_or_unsupported_prefix_remains_uncertain(self):
        for patterns in (['456'], ['456', '123456']):
            with self.subTest(patterns=patterns):
                self.assertIn('\ufffd', decode_cells(logical_cells(patterns), 'thai'))
                self.assertEqual(decode_cells_verbose(logical_cells(patterns), 'thai')[0]['warning'],
                                 'unknown_or_incomplete_symbol')

    def test_color_detection_to_punctuation_decoding(self):
        image, _, expected = page([['1245', '456', '25', '1245']])
        cells, debug = YOLOBrailleDetector().detect(image, lang='thai')
        self.assertEqual([c['dots'] for c in cells], expected)
        self.assertEqual(decode_cells(cells, 'thai'), 'ก:ก')
        self.assertFalse(any(t['warning'] for t in decode_cells_verbose(cells, 'thai')))
        self.assertFalse(debug.get('grid_pending'))

    def test_notice_names_blocking_cell_and_handles_assembly_only_failure(self):
        case = recorded_case()
        case['cells'][12]['dots'] = [1]
        tokens = decode_cells_verbose(case['cells'], 'thai')
        message = _uncertainty_notice(dict(verbose_results=tokens))
        self.assertIn('C12 [456]', message)
        self.assertIn('unknown/incomplete symbol', message)
        self.assertNotIn('other:', message)
        self.assertIn('text assembly incomplete', _uncertainty_notice(dict(verbose_results=[])))
        message = _uncertainty_notice(dict(verbose_results=[
            dict(warning='empty_crop', dots=[], consumed=False),
            dict(warning='empty_crop', dots=[], consumed=True),
            dict(warning='ambiguous_row_grid', dots=[1], consumed=False)]))
        self.assertIn('C1 [empty]: empty crop', message)
        self.assertIn('(+1 more)', message)

    def test_worker_recovers_and_ui_confirms_six_new_results_without_restart(self):
        case = recorded_case()

        class Detector:
            def detect(self, image, lang='thai'):
                cells = deepcopy(case['cells'])
                if image[0, 0, 0] == 0:
                    cells[12]['dots'] = [1]  # unsupported prefix body
                return cells, dict(dots=[], method='colored_cell_stream', grid_pending=False)

        scanner = RealTimeBrailleScanner(res_preset='480p')
        worker = AsyncBrailleWorker(Detector(), default_lang='thai')
        scanner.ai_worker = worker
        scanner._preview = Mock(motion_valid=True)
        scanner._preview.render.side_effect = lambda frame, *args: frame.copy()
        scanner._draw_top_hud = Mock()
        scanner._draw_notice = Mock()
        worker.start()
        original_thread = worker.thread
        try:
            for frame_id in range(1, 10):
                frame = np.full((120, 320, 3), 0 if frame_id <= 3 else 255, np.uint8)
                scanner._render_frame(frame, frame_id)
                deadline = time.monotonic()+3
                while time.monotonic() < deadline:
                    result = worker.get_latest_results()
                    if result['frame_id'] == frame_id:
                        break
                    time.sleep(.002)
                else:
                    self.fail('Worker failed to publish the next frame')
                # Repainting the same result cannot advance confirmation.
                for _ in range(3):
                    scanner._render_frame(frame, frame_id)
                self.assertIs(worker.thread, original_thread)
                self.assertTrue(worker.thread.is_alive())
                self.assertIsNone(result['error'])
                if frame_id <= 3:
                    self.assertEqual(result['status'], 'uncertain')
                    self.assertEqual(len(scanner.history), 0)
                    self.assertIn('C12 [456]', scanner._draw_notice.call_args.args[1])
                else:
                    self.assertEqual(result['status'], 'ok')
                    self.assertEqual(len(scanner.history), frame_id-3)
                display = scanner._preview.render.call_args.args[2]
                self.assertEqual(display['decoded_text'], case['expected_text'] if frame_id == 9 else '')
            self.assertTrue(scanner._draw_top_hud.call_args.args[2])
        finally:
            self.assertTrue(worker.stop())


if __name__ == '__main__':
    unittest.main()
