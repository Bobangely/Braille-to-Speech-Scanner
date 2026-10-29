"""A reviewed syllable must not suppress independent, stable readable lines."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from camera_reader import AsyncBrailleWorker, RealTimeBrailleScanner
from tests.test_camera_recovery import await_condition


FIXTURE = json.loads((Path(__file__).parent / 'fixtures/diag_039_confirmation.json').read_text(encoding='utf-8'))


def captured_page(lines):
    """Rebuild captured raw patterns, with measured logical boundaries, for replay.

    Detection is the external boundary under test here; decoding, confirmation,
    motion validation, preview and dashboard remain the real implementations.
    """
    width = max(len(line['patterns']) + len(line.get('gaps_before', [])) for line in lines) * 42 + 80
    frame = np.full((100 * len(lines) + 80, width, 3), 255, np.uint8)
    cells, dots = [], []
    for line_id, line in enumerate(lines):
        offset = 0
        for index, pattern in enumerate(line['patterns']):
            if index in line.get('gaps_before', []):
                offset += 42
            x, y = 40 + index * 42 + offset, 70 + line_id * 100
            slots = {d: (x + 12 * (d > 3), y + 12 * ((d - 1) % 3)) for d in range(1, 7)}
            source_dots = []
            for dot in map(int, pattern):
                center = slots[dot]
                source_dots.append(len(dots))
                dots.append(dict(center=center, area=64, confidence=.99))
                cv2.circle(frame, center, 4, (255, 0, 0), -1)
            cells.append(dict(dots=frozenset(map(int, pattern)), center=(x + 6, y + 12),
                              x=x, y=y, reading_x=x, dot_spacing=12, cell_pitch=42,
                              adjacency_limit=63, line_id=line_id, track_id=len(cells) + 1,
                              source_dot_ids=source_dots, grid_tracked=True, geometry_status='valid',
                              grid=dict(slots=slots, expected_cols=[x, x + 12],
                                        expected_rows=[y, y + 12, y + 24], bbox=(x - 6, y - 6, x + 18, y + 30))))
    return frame, cells, dots


class CapturedDetector:
    def __init__(self, cells, dots):
        self.cells, self.dots = cells, dots
        self.pending = False
        self.tracking = 'tracked'

    def detect(self, image, lang='thai', dot_color='blue', context=None):
        return deepcopy(self.cells), dict(dots=deepcopy(self.dots), grid_pending=self.pending,
                                          grid_tracking=self.tracking)


class LineConfirmationTests(unittest.TestCase):
    def setUp(self):
        # Diagnostic เดิมอ่านผ่านแล้ว ใช้ ambiguity จริงของ P1 ทดสอบ partial gate ต่อ
        self.partial_lines = deepcopy(FIXTURE['lines'])
        self.partial_lines[-1] = dict(patterns=['1245', '1', '1345', '135', '1'],
                                      gaps_before=[], text='กันอะ', status='uncertain')
        self.frame, self.cells, self.dots = captured_page(self.partial_lines)
        self.source = CapturedDetector(self.cells, self.dots)
        self.scanner = RealTimeBrailleScanner(res_preset='480p')
        self.worker = AsyncBrailleWorker(self.source)
        self.scanner.ai_worker = self.worker
        self.worker.start()
        self.addCleanup(self.worker.stop)
        self.frame_id = 0

    def feed(self):
        self.frame_id += 1
        self.scanner._render_frame(self.frame, self.frame_id)
        result = await_condition(lambda: (r if (r := self.worker.get_latest_results())['frame_id'] == self.frame_id else None))
        self.scanner._render_frame(self.frame, self.frame_id)
        return result

    def test_genuine_ambiguity_reports_line_scope_without_losing_page_review(self):
        result = self.feed()
        self.assertEqual(result['status'], 'uncertain')
        lines = result.get('line_results', [])
        self.assertEqual(len(lines), 4)
        self.assertEqual([line['status'] for line in lines], ['ok', 'ok', 'ok', 'uncertain'])
        self.assertEqual([line['text'] for line in lines], [line['text'] for line in self.partial_lines])
        self.assertEqual(lines[-1]['reason'], 'AMBIGUOUS_SEGMENTATION')

    def test_genuine_ambiguity_confirms_three_lines_but_never_claims_the_reviewed_line(self):
        for _ in range(6):
            result = self.feed()
        state = self.scanner._ui._state
        self.assertEqual(state.get('confirmed_lines', 0), 3)
        self.assertEqual(state.get('line_total', 0), 4)
        self.assertTrue(state.get('result_partial'))
        self.assertFalse(state['result_held'])
        for expected in ('กล่องของขวัญมีกบ', 'กบ กระโดดออกจากกล่องของขวัญ',
                         'แกะตกใจกระโดดกอด กา กับ ไก่'):
            self.assertIn(expected, state['text'])
        self.assertNotIn('กันอะ', state['text'])
        self.assertIn('รอตรวจสอบ', state['text'])
        self.assertFalse(self.scanner._confirmed_text)
        self.assertFalse(self.scanner.history)
        self.assertEqual(result['status'], 'uncertain')
        self.assertTrue(self.scanner._preview.motion_valid)

    def confirm_partial(self):
        for _ in range(6):
            self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 3)

    def test_displaying_one_result_repeatedly_does_not_confirm_lines(self):
        self.feed()
        for _ in range(5):
            self.scanner._render_frame(self.frame, self.frame_id)
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 0)
        for _ in range(4):
            self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 0)
        self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 3)

    def test_changed_line_does_not_publish_old_text_or_reset_unaffected_lines(self):
        self.confirm_partial()
        lines = deepcopy(self.partial_lines)
        lines[0]['patterns'][0] = '13'
        self.frame, self.source.cells, self.source.dots = captured_page(lines)
        self.feed()
        state = self.scanner._ui._state
        self.assertEqual(state.get('confirmed_lines', 0), 2)
        self.assertNotIn('กล่องของขวัญมีกบ', state['text'])
        self.assertIn('กบ กระโดดออกจากกล่องของขวัญ', state['text'])
        self.assertIn('แกะตกใจกระโดดกอด กา กับ ไก่', state['text'])
        self.assertIn('[บรรทัด 1: รอยืนยัน 1/6]', state['text'])

    def test_reacquired_track_ids_require_new_evidence_even_for_same_text(self):
        self.confirm_partial()
        for cell in self.source.cells:
            if cell['line_id'] == 0:
                cell['track_id'] += 1000
        self.feed()
        state = self.scanner._ui._state
        self.assertEqual(state.get('confirmed_lines', 0), 2)
        self.assertNotIn('กล่องของขวัญมีกบ', state['text'])

    def test_tracker_reacquisition_with_reused_ids_requires_six_new_results(self):
        for _ in range(5):
            self.feed()
        self.source.tracking = 'acquired'
        self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 0)
        self.assertEqual([line['samples'] for line in self.scanner._line_confirmation.lines], [1, 1, 1, 0])
        # A single acquired result can be rendered many times without resetting
        # its first sample or counting it again.
        for _ in range(3):
            self.scanner._render_frame(self.frame, self.frame_id)
        self.assertEqual([line['samples'] for line in self.scanner._line_confirmation.lines], [1, 1, 1, 0])
        self.source.tracking = 'tracked'
        for _ in range(4):
            self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 0)
        self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 3)

    def test_missing_line_resets_topology_without_combining_pages(self):
        self.confirm_partial()
        self.source.cells = [c for c in self.source.cells if c['line_id'] != 0]
        self.source.dots = [d for d in self.source.dots if d['center'][1] >= 140]
        self.frame[:140] = 255
        self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 0)
        for _ in range(5):
            self.feed()
        state = self.scanner._ui._state
        self.assertEqual(state.get('confirmed_lines', 0), 2)
        self.assertEqual(state.get('line_total', 0), 3)
        self.assertNotIn('กล่องของขวัญมีกบ\nกบ', state['text'])

    def test_pending_geometry_prevents_partial_confirmation(self):
        self.confirm_partial()
        self.source.pending = True
        self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 0)
        self.assertFalse(self.scanner._ui._state.get('result_partial'))
        self.source.pending = False
        for _ in range(5):
            self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 0)
        self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 3)

    def test_invalid_geometry_blocks_its_line_not_other_valid_lines(self):
        self.confirm_partial()
        self.source.cells[0]['geometry_status'] = 'invalid'
        self.source.cells[0]['geometry_error'] = 'clipped_grid'
        result = self.feed()
        state = self.scanner._ui._state
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(result['line_results'][0]['status'], 'uncertain')
        self.assertEqual(state.get('confirmed_lines', 0), 2)
        self.assertNotIn('กล่องของขวัญมีกบ', state['text'])
        self.assertIn('[บรรทัด 1: รอตรวจสอบ]', state['text'])

    def test_stale_inference_clears_partial_confirmation(self):
        self.confirm_partial()
        with self.worker._output_lock:
            self.worker.result_submitted_at -= 2
        self.scanner._render_frame(self.frame, self.frame_id)
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 0)
        self.assertFalse(self.scanner._ui._state.get('result_partial'))
        for _ in range(5):
            self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 0)
        self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 3)

    def test_real_motion_rejection_withholds_partial_text(self):
        self.confirm_partial()
        white = np.full_like(self.frame, 255)
        self.scanner._render_frame(white, self.frame_id)
        self.assertFalse(self.scanner._preview.motion_valid)
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 0)
        self.assertFalse(self.scanner._ui._state.get('result_partial'))

    def test_decoder_failure_cannot_reuse_line_results(self):
        self.confirm_partial()
        with patch.object(self.source, 'detect', side_effect=RuntimeError('injected detection failure')):
            with self.assertLogs('camera_reader', 'ERROR'):
                result = self.feed()
        self.assertEqual(result['status'], 'error')
        self.assertFalse(result['line_results'])
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 0)

    def test_all_valid_lines_still_confirm_full_page_through_original_gate(self):
        self.confirm_partial()
        lines = deepcopy(self.partial_lines)
        lines[3] = dict(patterns=['125'], gaps_before=[])
        self.frame, self.source.cells, self.source.dots = captured_page(lines)
        for _ in range(5):
            self.feed()
        self.assertFalse(self.scanner._confirmed_text)
        self.feed()
        expected = 'กล่องของขวัญมีกบ\nกบ กระโดดออกจากกล่องของขวัญ\nแกะตกใจกระโดดกอด กา กับ ไก่\nห'
        self.assertEqual(self.scanner._confirmed_text, expected)
        self.assertEqual(self.scanner._ui._state['text'], expected)
        self.assertFalse(self.scanner._ui._state['result_partial'])

    def test_diag_039_standard_raw_now_confirms_all_four_lines(self):
        # เก็บ fixture จาก Diagnostic เดิม ไม่แก้ raw ให้ตรงกับคำตอบที่ต้องการ
        self.frame, self.source.cells, self.source.dots = captured_page(FIXTURE['lines'])
        for _ in range(6):
            result = self.feed()
        expected = '\n'.join(line['text'] for line in FIXTURE['lines'])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual([line['status'] for line in result['line_results']], ['ok'] * 4)
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines'), 4)
        self.assertEqual(self.scanner._confirmed_text, expected)
        self.assertFalse(self.scanner._ui._state['result_partial'])

    def test_language_and_color_changes_clear_line_confirmation(self):
        self.confirm_partial()
        self.scanner.cycle_dot_color()
        self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 0)
        for _ in range(5):
            self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 3)
        self.scanner.lang = 'english'
        self.feed()
        self.assertEqual(self.scanner._ui._state.get('confirmed_lines', 0), 0)
        self.assertFalse(self.scanner._confirmed_text)

    def test_english_full_confirmation_is_unchanged(self):
        self.scanner.lang = 'english'
        self.frame, self.source.cells, self.source.dots = captured_page([dict(patterns=['1']), dict(patterns=['12'])])
        for _ in range(6):
            result = self.feed()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(self.scanner._confirmed_text, 'a\nb')
        self.assertEqual(self.scanner._ui._state['text'], 'a\nb')
        self.assertFalse(self.scanner._ui._state['result_partial'])


if __name__ == '__main__':
    unittest.main()
