"""Temporal display geometry and confirmation never invent recognized dots."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from camera_reader import RealTimeBrailleScanner
from colored_braille import ColoredCellStream
from decoder import decode_cells, decode_cells_verbose
from grid_stability import GridStabilizer
from live_preview import LivePreview
from tests.test_handheld_reader import moved
from tests.test_yolo_cell_stream import page
from yolo_detector import YOLOBrailleDetector


def jittered(cells, offset):
    result = deepcopy(cells)
    for cell in result:
        shift = np.array([offset, -offset])
        cell['center'] = tuple(np.array(cell['center']) + shift)
        cell['x'], cell['y'] = cell['center']
        cell['crop_quad'] = (np.array(cell['crop_quad']) + shift).tolist()
        grid = cell['grid']
        grid['slots'] = {key: tuple(np.array(point)+shift) for key, point in grid['slots'].items()}
        grid['expected_cols'] = [x+offset for x in grid['expected_cols']]
        grid['expected_rows'] = [y-offset for y in grid['expected_rows']]
        grid['bbox'] = tuple(round(value+shift[i % 2]) for i, value in enumerate(grid['bbox']))
    return result


class StableGridTests(unittest.TestCase):
    def setUp(self):
        self.image, _, _ = page([['1245', '6', '13456', '16']])
        self.reader = YOLOBrailleDetector()
        self.cells, self.debug = self.reader.detect(self.image)

    def test_uneven_painted_rows_do_not_split_a_single_line(self):
        _, dots, expected = page([['1245', '6', '13456', '16']])
        rng = np.random.default_rng(41)
        for index in range(36):
            image = np.full_like(self.image, 245)
            for dot in dots:
                point = np.array(dot['center'], float)
                if point[1] > 120:
                    point[1] += 12 if index % 2 else 16
                point += rng.uniform(-1.4, 1.4, 2)
                cv2.circle(image, tuple(np.round(point).astype(int)), 8, (255, 0, 0), -1)
            cells, debug = ColoredCellStream().detect(image)
            with self.subTest(frame=index):
                self.assertEqual([c['dots'] for c in cells], expected)
                self.assertEqual(debug['line_count'], 1)
                self.assertEqual(decode_cells(cells, 'thai'), 'กญา')

    def test_grid_jitter_reduces_without_changing_raw_dots_or_input(self):
        tracker = GridStabilizer()
        tracker.update(self.cells, self.image, 'blue')
        raw = jittered(self.cells, 3)
        before = deepcopy(raw)
        stable = tracker.update(raw, self.image, 'blue')
        self.assertEqual(raw, before)
        for original, current, display in zip(self.cells, raw, stable):
            self.assertEqual(display['dots'], current['dots'])
            np.testing.assert_allclose(np.array(display['center'])-original['center'], [.75, -.75])
        self.assertEqual([c['track_id'] for c in stable], [1, 2, 3, 4])

    def test_global_camera_motion_is_compensated_before_smoothing(self):
        tracker = GridStabilizer()
        tracker.update(self.cells, self.image, 'blue')
        current, _ = moved(self.image, dx=10, dy=6)
        detected, _ = self.reader.detect(current)
        stable = tracker.update(detected, current, 'blue')
        for original, display in zip(self.cells, stable):
            np.testing.assert_allclose(np.array(display['center'])-original['center'], [10, 6], atol=1.)
        self.assertEqual([c['track_id'] for c in stable], [1, 2, 3, 4])

    def test_missing_cell_does_not_renumber_survivors_or_create_ghosts(self):
        tracker = GridStabilizer()
        tracker.update(self.cells, self.image, 'blue')
        stable = tracker.update([self.cells[0], self.cells[2], self.cells[3]], self.image, 'blue')
        self.assertEqual([c['track_id'] for c in stable], [1, 3, 4])
        self.assertEqual(len(stable), 3)
        self.assertEqual(tracker.update([], self.image, 'blue'), [])

    def test_context_expiry_and_scene_loss_do_not_reuse_geometry(self):
        for reset in ('context', 'expiry', 'scene'):
            with self.subTest(reset=reset):
                tracker = GridStabilizer()
                with patch('grid_stability.time.monotonic', return_value=10):
                    tracker.update(self.cells, self.image, 'blue')
                raw = jittered(self.cells, 3)
                image = np.full_like(self.image, 255) if reset == 'scene' else self.image
                with patch('grid_stability.time.monotonic', return_value=11 if reset == 'expiry' else 10.1):
                    stable = tracker.update(raw, image, 'red' if reset == 'context' else 'blue')
                self.assertEqual([c['center'] for c in stable], [c['center'] for c in raw])

    def test_preview_never_smooths_same_result_twice_or_rebuilds_waiting_footer(self):
        preview = LivePreview()
        result = dict(result_id=1, cells=self.cells, dots=self.debug['dots'],
            decoded_text='', verbose_results=[], camera_roi=self.image, context='blue')
        with patch.object(preview._grids, 'update', wraps=preview._grids.update) as update, \
                patch.object(self.reader, 'annotate_with_text', wraps=self.reader.annotate_with_text) as draw:
            preview.render(self.image, self.reader, result, 'thai')
            result['decoded_text'] = 'กญา'
            preview.render(self.image, self.reader, result, 'thai')
            self.assertEqual(update.call_count, 1)
            self.assertEqual(draw.call_count, 3)  # Two result overlays, one waiting footer.
            result['result_id'] = 2
            preview.render(self.image, self.reader, result, 'thai')
            self.assertEqual(update.call_count, 2)
            self.assertEqual(draw.call_count, 4)


class ConfirmationTests(unittest.TestCase):
    def test_text_waits_for_new_matching_results_and_clears_on_change_or_error(self):
        scanner = RealTimeBrailleScanner(stability_threshold=3)
        image, _, _ = page([['1245', '6', '13456', '16']])
        cells, debug = scanner.detector.detect(image)
        worker = scanner.ai_worker = Mock()
        result = dict(ai_fps=20., result_id=1, lang='thai', status='ok', reason=None,
            error=None, source_shape=image.shape, decoded_text='กญา', cells=cells,
            dots=debug['dots'], verbose_results=decode_cells_verbose(cells, 'thai'),
            context=None, camera_roi=image, result_age=0.)
        worker.get_latest_results.side_effect = lambda **kwargs: dict(result, context=scanner._last_submitted_context)
        with patch.object(scanner.detector, 'annotate_with_text', wraps=scanner.detector.annotate_with_text) as draw, \
                patch.object(scanner, '_draw_top_hud') as hud:
            for _ in range(8):
                scanner._render_frame(image, 1)
            self.assertEqual(len(scanner.history), 1)
            self.assertTrue(all(not call.kwargs.get('decoded_text') for call in draw.call_args_list))
            for index in (2, 3):
                result['result_id'] = index
                scanner._render_frame(image, index)
            self.assertEqual(hud.call_args.args[1:], ('กญา', True))
            result.update(result_id=4, decoded_text='กา')
            scanner._render_frame(image, 4)
            self.assertEqual(hud.call_args.args[1:], ('', False))
            self.assertEqual(list(scanner.history), ['กา'])
            result.update(result_id=5, status='error', error='injected')
            scanner._render_frame(image, 5)
            self.assertFalse(scanner.history)
            self.assertEqual(hud.call_args.args[1:], ('', False))
        self.assertEqual(worker.get_latest_results()['decoded_text'], 'กา')  # Raw result remains available.


if __name__ == '__main__':
    unittest.main()
