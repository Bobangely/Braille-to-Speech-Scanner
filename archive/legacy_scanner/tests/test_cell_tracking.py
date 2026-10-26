"""Motion, proposal noise and source evidence regressions for the camera grid."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from cell_tracking import CellGridTracker, split_joined_dots, transform_grid
from camera_reader import AsyncBrailleWorker
from colored_braille import ColoredCellStream
from colored_roi import ColoredRoiReader
from decoder import decode_cells
from frame_motion import cell_motion_mask, estimate_motion, tracking_gray
from live_preview import LivePreview
from tests.test_handheld_reader import moved
from tests.test_camera_recovery import await_condition
from tests.test_yolo_cell_stream import page
from yolo_cell_stream import _cell_pitch, plan_cells, UnreadableBrailleFrame
from yolo_detector import YOLOBrailleDetector


class CellTrackingTests(unittest.TestCase):
    def setUp(self):
        self.image, _, self.expected = page([['1245', '6', '13456', '16']])
        self.reader = ColoredCellStream()
        self.dots, _ = self.reader.find_dots(self.image)
        self.plan = plan_cells(self.dots, self.image.shape)

    def test_full_cell_left_anchors_cannot_become_right_columns(self):
        # Measured columns from the failing book geometry, expressed in pixels.
        # The old objective selects ~78 instead of ~116, rephasing a sparse cell.
        columns = np.array([569.223, 611.720, 688.787, 803.824, 846.139, 923.459, 962.739,
                            1035.319, 1075.091, 1151.330, 1190.849, 1265.964, 1307.970, 1499.260, 1538.107])
        anchors = columns[[0, 3, 5, 7, 9, 11, 13]]
        spacing = np.median(columns[[1, 4, 6, 8, 10, 12, 14]]-anchors)
        self.assertAlmostEqual(_cell_pitch(columns, anchors, spacing), 116., delta=2.)

    def test_multiple_separated_lines_fit_their_own_uneven_rows(self):
        image, dots, expected = page([['1245', '6', '13456', '16']]*2, line_pitch=320)
        image[:] = 245
        for dot in dots:
            x, y = dot['center']
            if y > 440:
                y += 16
            cv2.circle(image, (round(x), round(y)), 8, (255, 0, 0), -1)
        cells, debug = self.reader.detect(image)
        self.assertEqual(debug['line_count'], 2)
        self.assertEqual([c['dots'] for c in cells], expected)

    def test_one_uneven_row_does_not_become_two_rows_at_cold_start(self):
        image, dots, expected = page([['123456']*8])
        image[:] = 245
        last_x = max(dot['center'][0] for dot in dots)
        for dot in dots:
            x, y = dot['center']
            if y < 80:
                y += 20*(x-60)/(last_x-60)
            else:
                y += 10  # Keep distinct dot rows separated despite the uneven top row.
            cv2.circle(image, (round(x), round(y)), 8, (255, 0, 0), -1)
        cells, debug = self.reader.detect(image)
        self.assertEqual(debug['line_count'], 1)
        self.assertEqual([c['dots'] for c in cells], expected)

    def test_stationary_camera_does_not_drift(self):
        first, _ = self.reader.detect(self.image, context='camera')
        for _ in range(15):
            cells, debug = self.reader.detect(self.image, context='camera')
        self.assertEqual(debug['grid_tracking'], 'tracked')
        np.testing.assert_allclose([c['center'] for c in cells], [c['center'] for c in first], atol=.01)
        self.assertEqual([c['track_id'] for c in cells], [1, 2, 3, 4])

    def test_two_bad_proposals_do_not_rephase_or_reorder_recognition(self):
        first, _ = self.reader.detect(self.image, context='camera')
        bad = deepcopy(self.plan)
        shift = np.array([[1., 0, 40], [0, 1., 0], [0, 0, 1.]])
        bad[1] = transform_grid(bad[1], shift, self.image.shape)
        bad.reverse()
        for _ in range(2):
            with patch('colored_braille.plan_cells', return_value=bad):
                cells, debug = self.reader.detect(self.image, context='camera')
            self.assertEqual([c['dots'] for c in cells], self.expected)
            self.assertEqual(decode_cells(cells, 'thai'), 'กญา')
            self.assertEqual([c['track_id'] for c in cells], [1, 2, 3, 4])
            np.testing.assert_allclose([c['center'] for c in cells], [c['center'] for c in first], atol=.01)

    def test_geometry_survives_temporary_row_planning_failure(self):
        self.reader.detect(self.image, context='camera')
        with patch('colored_braille.plan_cells', side_effect=UnreadableBrailleFrame('ambiguous rows')):
            cells, debug = self.reader.detect(self.image, context='camera')
        self.assertEqual([c['dots'] for c in cells], self.expected)
        self.assertEqual(debug['grid_planning_error'], 'ambiguous rows')
        self.assertFalse(debug['grid_pending'])

    def test_actual_translation_and_rotation_follow_source_pixels(self):
        first, _ = self.reader.detect(self.image, context='camera')
        for index in range(1, 9):
            current, matrix = moved(self.image, dx=index*2, dy=index, angle=index*.15)
            cells, debug = self.reader.detect(current, context='camera')
            self.assertEqual([c['dots'] for c in cells], self.expected)
            self.assertEqual([c['track_id'] for c in cells], [1, 2, 3, 4])
            expected = cv2.transform(np.float32([[c['center'] for c in first]]), matrix)[0]
            np.testing.assert_allclose([c['center'] for c in cells], expected, atol=1.5)

    def test_missing_dot_is_never_filled_from_previous_pattern(self):
        self.reader.detect(self.image, context='camera')
        reduced = self.dots[1:]
        # The image is unchanged: this isolates a threshold/detector dropout
        # from motion. Recognition must still use the reduced current evidence.
        with patch.object(self.reader, 'find_dots', return_value=(reduced, {})):
            cells, _ = self.reader.detect(self.image, context='camera')
        self.assertEqual(sum(len(c['read_dots']) for c in cells), len(reduced))
        self.assertNotEqual(cells[0]['dots'], self.expected[0])

    def test_two_ink_lobes_joined_by_thin_bridge_still_read_two_current_dots(self):
        cells, _ = self.reader.detect(self.image, context='camera')
        joined = self.image.copy()
        slots = cells[0]['grid']['slots']
        left, right = tuple(map(round, slots[1])), tuple(map(round, slots[4]))
        cv2.circle(joined, left, 10, (255, 0, 0), -1)
        cv2.circle(joined, right, 10, (255, 0, 0), -1)
        cv2.line(joined, left, right, (255, 0, 0), 1)
        dots, diagnostics = self.reader.find_dots(joined)
        refined = split_joined_dots(dots, diagnostics['mask'], cells)
        self.assertEqual(len(refined), len(self.dots))
        self.assertEqual(sum('split_from_bbox' in dot for dot in refined), 2)
        # Detector segmentation failure on an unchanged tracking image isolates
        # segmentation from the optical-flow photometric rejection policy.
        with patch.object(self.reader, 'find_dots', return_value=(dots, diagnostics)):
            actual, debug = self.reader.detect(self.image, context='camera')
        self.assertEqual([cell['dots'] for cell in actual], self.expected)
        self.assertFalse(debug['grid_pending'])

    def test_solid_smudge_and_unpainted_slots_cannot_be_split_into_dots(self):
        slots = self.plan[0]['grid']['slots']
        mask = np.zeros(self.image.shape[:2], np.uint8)
        left, right = np.asarray(slots[1]), np.asarray(slots[4])
        low, high = (left-[10, 8]).astype(int), (right+[10, 8]).astype(int)
        cv2.rectangle(mask, tuple(low), tuple(high), 255, -1)
        smudge = dict(self.dots[0], center=tuple((left+right)/2), area=400., bbox=tuple((*low, *high)))
        dots = [smudge, *self.dots[2:]]
        self.assertEqual(split_joined_dots(dots, mask, self.plan), dots)
        self.assertEqual(split_joined_dots(dots, np.zeros_like(mask), self.plan), dots)

    def test_missing_cell_hidden_then_reacquired_with_same_id_within_two_frames(self):
        self.reader.detect(self.image, context='camera')
        removed = set(self.plan[1]['source_dot_ids'])
        reduced = [d for i, d in enumerate(self.dots) if i not in removed]
        for _ in range(2):
            with patch.object(self.reader, 'find_dots', return_value=(reduced, {})):
                cells, debug = self.reader.detect(self.image, context='camera')
            self.assertEqual([c['track_id'] for c in cells], [1, 3, 4])
            self.assertTrue(debug['grid_pending'])
        cells, _ = self.reader.detect(self.image, context='camera')
        self.assertEqual([c['track_id'] for c in cells], [1, 2, 3, 4])

    def test_new_cell_requires_three_consistent_proposals(self):
        tracker = CellGridTracker()
        excluded = set(self.plan[-1]['source_dot_ids'])
        original_dots = [d for i, d in enumerate(self.dots) if i not in excluded]
        tracker.update(self.plan[:-1], original_dots, self.image, 'camera')
        for frame in range(3):
            cells, status = tracker.update(self.plan, self.dots, self.image, 'camera')
            self.assertEqual(len(cells), 3 if frame < 2 else 4)
            self.assertEqual(status, 'pending' if frame < 2 else 'reacquired')

    def test_unassigned_current_color_never_confirms_partial_word(self):
        tracker = CellGridTracker()
        extra = dict(self.dots[0], center=(490., 190.))
        cells, status = tracker.update(self.plan, self.dots+[extra], self.image, 'camera')
        self.assertEqual(len(cells), 4)
        self.assertEqual(status, 'pending')

    def test_worker_marks_pending_geometry_uncertain_even_with_valid_text(self):
        real = YOLOBrailleDetector()
        class Detector:
            def detect(inner, image, **kwargs):
                cells, debug = real.detect(image)
                return cells, dict(debug, grid_pending=True, grid_tracking='pending')
        worker = AsyncBrailleWorker(Detector())
        worker.start()
        try:
            worker.submit_frame(self.image, lang='thai', frame_id=1)
            result = await_condition(lambda: (r if (r := worker.get_latest_results())['result_id'] else None))
            self.assertEqual(result['status'], 'uncertain')
            self.assertEqual(result['reason'], 'unstable_grid')
            self.assertEqual(result['decoded_text'], 'กญา')
            self.assertTrue(result['debug_info']['grid_pending'])
        finally:
            worker.stop()

    def test_context_blank_scene_and_time_gap_discard_old_state(self):
        tracker = CellGridTracker()
        with patch('cell_tracking.time.monotonic', return_value=10):
            tracker.update(self.plan, self.dots, self.image, 'blue')
        with patch('cell_tracking.time.monotonic', return_value=11):
            cells, status = tracker.update(self.plan, self.dots, self.image, 'blue')
        self.assertEqual(status, 'acquired')
        cells, status = tracker.update([], [], np.full_like(self.image, 255), 'blue')
        self.assertEqual(cells, [])
        self.assertFalse(tracker.cells)
        cells, status = tracker.update(self.plan, self.dots, self.image, 'red')
        self.assertEqual(status, 'acquired')

    def test_roi_box_jitter_keeps_full_frame_coordinates(self):
        first, _ = self.reader.detect(self.image, context='camera', roi=(15, 15, 500, 220))
        for offset in (4, -3, 6, -5):
            cells, debug = self.reader.detect(self.image, context='camera', roi=(15+offset, 15, 500+offset, 220))
            self.assertEqual([c['dots'] for c in cells], self.expected)
            np.testing.assert_allclose([c['center'] for c in cells], [c['center'] for c in first], atol=.1)
            self.assertEqual([c['track_id'] for c in cells], [1, 2, 3, 4])

    def test_yolo_roi_refresh_preserves_camera_geometry(self):
        proposal = list(self.dots)
        roi = ColoredRoiReader(lambda _: proposal)
        with patch('colored_roi.time.monotonic', return_value=10):
            first, _ = roi.detect(self.image, self.reader, context='camera')
        proposal = [dict(d, center=(d['center'][0]+3, d['center'][1]+2)) for d in self.dots]
        with patch('colored_roi.time.monotonic', return_value=10.4):
            cells, debug = roi.detect(self.image, self.reader, context='camera')
        self.assertTrue(debug['yolo_inference'])
        self.assertEqual([c['dots'] for c in cells], self.expected)
        np.testing.assert_allclose([c['center'] for c in cells], [c['center'] for c in first], atol=.1)

    def test_preview_uses_recognition_grid_without_second_smoothing(self):
        detector, preview = YOLOBrailleDetector(), LivePreview()
        cells, debug = detector.detect(self.image, context='camera')
        result = dict(result_id=1, cells=cells, dots=debug['dots'], camera_roi=self.image,
                      decoded_text='', verbose_results=[])
        with patch.object(preview._grids, 'update', side_effect=AssertionError('second smoother')):
            preview.render(self.image, detector, result, 'thai')
        self.assertIs(preview._stable_cells, cells)

    def test_braille_motion_is_not_replaced_by_stationary_background(self):
        scene = np.full((640, 900, 3), 245, np.uint8)
        # A richly textured desk has many stronger features than the small word.
        rng = np.random.default_rng(10)
        scene[280:] = rng.integers(0, 256, scene[280:].shape, dtype=np.uint8)
        h, w = self.image.shape[:2]
        scene[:min(h, 250), :w] = self.image[:min(h, 250)]
        moved_scene = scene.copy()
        shifted, _ = moved(self.image, dx=8, dy=3)
        moved_scene[:min(h, 250), :w] = shifted[:min(h, 250)]
        before, after = tracking_gray(scene), tracking_gray(moved_scene)
        mask = cell_motion_mask(self.plan, scene.shape, before.shape)
        motion = estimate_motion(before, after, mask)
        self.assertIsNotNone(motion)
        np.testing.assert_allclose(motion[:2, 2], [8*before.shape[1]/900, 3*before.shape[0]/640], atol=.4)
        full_motion = estimate_motion(before, after)
        # Reproduces why the former whole-image tracker could follow the desk.
        if full_motion is not None:
            self.assertLess(np.linalg.norm(full_motion[:2, 2]), 1.)


if __name__ == '__main__':
    unittest.main()
