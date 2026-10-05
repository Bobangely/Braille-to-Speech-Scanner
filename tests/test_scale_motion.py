"""Small-dot motion stays measurable without weakening scene-change checks."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from cell_tracking import CellGridTracker
from frame_motion import cell_motion_mask, estimate_motion, full_size_motion, tracking_gray, tracking_width
from grid_stability import GridStabilizer
from live_preview import LivePreview


def small_dot_page(spacing, lines=4):
    image = np.full((2160, 3840, 3), 245, np.uint8)
    cells, dots = [], []
    for line in range(lines):
        for index in range(20):
            x, y = 200+round(index*spacing*2.6), 200+line*spacing*4
            slots = {d: (x+(d > 3)*spacing, y+((d-1) % 3)*spacing) for d in range(1, 7)}
            radius = max(1, round(spacing*.2))
            for d in (1, 2, 4, 5):
                cx, cy = slots[d]
                cv2.circle(image, (cx, cy), radius, (255, 0, 0), -1)
                dots.append(dict(center=(cx, cy), area=np.pi*radius**2))
            bbox = (x-radius, y-radius, x+spacing+radius, y+2*spacing+radius)
            quad = [[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
            cells.append(dict(center=(x+spacing/2, y+spacing), dots=frozenset((1, 2, 4, 5)),
                              x=x+spacing/2, y=y+spacing, row_ambiguous=False,
                              column_ambiguous=False, pitch_ambiguous=False,
                              dot_spacing=spacing, cell_pitch=spacing*2.6, crop_quad=quad,
                              grid=dict(slots=slots, bbox=bbox, expected_cols=[x, x+spacing],
                                        expected_rows=[y, y+spacing, y+2*spacing])))
    return image, cells, dots


def translated(image):
    return cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, 2]]), image.shape[1::-1],
                          borderValue=(245, 245, 245))


class ScaleMotionTests(unittest.TestCase):
    def test_small_four_line_page_reproduces_fixed_width_failure(self):
        image, cells, _ = small_dot_page(32)
        other = translated(image)
        before, after = tracking_gray(image), tracking_gray(other)
        self.assertIsNone(estimate_motion(before, after))
        width = tracking_width(cells, image.shape)
        self.assertEqual(width, 960)
        before, after = tracking_gray(image, width), tracking_gray(other, width)
        motion = estimate_motion(before, after)
        self.assertIsNotNone(motion)
        np.testing.assert_allclose(full_size_motion(motion, image.shape, before.shape)[:2, 2],
                                   [3, 2], atol=.3)

    def test_motion_across_one_three_four_lines_and_dot_scales(self):
        for lines in (1, 3, 4):
            for spacing in (48, 32, 24, 16):
                with self.subTest(lines=lines, spacing=spacing):
                    image, cells, _ = small_dot_page(spacing, lines)
                    width = tracking_width(cells, image.shape)
                    before, after = tracking_gray(image, width), tracking_gray(translated(image), width)
                    mask = cell_motion_mask(cells, image.shape, before.shape)
                    motion = estimate_motion(before, after, mask)
                    self.assertIsNotNone(motion)
                    np.testing.assert_allclose(full_size_motion(motion, image.shape, before.shape)[:2, 2],
                                               [3, 2], atol=.4)

    def test_width_preserves_near_default_has_hysteresis_and_budget(self):
        shape = (2160, 3840, 3)
        self.assertEqual(tracking_width([dict(dot_spacing=48)], shape), 640)
        self.assertEqual(tracking_width([dict(dot_spacing=47.9)], shape), 960)
        self.assertEqual(tracking_width([dict(dot_spacing=48.1)], shape, 960), 960)
        self.assertEqual(tracking_width([dict(dot_spacing=60)], shape, 960), 640)
        self.assertEqual(tracking_width([dict(dot_spacing=2)], shape), 1280)
        self.assertEqual(tracking_width([], shape), 640)
        self.assertEqual(tracking_width([dict(dot_spacing=2)], (200, 400, 3)), 400)

    def test_blank_and_defocused_frames_still_reject_motion(self):
        image, cells, _ = small_dot_page(24)
        width = tracking_width(cells, image.shape)
        reference = tracking_gray(image, width)
        blank = np.full_like(reference, 245)
        blurred = tracking_gray(cv2.GaussianBlur(image, (45, 45), 12), width)
        self.assertIsNone(estimate_motion(reference, blank))
        self.assertIsNone(estimate_motion(reference, blurred))

    def test_trackers_keep_matching_reference_shape_on_scale_transition(self):
        image, cells, dots = small_dot_page(32)
        tracker = CellGridTracker()
        tracker.update(cells, dots, image, 'camera')
        self.assertEqual(tracker.reference.shape[1], 960)
        # A changed measurement may request a new working size only after the
        # current pair has been compared at the reference's existing size.
        smaller_image, smaller, smaller_dots = small_dot_page(24)
        with patch('cell_tracking.estimate_motion', wraps=estimate_motion) as motion:
            tracker.update(smaller, smaller_dots, smaller_image, 'camera')
        before, after = motion.call_args.args[:2]
        self.assertEqual(before.shape, after.shape)
        self.assertEqual(tracker.reference.shape[1], 1280)
        with patch('cell_tracking.estimate_motion', wraps=estimate_motion) as motion:
            tracker.update(smaller, smaller_dots, smaller_image, 'camera')
        before, after = motion.call_args.args[:2]
        self.assertEqual(before.shape, after.shape)
        self.assertEqual(before.shape[1], 1280)
        stabilizer = GridStabilizer()
        stabilizer.update(cells, image, 'camera')
        with patch('grid_stability.estimate_motion', wraps=estimate_motion) as motion:
            stabilizer.update(smaller, smaller_image, 'camera')
        before, after = motion.call_args.args[:2]
        self.assertEqual(before.shape, after.shape)
        self.assertEqual(stabilizer._reference.shape[1], 1280)

    def test_preview_checks_current_frame_at_reference_resolution(self):
        image, cells, dots = small_dot_page(32)
        class Detector:
            mode = 'color'
            def annotate_with_text(self, canvas, *args, **kwargs):
                return canvas
        result = dict(cells=cells, dots=dots, camera_roi=image, result_id=1,
                      decoded_text='', verbose_results=[], context='camera', reader_name='COLOR BLUE')
        preview = LivePreview()
        preview.render(image, Detector(), result, 'thai')
        self.assertEqual(preview._reference.shape[1], 960)
        preview.render(translated(image), Detector(), result, 'thai')
        self.assertTrue(preview.motion_valid)


if __name__ == '__main__':
    unittest.main()
