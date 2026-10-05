"""Scale regressions use known source dots, not inferred decoded characters."""

import unittest

import cv2
import numpy as np

from colored_braille import ColoredCellStream
from tests.test_yolo_cell_stream import page
from yolo_cell_stream import plan_cells


PATTERNS = ('123456', '1245', '16', '135', '123456')


def scaled_page(lines, scale, *, color='blue', angle=0, offset=(0., 0.), patterns=PATTERNS):
    image, dots, expected = page([list(patterns)] * lines, line_pitch=180, angle=angle)
    if color == 'red':
        image = image[:, :, ::-1].copy()
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # INTER_AREA pixel centres include the half-pixel resampling offset.
    centers = (np.asarray([dot['center'] for dot in dots]) + .5) * scale - .5
    if offset != (0., 0.):
        transform = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
        image = cv2.warpAffine(image, transform, image.shape[1::-1],
                               borderValue=(255, 255, 255))
        centers += offset
    return image, centers, expected


class ScaleDetectionTests(unittest.TestCase):
    def assert_page(self, image, centers, expected, color, lines):
        reader = ColoredCellStream(color)
        dots, _ = reader.find_dots(image)
        self.assertEqual(len(dots), len(centers), 'missed or false colored dot')
        actual = np.asarray([dot['center'] for dot in dots])
        distances = np.linalg.norm(centers[:, None] - actual[None], axis=2)
        self.assertLess(float(distances.min(axis=1).max()), 1.0)
        self.assertEqual(len(set(distances.argmin(axis=1))), len(centers))
        cells, debug = reader.detect(image)
        self.assertEqual([cell['dots'] for cell in cells], expected)
        self.assertEqual(debug['line_count'], lines)
        self.assertTrue(all(set(cell['grid']['slots']) == set(range(1, 7))
                            for cell in cells))

    def test_small_dot_pixels_keep_one_three_and_four_lines_for_both_colors(self):
        for color in ('blue', 'red'):
            for lines in (1, 3, 4):
                for scale in (1., .75, .5, .35, .25, .2, .15):
                    with self.subTest(color=color, lines=lines, scale=scale):
                        image, centers, expected = scaled_page(lines, scale, color=color)
                        self.assert_page(image, centers, expected, color, lines)

    def test_small_dots_with_rotation_and_fractional_camera_translation(self):
        for color in ('blue', 'red'):
            for angle in (-3, 0, 3):
                for scale in (.25, .2, .15):
                    with self.subTest(color=color, angle=angle, scale=scale):
                        image, centers, expected = scaled_page(
                            4, scale, color=color, angle=angle, offset=(.35, .2))
                        self.assert_page(image, centers, expected, color, 4)

    def test_missing_slots_and_isolated_single_pixel_noise_are_not_invented_dots(self):
        # All rows/columns still have evidence; empty slots are intentional.
        sparse = ('123456', '125', '16', '13', '123456')
        for color in ('blue', 'red'):
            for scale in (.5, .25, .15):
                with self.subTest(color=color, scale=scale):
                    image, centers, expected = scaled_page(3, scale, color=color, patterns=sparse)
                    ink = (255, 0, 0) if color == 'blue' else (0, 0, 255)
                    for x, y in ((3, 3), (image.shape[1] - 4, 3),
                                 (image.shape[1] - 4, image.shape[0] - 4)):
                        image[y, x] = ink
                    self.assert_page(image, centers, expected, color, 3)

    def test_isolated_two_to_five_pixel_specks_do_not_create_extra_cells(self):
        shapes = (((0, 0), (1, 0)), ((0, 0), (1, 0), (2, 0)),
                  ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2)))
        for color in ('blue', 'red'):
            for lines in (1, 3, 4):
                for shape in shapes:
                    with self.subTest(color=color, lines=lines, pixels=len(shape)):
                        image, centers, expected = scaled_page(lines, .15, color=color)
                        for x, y in shape:
                            image[image.shape[0]-10+y, image.shape[1]-10+x] = (
                                (255, 0, 0) if color == 'blue' else (0, 0, 255))
                        self.assert_page(image, centers, expected, color, lines)

    def test_small_component_budget_precedes_pairwise_scale_work(self):
        from unittest.mock import patch
        from yolo_cell_stream import UnreadableBrailleFrame
        image, _, _ = scaled_page(4, .15)
        with patch('colored_braille._spacing', side_effect=AssertionError('over-budget fitting')):
            with self.assertRaisesRegex(UnreadableBrailleFrame, 'Too many small'):
                ColoredCellStream(max_cells=2).find_dots(image)

    def test_geometry_does_not_depend_on_absolute_page_scale(self):
        image, dots, expected = page([list(PATTERNS)] * 4, line_pitch=180)
        for scale in (1., .5, .25, .2, .15, .1, .075, .05):
            with self.subTest(scale=scale):
                scaled = [dict(dot, center=tuple(np.asarray(dot['center']) * scale),
                               area=dot['area'] * scale * scale) for dot in dots]
                shape = tuple(round(size * scale) for size in image.shape[:2]) + (3,)
                cells = plan_cells(scaled, shape)
                self.assertEqual([cell['dots'] for cell in cells], expected)
                self.assertEqual(len(set(cell['line_id'] for cell in cells)), 4)
                self.assertTrue(all(set(cell['grid']['slots']) == set(range(1, 7))
                                    for cell in cells))


if __name__ == '__main__':
    unittest.main()
