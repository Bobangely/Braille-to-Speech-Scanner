"""RED color-path regression cases; no YOLO weights or camera are needed."""

import unittest

import cv2
import numpy as np

from colored_braille import ColoredCellStream
from decoder import decode_cells


PATTERNS = ('1245', '16', '135', '123456')


def painted_page(radii, *, colors=None, incomplete=False, angle=0):
    image = np.full((max(500, 205*len(radii)+100), 900, 3), 245, np.uint8)
    for line, radius in enumerate(radii):
        for cell, pattern in enumerate(PATTERNS):
            for dot in map(int, pattern):
                x = 60+cell*95+(dot > 3)*40
                y = 60+line*205+((dot-1) % 3)*40
                color = (colors or ((0, 0, 220),))[line % len(colors or ((0, 0, 220),))]
                cv2.circle(image, (x, y), radius, color, -1)
                if incomplete and dot == 2:
                    cv2.rectangle(image, (x, y-radius), (x+radius, y-1), (245, 245, 245), -1)
    if angle:
        transform = cv2.getRotationMatrix2D((450, image.shape[0]/2), angle, 1)
        image = cv2.warpAffine(image, transform, image.shape[1::-1],
                               borderValue=(245, 245, 245))
    return image


def patterns(cells):
    return [''.join(map(str, sorted(cell['dots']))) for cell in cells]


class RedDetectionTests(unittest.TestCase):
    def test_one_to_four_lines_keep_smaller_red_dots_and_six_slots(self):
        for count in range(1, 5):
            with self.subTest(lines=count):
                radii = [12]*count
                radii[-1] = 4
                image = painted_page(radii)
                reader = ColoredCellStream('red')
                dots, diagnostics = reader.find_dots(image)
                cells, result = reader.detect(image)
                self.assertEqual(diagnostics['color_components'], 15*count)
                self.assertEqual(len(dots), 15*count)
                self.assertEqual(result['line_count'], count)
                self.assertEqual(patterns(cells), list(PATTERNS)*count)
                self.assertTrue(all(set(c['grid']['slots']) == set(range(1, 7))
                                    for c in cells))

    def test_dark_light_partial_red_and_small_tilt(self):
        dark = tuple(map(int, cv2.cvtColor(np.uint8([[[179, 210, 120]]]),
                                               cv2.COLOR_HSV2BGR)[0, 0]))
        light = tuple(map(int, cv2.cvtColor(np.uint8([[[2, 80, 235]]]),
                                                cv2.COLOR_HSV2BGR)[0, 0]))
        for angle in (0, 4):
            with self.subTest(angle=angle):
                image = painted_page((12, 12, 12, 4), colors=(dark, light),
                                     incomplete=True, angle=angle)
                cells, result = ColoredCellStream('red').detect(image)
                self.assertEqual(result['line_count'], 4)
                self.assertEqual(patterns(cells), list(PATTERNS)*4)

    def test_small_red_speckles_and_blue_regression(self):
        red = painted_page((12,))
        for point in ((20, 20), (300, 40), (750, 300), (810, 380)):
            cv2.circle(red, point, 2, (0, 0, 220), -1)
        cells, _ = ColoredCellStream('red').detect(red)
        self.assertEqual(patterns(cells), list(PATTERNS))
        blue = painted_page((12, 12, 12, 4), colors=((220, 0, 0),))
        cells, result = ColoredCellStream('blue').detect(blue)
        self.assertEqual(result['line_count'], 4)
        self.assertEqual(patterns(cells), list(PATTERNS)*4)
        self.assertEqual(ColoredCellStream('red').detect(blue)[0], [])

    def test_repeated_red_frames_keep_current_patterns(self):
        image = painted_page((12, 12, 12, 4))
        reader = ColoredCellStream('red')
        for frame in range(4):
            with self.subTest(frame=frame):
                cells, result = reader.detect(image, context=('camera', 'thai'))
                self.assertEqual(patterns(cells), list(PATTERNS)*4)
                self.assertFalse(result['grid_pending'])

    def test_sparse_small_red_cell_reaches_thai_decoder(self):
        image = np.full((900, 900, 3), 245, np.uint8)
        sparse = ('1245', '6', '13456', '16')
        for line in range(4):
            for cell, pattern in enumerate(sparse):
                for dot in map(int, pattern):
                    x = 60+cell*95+(dot > 3)*40
                    y = 60+line*205+((dot-1) % 3)*40
                    cv2.circle(image, (x, y), 12 if line < 3 else 4,
                               (0, 0, 220), -1)
        reader = ColoredCellStream('red')
        dots, diagnostics = reader.find_dots(image)
        self.assertEqual(diagnostics['color_components'], 48)
        self.assertEqual(len(dots), 48)
        for frame in range(4):
            with self.subTest(frame=frame):
                cells, result = reader.detect(image, context=('camera', 'thai'))
                self.assertEqual(patterns(cells), list(sparse)*4)
                self.assertFalse(result['grid_pending'])
                self.assertNotIn('\ufffd', decode_cells(cells, 'thai'))

    def test_off_frame_red_fragments_do_not_create_a_fake_line(self):
        image = painted_page((12, 12))
        height, width = image.shape[:2]
        for y, radius in ((height-2, 12), (height-35, 8), (height-70, 5)):
            cv2.circle(image, (width-1, y), radius, (0, 0, 220), -1)
        cells, result = ColoredCellStream('red').detect(image)
        self.assertEqual(result['line_count'], 2)
        self.assertEqual(len(cells), 2*len(PATTERNS))
        self.assertEqual(patterns(cells), list(PATTERNS)*2)


if __name__ == '__main__':
    unittest.main()
