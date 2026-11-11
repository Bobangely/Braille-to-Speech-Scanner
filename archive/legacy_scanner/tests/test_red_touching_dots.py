"""Measured red lobes must reach grouping separately; solid marks stay single."""

import base64
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from colored_braille import ColoredCellStream


PATTERNS = ('123456', '124', '16', '135', '123456')


def page(*, hue=0, saturation=230, value=160, lines=1, radius=9, bridge=3):
    image = np.full((80+lines*140, 420, 3), 245, np.uint8)
    color = tuple(map(int, cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0, 0]))
    centers = []
    for line in range(lines):
        for cell, pattern in enumerate(PATTERNS):
            origin = np.array([55+cell*62, 50+line*140])
            points = {}
            for number in map(int, pattern):
                center = origin + ((number-1)//3*26, (number-1)%3*26)
                points[number] = tuple(map(int, center))
                centers.append(center)
                cv2.circle(image, points[number], radius, color, -1)
            if cell == 1 and bridge:
                for a, b in ((1, 2), (1, 4)):
                    cv2.line(image, points[a], points[b], color, bridge)
    return image, np.asarray(centers, dtype=float)


class RedTouchingDotsTests(unittest.TestCase):
    def assert_dots(self, image, expected, tolerance=4.):
        actual, debug = ColoredCellStream('red').find_dots(image)
        self.assertEqual(len(actual), len(expected))  # no extra / missed dots
        distance = np.linalg.norm(np.array([d['center'] for d in actual])[:, None]-expected[None], axis=2)
        self.assertLess(float(distance.min(axis=1).max()), tolerance)
        self.assertEqual(len(set(distance.argmin(axis=1))), len(expected))
        return actual, debug

    def test_three_touching_lobes_across_positions_hues_lighting_and_lines(self):
        for lines in (1, 2, 4):
            for hue, saturation, value in ((0, 230, 160), (179, 210, 70),
                                           (12, 100, 230), (168, 65, 240)):
                with self.subTest(lines=lines, hue=hue, saturation=saturation, value=value):
                    image, centers = page(lines=lines, hue=hue, saturation=saturation, value=value)
                    _, debug = self.assert_dots(image, centers)
                    self.assertEqual(debug['split_color_components'], lines)
                    cells, info = ColoredCellStream('red').detect(image)
                    self.assertEqual([c['dots'] for c in cells],
                                     [frozenset(map(int, p)) for _ in range(lines) for p in PATTERNS])
                    self.assertEqual(info['line_count'], lines)

    def test_blur_resize_tilt_and_unequal_paint_do_not_change_dot_count(self):
        image, centers = page(lines=2)
        # Vary paint amount on two lobes without changing their ground-truth centers.
        for center in ((117, 50), (143, 50)):
            cv2.circle(image, center, 11, (20, 20, 160), -1)
        for scale in (.6, .8, 1., 1.25):
            for angle in (-3, 0, 3):
                with self.subTest(scale=scale, angle=angle):
                    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    expected = (centers+.5)*scale-.5
                    matrix = cv2.getRotationMatrix2D((resized.shape[1]/2, resized.shape[0]/2), angle, 1.)
                    transformed = cv2.warpAffine(resized, matrix, resized.shape[1::-1], borderValue=(245,)*3)
                    transformed = cv2.GaussianBlur(transformed, (3, 3), .5)
                    expected = cv2.transform(expected[None], matrix)[0]
                    self.assert_dots(transformed, expected, tolerance=4.*scale)

    def test_solid_smudge_and_partial_dot_are_not_split(self):
        image, centers = page(bridge=0)
        cv2.ellipse(image, (350, 165), (23, 10), 0, 0, 360, (20, 20, 160), -1)
        # A scratch inside one painted disk should still produce a single dot.
        image[49:51, 48:57] = 245
        dots, _ = self.assert_dots(image, np.vstack([centers, (350, 165)]))
        smudge = min(dots, key=lambda d: np.linalg.norm(np.array(d['center'])-(350, 165)))
        self.assertNotIn('split_from_bbox', smudge)

    def test_blue_neutral_relief_and_small_speckles_do_not_become_red_dots(self):
        image, centers = page()
        for x in range(30, 390, 30):
            cv2.circle(image, (x, 175), 8, (170, 30, 30), -1)
            cv2.circle(image, (x, 195), 7, (160, 160, 160), -1)
            cv2.circle(image, (x, 15), 1, (20, 20, 160), -1)
        self.assert_dots(image, centers)

    def test_red_and_blue_match_when_ink_is_separate_and_blue_skips_new_splitter(self):
        red, expected = page(bridge=0, lines=3)
        blue, _ = page(hue=120, bridge=0, lines=3)
        red_cells, _ = ColoredCellStream('red').detect(red)
        with patch('colored_braille.split_red_components', side_effect=AssertionError('BLUE entered RED path')):
            blue_cells, _ = ColoredCellStream('blue').detect(blue)
        self.assertEqual([c['dots'] for c in red_cells], [c['dots'] for c in blue_cells])
        self.assert_dots(red, expected)

    def test_real_diagnostic_crops_recover_each_manually_counted_lobe(self):
        fixtures = json.loads(Path(__file__).with_name('fixtures').joinpath('red_touching_dots.json').read_text(encoding='utf-8'))
        for fixture in fixtures:
            with self.subTest(source=fixture['source'], bbox=fixture['bbox']):
                crop = cv2.imdecode(np.frombuffer(base64.b64decode(fixture['png']), np.uint8), cv2.IMREAD_COLOR)
                image = np.full((260, 500, 3), 245, np.uint8)
                h, w = crop.shape[:2]
                image[40:40+h, 40:40+w] = crop
                references = [(300+c*46, 50+r*46) for c in range(2) for r in range(3)]
                for center in references:
                    cv2.circle(image, center, 16, (25, 25, 160), -1)
                expected = np.vstack([np.asarray(fixture['centers'])+(40, 40), references])
                self.assert_dots(image, expected, tolerance=5.)


if __name__ == '__main__':
    unittest.main()
