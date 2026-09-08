"""Independent dot fixtures from WBU/Liblouis, not reverse-generated config assertions."""

import unittest
import cv2
import numpy as np

from decoder import decode_cells, decode_cells_verbose
from thai_decoder import normalize_thai
from generate_test import generate_braille_image


CASES = [
    ('ญ', ['6', '13456']), ('ศ', ['6', '234']),
    ('ภ', ['6', '1456']), ('ธ', ['356', '23456']),
    ('ฬ', ['6', '123']), ('ฆ', ['6', '136']), ('ฌ', ['6', '346']),
    ('ฎ', ['6', '145']), ('ฏ', ['6', '1256']), ('ฐ', ['6', '2345']),
    ('ฑ', ['6', '23456']), ('ณ', ['6', '1345']),
    ('ฅ', ['36', '136']), ('ฒ', ['36', '23456']), ('ษ', ['36', '234']),
    ('ฃ', ['356', '13']),
    ('กึ', ['1245', '246']), ('กื', ['1245', '26']),
    ('ก่า', ['1245', '35', '16']), ('ก้า', ['1245', '256', '16']),
    ('ก๊า', ['1245', '2356', '16']), ('ก๋า', ['1245', '236', '16']),
    ('เก้า', ['1245', '235', '256']), ('เสีย', ['234', '12356']),
    ('เสื้อ', ['234', '12345', '256']), ('กลัว', ['1245', '123', '15']),
    ('ภู่', ['6', '1456', '25', '35']), ('ศักดิ์', ['6', '234', '345', '1245', '145', '12', '356']),
    ('ใคร', ['156', '2', '136', '1235']),
]


def logical_cells(patterns):
    return [dict(dots=frozenset(map(int, p)), x=60+i*95, y=100,
                 center=(60+i*95, 100), grid={'expected_cols': [60+i*95, 100+i*95]})
            for i, p in enumerate(patterns)]


def render(patterns):
    image = np.full((220, 120+95*len(patterns), 3), 245, np.uint8)
    for i, pattern in enumerate(patterns):
        for value in pattern:
            d = int(value)-1
            cv2.circle(image, (60+i*95+(d//3)*40, 60+(d%3)*40), 10, (255, 120, 0), -1)
    return image


class ThaiStandardTests(unittest.TestCase):
    def test_reference_patterns_direct(self):
        for expected, patterns in CASES:
            with self.subTest(expected=expected):
                self.assertEqual(decode_cells(logical_cells(patterns), lang='thai'), expected)


    def test_multicell_overlay_has_single_label(self):
        labels = decode_cells_verbose(logical_cells(['6', '13456', '16']), lang='thai')
        self.assertEqual([r['char'] for r in labels], ['ญ', '', 'า'])
        self.assertEqual(labels[0]['cell_end'], 1)
        self.assertTrue(labels[1]['consumed'])

    def test_no_prefix_consumption_across_space_or_line(self):
        for modify in [dict(x=1000), dict(y=300), dict(line_id=2)]:
            cells = logical_cells(['6', '13456'])
            cells[0]['line_id'] = 1
            cells[1].update(modify)
            self.assertNotIn('ญ', decode_cells(cells, lang='thai'))

    def test_tones_before_and_after_compound_vowels(self):
        for patterns in [['234', '256', '12345'], ['234', '12345', '256']]:
            self.assertEqual(decode_cells(logical_cells(patterns), lang='thai'), 'เสื้อ')

    def test_normalization_does_not_guess_different_dots(self):
        self.assertEqual(normalize_thai('ก้ิ'), 'กิ้')
        self.assertEqual(normalize_thai('ขูาว แลืว'), 'ขูาว แลืว')



if __name__ == '__main__':
    unittest.main()
