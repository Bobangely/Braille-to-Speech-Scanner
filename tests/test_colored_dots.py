"""หาจุดสีน้ำเงิน: แยกจุดที่หมึกเชื่อมกันได้ และไม่สร้างจุดจากสิ่งที่ไม่ใช่จุดน้ำเงิน"""
import unittest

import cv2
import numpy as np

from colored_braille import ColoredCellStream

BLUE, RED, GRAY = (160, 20, 20), (20, 20, 160), (160, 160, 160)


def page():
    """สองบรรทัด บรรทัดละ 5 เซลล์เต็ม ระยะจุด 26 px"""
    image = np.full((360, 420, 3), 245, np.uint8)
    centers = [(55 + cell*62 + column*26, 50 + line*140 + row*26)
               for line in range(2) for cell in range(5) for column in range(2) for row in range(3)]
    for center in centers:
        cv2.circle(image, center, 9, BLUE, -1)
    return image, np.asarray(centers, dtype=float)


class ColoredDotsTests(unittest.TestCase):
    def assert_dots(self, image, expected):
        dots, debug = ColoredCellStream().find_dots(image)
        self.assertEqual(len(dots), len(expected))
        distance = np.linalg.norm(np.array([d['center'] for d in dots])[:, None] - expected[None], axis=2)
        self.assertLess(float(distance.min(axis=1).max()), 4.)
        return dots, debug

    def test_two_dots_joined_by_a_thin_neck_stay_two_dots(self):
        image, centers = page()
        cv2.line(image, (55, 50), (55, 76), BLUE, 3)
        _, debug = self.assert_dots(image, centers)
        self.assertEqual(debug['split_color_components'], 1)

    def test_solid_smudge_is_not_split(self):
        image, centers = page()
        cv2.ellipse(image, (390, 170), (23, 10), 0, 0, 360, BLUE, -1)
        dots, _ = self.assert_dots(image, np.vstack([centers, (390, 170)]))
        self.assertFalse(any('split_from_bbox' in dot for dot in dots))

    def test_red_gray_and_tiny_specks_are_not_dots(self):
        image, centers = page()
        for x in range(30, 390, 30):
            cv2.circle(image, (x, 165), 8, RED, -1)
            cv2.circle(image, (x, 15), 7, GRAY, -1)
            cv2.circle(image, (x, 345), 1, BLUE, -1)
        self.assert_dots(image, centers)


if __name__ == '__main__':
    unittest.main()
