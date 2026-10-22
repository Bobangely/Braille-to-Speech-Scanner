import unittest
from unittest.mock import patch

import numpy as np

from dot_fusion import tile_windows, merge_dots
from yolo_detector import YOLOBrailleDetector


class YoloOnlyTests(unittest.TestCase):
    def test_removed_detection_modes_are_rejected(self):
        with patch.object(YOLOBrailleDetector, '_load_model'):
            for mode in ('cv', 'opencv', 'hybrid', 'unknown'):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    YOLOBrailleDetector(mode=mode)
            with self.assertRaises(ValueError):
                YOLOBrailleDetector(yolo_pipeline='legacy')
            detector = YOLOBrailleDetector()
        self.assertFalse(hasattr(detector, '_opencv_detector'))
        self.assertFalse(hasattr(detector, '_detect_hybrid'))

    def test_tiles_cover_the_whole_image_and_obey_budget(self):
        windows = tile_windows(3840, 2160)
        coverage = np.zeros((2160, 3840), bool)
        for x1, y1, x2, y2 in windows:
            coverage[y1:y2, x1:x2] = True
        self.assertTrue(coverage.all())
        self.assertIsNone(tile_windows(10000, 10000))
        self.assertEqual(tile_windows(3840, 2160, size=0), [])

    def test_tile_duplicates_do_not_remove_adjacent_small_dots(self):
        original = dict(center=(10, 10), area=12, confidence=.8)
        duplicates = [dict(original, center=(10.5, 10)), dict(original, center=(15, 10))]
        merged = merge_dots([original], duplicates)
        self.assertEqual([dot['center'] for dot in merged], [(10, 10), (15, 10)])


if __name__ == '__main__':
    unittest.main()
