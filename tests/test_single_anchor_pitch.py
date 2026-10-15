"""ระยะเซลล์จาก anchor เดียวต้องมีหลักฐานพอ ไม่กลับฝั่งคอลัมน์โดยเดา"""

from copy import deepcopy
import unittest

import cv2
import numpy as np

from colored_braille import ColoredCellStream
from decoder import decode_cells
from tests.test_yolo_cell_stream import page
from yolo_cell_stream import _cell_pitch, plan_cells


class SingleAnchorPitchTests(unittest.TestCase):
    def test_measured_last_line_keeps_left_columns(self):
        # พิกัดจุดจริงจาก diag_062; คาดหมายจากตำแหน่งบนกระดาษ ไม่ใช้ผล Grid เดิม
        centers = [
            (999.7346368715084, 1088.5055865921788),
            (782.8356940509915, 1091.9688385269121),
            (450.3091922005571, 1097.401114206128),
            (887.7138643067847, 1131.825958702065),
            (784.5821727019498, 1133.242339832869),
            (556.5195530726257, 1137.4832402234638),
            (452.45714285714286, 1139.202857142857),
            (1044.0462962962963, 1170.1512345679012),
            (785.4273255813954, 1174.5377906976744),
            (453.43859649122805, 1180.672514619883),
        ]
        dots = [dict(center=center, area=450) for center in centers]
        cells = plan_cells(dots, (2160, 3840, 3))
        self.assertEqual([c['dots'] for c in cells],
                         [frozenset(map(int, p)) for p in ('123', '2', '123', '2', '16')])
        self.assertEqual(decode_cells(cells, 'thai'), 'ฦ ฦา')
        self.assertEqual(plan_cells(deepcopy(dots), (2160, 3840, 3)), cells)
        sources = [i for c in cells for i in c['source_dot_ids']]
        self.assertEqual(sorted(sources), list(range(10)))
        for cell in cells:
            self.assertEqual(set(cell['grid']['slots']), set(range(1, 7)))
            self.assertEqual(cell['geometry_status'], 'valid')
            self.assertEqual(cell['overlap_ratio'], 0)

    def test_one_anchor_across_scale_angle_and_other_lines(self):
        for lines in (1, 3):
            for scale in (.5, 1):
                for angle in (-3, 0, 3):
                    with self.subTest(lines=lines, scale=scale, angle=angle):
                        patterns = [['123', '2', '', '123', '2', '16']]
                        patterns += [['123456', '1245', '16']] * (lines - 1)
                        image, _, expected = page(patterns, pitch=105,
                                                  line_pitch=180, angle=angle)
                        image = cv2.resize(image, None, fx=scale, fy=scale,
                                           interpolation=cv2.INTER_AREA)
                        cells, debug = ColoredCellStream('blue').detect(image)
                        self.assertEqual([c['dots'] for c in cells],
                                         [p for p in expected if p])
                        self.assertEqual(debug['line_count'], lines)

    def test_two_supported_column_phases_keep_uncertainty(self):
        # ระยะ 80 และ 120 อธิบายทุกคอลัมน์ได้ แต่คอลัมน์ 120 อยู่คนละฝั่ง
        columns = np.array([0., 40., 120., 240.])
        self.assertEqual(_cell_pitch(columns, [0.], 40.), 95.)
        self.assertEqual(_cell_pitch(np.array([0., 110., 330., 440., 550., 590.]),
                                    [550.], 40.), 95.)

    def test_inconsistent_column_is_not_ignored(self):
        columns = np.array([0., 40., 151., 387.])
        self.assertEqual(_cell_pitch(columns, [0.], 40.), 95.)

    def test_insufficient_columns_keep_default(self):
        self.assertEqual(_cell_pitch(np.array([0., 40., 170.]), [0.], 40.), 95.)
        self.assertEqual(_cell_pitch(np.array([0., 170.]), [], 40.), 95.)

    def test_multiple_anchors_and_normal_cells_keep_geometry(self):
        image, dots, expected = page([['123456', '2', '123456', '16']], pitch=110)
        cells = plan_cells(dots, image.shape)
        self.assertEqual([c['dots'] for c in cells], expected)
        self.assertEqual(_cell_pitch(np.array([0., 40., 110., 220., 260.]),
                                    [0., 220.], 40.), 110.)


if __name__ == '__main__':
    unittest.main()
