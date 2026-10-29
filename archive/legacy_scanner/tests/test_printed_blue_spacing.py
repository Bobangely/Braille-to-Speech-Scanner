"""จุด sparse แนวทแยงต้องไม่ทำให้ระยะ Grid โตและจับคู่ข้ามเซลล์"""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from colored_braille import ColoredCellStream
from tests.test_yolo_cell_stream import page
from yolo_cell_stream import plan_cells


class PrintedBlueSpacingTests(unittest.TestCase):
    def assert_geometry(self, cells, dot_count):
        sources = [i for cell in cells for i in cell['source_dot_ids']]
        self.assertEqual(len(sources), dot_count)
        self.assertEqual(len(set(sources)), dot_count)
        for cell in cells:
            self.assertEqual(set(cell['grid']['slots']), set(range(1, 7)))
            self.assertEqual(cell['geometry_status'], 'valid')
            self.assertEqual(cell['overlap_ratio'], 0.)

    def test_real_printed_dots_keep_sparse_pairs_and_other_lines(self):
        # เก็บพิกัดจากภาพจริง ไม่ต้องพึ่ง output/ ที่สร้างระหว่างใช้งาน
        fixture = json.loads((Path(__file__).parent/'fixtures/printed_blue_spacing.json')
                             .read_text(encoding='utf-8'))
        for name, sample in fixture.items():
            with self.subTest(diagnostic=name):
                dots = [dict(center=(x, y), area=area) for x, y, area in sample['dots']]
                cells = plan_cells(dots, sample['shape'])
                self.assert_geometry(cells, len(dots))
                for line, expected in sample['expected_patterns'].items():
                    actual = [c['dots'] for c in cells if c['line_id'] == int(line)]
                    self.assertEqual(actual, [frozenset(map(int, p)) for p in expected])
                self.assertEqual(plan_cells(deepcopy(dots), sample['shape']), cells)

    def test_sparse_diagonal_cells_across_scale_color_angle_and_line_count(self):
        sparse = ['135', '1', '135', '16', '135', '1']
        dense = ['123456', '1245', '16', '135', '123456']
        for lines in (1, 2, 3, 5):
            patterns = [sparse] + [dense]*(lines-1)
            for scale in (.35, .7, 1.):
                for angle in (-3, 0, 3):
                    for color in ('blue', 'red'):
                        with self.subTest(lines=lines, scale=scale, angle=angle, color=color):
                            image, _, expected = page(patterns, pitch=105, line_pitch=180,
                                                      angle=angle)
                            image = cv2.resize(image, None, fx=scale, fy=scale,
                                               interpolation=cv2.INTER_AREA)
                            if color == 'red':
                                image = image[:, :, ::-1].copy()
                            cells, debug = ColoredCellStream(color).detect(image)
                            self.assertEqual([c['dots'] for c in cells], expected)
                            self.assertEqual(debug['line_count'], lines)
                            self.assertEqual(debug['colored_dots'],
                                             sum(len(p) for p in expected))
                            self.assert_geometry(cells, debug['colored_dots'])


if __name__ == '__main__':
    unittest.main()
