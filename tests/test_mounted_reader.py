"""Mounted-camera regressions: current ink evidence, line geometry and fixed ROI."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from camera_reader import RealTimeBrailleScanner
from colored_braille import ColoredCellStream, split_color_components
from decoder import decode_cells
from tests.test_yolo_cell_stream import page
from yolo_cell_stream import plan_cells


class MountedReaderTests(unittest.TestCase):
    def test_joined_dots_read_at_cold_start_in_both_colors(self):
        original, _, expected = page([['1345', '12', '23456', '16']])
        for color, ink in [('blue', (255, 0, 0)), ('red', (0, 0, 255))]:
            for thickness in (1, 3, 5):
                with self.subTest(color=color, bridge=thickness):
                    image = original.copy()
                    image[np.all(image == (255, 0, 0), axis=2)] = ink
                    cv2.line(image, (60, 60), (100, 60), ink, thickness)
                    reader = ColoredCellStream(color)
                    cells, debug = reader.detect(image, context='camera')
                    self.assertEqual([c['dots'] for c in cells], expected)
                    self.assertEqual(debug['split_color_components'], 1)
                    self.assertFalse(debug['grid_pending'])
                    self.assertEqual(debug['rejected_color_components'], 0)
                    self.assertEqual(len({i for c in cells for i in c['source_dot_ids']}),
                                     sum(len(c['read_dots']) for c in cells))

    def test_solid_wide_color_mark_has_no_two_dot_cores(self):
        image, _, _ = page([['1345', '12', '23456', '16']])
        cv2.rectangle(image, (52, 52), (108, 68), (255, 0, 0), -1)
        dots, debug = ColoredCellStream().find_dots(image)
        self.assertEqual(debug['split_color_components'], 0)
        self.assertFalse(any('split_from_bbox' in dot for dot in dots))

    def test_noisy_frame_cannot_exceed_pairwise_component_budget(self):
        dots = [dict(center=(i*20., 20.), bbox=(i*20, 12, i*20+16, 28), area=200.)
                for i in range(25)]
        with patch('colored_braille._spacing', side_effect=AssertionError('over budget')):
            self.assertIs(split_color_components(dots, None, max_dots=24), dots)

    def test_different_line_slopes_preserve_slots_and_source_ownership(self):
        image = np.full((800, 2000, 3), 255, np.uint8)
        dots, expected = [], []
        patterns = ['1345', '12', '23456', '16', '1345', '12356', '15', '145']
        for row, slope in enumerate((-.018, .045)):
            for cell, pattern in enumerate(patterns*2):
                expected.append(frozenset(map(int, pattern)))
                for number in map(int, pattern):
                    u, v = cell*95+(number > 3)*40, ((number-1) % 3)*40
                    x = 100+np.cos(slope)*u-np.sin(slope)*v
                    y = 150+row*350+np.sin(slope)*u+np.cos(slope)*v
                    dots.append(dict(center=(x, y), area=200., confidence=1.))
        cells = plan_cells(dots, image.shape)
        self.assertEqual([c['dots'] for c in cells], expected)
        self.assertEqual([c['line_id'] for c in cells], [0]*16+[1]*16)
        ids = [i for c in cells for i in c['source_dot_ids']]
        self.assertEqual(sorted(ids), list(range(len(dots))))
        for cell in cells:
            slots = np.array(list(cell['grid']['slots'].values()))
            for index in cell['source_dot_ids']:
                self.assertLess(np.linalg.norm(slots-dots[index]['center'], axis=1).min(), 1.)

    def test_measured_book_rows_do_not_fragment_at_acquisition(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/mounted_rows.json').read_text(encoding='utf-8'))
        for case in fixture:
            with self.subTest(frame=case['frame']):
                dots = [dict(center=p[:2], area=p[2], confidence=1.) for p in case['points']]
                cells = plan_cells(dots, (2160, 3840, 3))
                self.assertEqual(len(cells), 26)
                self.assertEqual(decode_cells(cells, 'thai'), 'นิทานตัว ต\nกระต่าย เต่า ต้วมเตี้ยม')
                self.assertFalse(any(c['row_ambiguous'] for c in cells))

    def test_fixed_roi_preserves_source_pixels_across_resolutions_and_zoom(self):
        scanner = RealTimeBrailleScanner(scan_roi=(.2, .1, .9, .6))
        for height, width in ((1, 1), (240, 400), (1080, 1920), (2160, 3840)):
            image = np.zeros((height, width, 3), np.uint8)
            for zoom in (1., 2., 4.):
                scanner.zoom_level = zoom
                for center in ((0., 0.), (.5, .5), (1., 1.)):
                    scanner.zoom_center = center
                    crop, box = scanner._apply_zoom(image)
                    x1, y1, x2, y2 = box
                    self.assertGreater(crop.size, 0)
                    self.assertTrue(np.shares_memory(crop, image))
                    self.assertEqual(crop.shape, (y2-y1, x2-x1, 3))
                    self.assertGreaterEqual(x1, int(.2*width))
                    self.assertGreaterEqual(y1, int(.1*height))
                    self.assertLessEqual(x2, int(np.ceil(.9*width)))
                    self.assertLessEqual(y2, int(np.ceil(.6*height)))

    def test_no_roi_keeps_existing_full_frame_behavior(self):
        scanner = RealTimeBrailleScanner()
        image = np.zeros((240, 400, 3), np.uint8)
        crop, box = scanner._apply_zoom(image)
        self.assertIs(crop, image)
        self.assertIsNone(box)

    def test_fixed_roi_filters_background_before_geometry(self):
        image, _, expected = page([['1245', '6', '13456', '16']])
        cv2.circle(image, (800, 290), 8, (255, 0, 0), -1)
        scanner = RealTimeBrailleScanner(scan_roi=(0., 0., .6, .7))
        crop, box = scanner._apply_zoom(image)
        cells, debug = scanner.detector.detect(crop, context=('camera', box))
        self.assertEqual([c['dots'] for c in cells], expected)
        self.assertFalse(debug['grid_pending'])
        self.assertEqual(decode_cells(cells, 'thai'), 'กญา')

    def test_invalid_fixed_roi_fails_before_starting_camera_or_model(self):
        for roi in ((0, 0, 0, 1), (0, .8, 1, .2), (-.1, 0, 1, 1),
                    (0, 0, 1.1, 1), (0, 0, float('nan'), 1), (0, 1)):
            with self.subTest(roi=roi), patch('camera_reader.YOLOBrailleDetector') as detector:
                with self.assertRaisesRegex(ValueError, 'scan_roi'):
                    RealTimeBrailleScanner(scan_roi=roi)
                detector.assert_not_called()


if __name__ == '__main__':
    unittest.main()
