"""Ground-truth mixed painted/unpainted pages; filtering is tested before decoding."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from camera_reader import AsyncBrailleWorker, RealTimeBrailleScanner
from colored_braille import ColoredCellStream
from decoder import decode_cells, decode_cells_verbose
from tests.test_camera_recovery import await_condition
from tests.test_yolo_cell_stream import page
from tools.diagnostics.export_yolo_stream import export_stream
from yolo_detector import YOLOBrailleDetector


def mixed_page(lines, marked, pitch=95):
    image, dots, _ = page(lines, pitch=pitch)
    index = 0
    for line_id, line in enumerate(lines):
        for cell_id, pattern in enumerate(line):
            for _ in pattern:
                center = tuple(map(round, dots[index]['center']))
                if (line_id, cell_id) not in marked:
                    cv2.circle(image, center, 8, (130, 130, 130), -1)
                    cv2.circle(image, (center[0]-2, center[1]-2), 3, (205, 205, 205), -1)
                index += 1
    return image


class ColoredOnlyTests(unittest.TestCase):
    def test_many_unpainted_cells_do_not_reach_decoding(self):
        lines = [['123456']*12, ['1245', '6', '13456', '16'], ['123456']*10]
        image = mixed_page(lines, {(1, i) for i in range(4)})
        cells, debug = ColoredCellStream().detect(image)
        self.assertEqual([cell['dots'] for cell in cells], [frozenset(map(int, p)) for p in lines[1]])
        self.assertEqual(decode_cells(cells, 'thai'), 'กญา')
        self.assertEqual(debug['line_count'], 1)
        self.assertTrue(all(cell['colored_dot_count'] > 0 for cell in cells))
        self.assertFalse(any(token['warning'] for token in decode_cells_verbose(cells, 'thai')))

    def test_unpainted_page_returns_no_cells_or_replacement_text(self):
        image = mixed_page([['123456']*7]*3, set())
        cells, debug = ColoredCellStream().detect(image)
        self.assertEqual(cells, [])
        self.assertEqual(debug['colored_dots'], 0)
        self.assertEqual(decode_cells(cells, 'thai'), '')

    def test_partial_ink_broken_edges_and_small_noise_keep_dot_positions(self):
        image, dots, expected = page([['1245', '6', '13456', '16'], ['123456', '2345', '1256']])
        for index, dot in enumerate(dots):
            x, y = map(round, dot['center'])
            if index % 2:
                image[y-8:y+9, x+4:x+9] = 210  # partly painted disk
            else:
                image[y:y+2, x-8:x+9] = 210  # two-pixel break
        for point in ((20, 20), (300, 40), (750, 300), (810, 380)):
            cv2.circle(image, point, 2, (255, 0, 0), -1)
        cells, _ = ColoredCellStream().detect(image)
        self.assertEqual([cell['dots'] for cell in cells], expected)

    def test_rotation_and_mild_perspective_preserve_rows_and_six_slots(self):
        patterns = [['1245', '6', '13456', '16'], ['123456', '2345', '1256']]
        for transform in ('rotation', 'perspective'):
            with self.subTest(transform=transform):
                image, _, expected = page(patterns, angle=5 if transform == 'rotation' else 0)
                if transform == 'perspective':
                    h, w = image.shape[:2]
                    matrix = cv2.getPerspectiveTransform(
                        np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
                        np.float32([[15, 5], [w-20, 0], [w-5, h-10], [0, h]]))
                    image = cv2.warpPerspective(image, matrix, (w, h), borderValue=(255, 255, 255))
                cells, _ = ColoredCellStream().detect(image)
                self.assertEqual([cell['dots'] for cell in cells], expected)
                self.assertEqual([cell['line_id'] for cell in cells], [0]*4+[1]*3)

    def test_markers_do_not_jump_over_unpainted_cell(self):
        lines = [['6', '123456', '13456']]
        image = mixed_page(lines, {(0, 0), (0, 2)})
        cells, _ = ColoredCellStream().detect(image)
        self.assertEqual(len(cells), 2)
        self.assertFalse(any(cell.get('symbol') == 'ญ' for cell in cells))
        self.assertNotEqual(decode_cells(cells, 'thai'), 'ญ')
        self.assertTrue(decode_cells_verbose(cells, 'thai')[1]['break_before'])

    def test_adjacent_markers_work_with_wide_printed_spacing(self):
        image, _, _ = page([['1245', '6', '13456', '16']], pitch=145)
        cells, _ = ColoredCellStream().detect(image)
        self.assertEqual(decode_cells(cells, 'thai'), 'กญา')

    def test_unpainted_other_line_cannot_complete_colored_marker(self):
        image = mixed_page([['1245', '6'], ['13456', '123456']], {(0, 0), (0, 1)})
        cells, _ = ColoredCellStream().detect(image)
        self.assertEqual(len(cells), 2)
        self.assertNotIn('ญ', decode_cells(cells, 'thai'))

    def test_blue_and_both_red_hue_ranges_are_selected(self):
        base, dots, expected = page([['123456']])
        for hue, name in ((120, 'blue'), (0, 'red'), (179, 'red')):
            color = tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 255, 210]]]),
                                                cv2.COLOR_HSV2BGR)[0, 0]))
            image = base.copy()
            for dot in dots:
                cv2.circle(image, tuple(map(round, dot['center'])), 8, color, -1)
            cells, _ = ColoredCellStream(name).detect(image)
            self.assertEqual([cell['dots'] for cell in cells], expected)
            other = 'red' if name == 'blue' else 'blue'
            self.assertEqual(ColoredCellStream(other).detect(image)[0], [])

    def test_only_blue_and_red_are_supported(self):
        for name in ('green', 'black', 'auto'):
            with self.subTest(color=name), self.assertRaisesRegex(ValueError, 'Unsupported'):
                ColoredCellStream(name)

    def test_colored_reader_does_not_load_or_classify_with_yolo(self):
        image, _, expected = page([['123456', '1245']])
        with patch.object(YOLOBrailleDetector, '_load_model', side_effect=AssertionError('unexpected YOLO load')):
            detector = YOLOBrailleDetector(dot_color='blue', model_path='missing_weights.pt')
        self.assertIsNone(detector.model)
        cells, debug = detector.detect(image)
        self.assertEqual([cell['dots'] for cell in cells], expected)
        self.assertEqual(debug['method'], 'colored_cell_stream')
        self.assertFalse(debug['yolo_inference'])

    def test_small_dots_and_mild_blur_preserve_patterns(self):
        image, _, expected = page([['1245', '6', '13456', '16'], ['123456', '2345', '1256']])
        for scale in (.5, .3, .25, .2):
            with self.subTest(scale=scale):
                small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                blurred = cv2.GaussianBlur(small, (3, 3), .5)
                cells, _ = ColoredCellStream().detect(blurred)
                self.assertEqual([cell['dots'] for cell in cells], expected)

    def test_colored_diagnostics_export_without_yolo_weights(self):
        image, _, _ = page([['1245', '6', '13456']])
        detector = YOLOBrailleDetector(dot_color='blue')
        cells, debug = detector.detect(image)
        with tempfile.TemporaryDirectory() as folder:
            manifest = export_stream(image, cells, debug, folder, detector=detector)
            report = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertIsNone(report['model'])
            self.assertEqual(report['crop_role'], 'diagnostic_view')
            self.assertEqual(report['reader']['dot_color'], 'blue')
            self.assertEqual(report['text'], 'กญ')

    def test_worker_clears_previous_text_when_only_unpainted_braille_remains(self):
        painted, _, _ = page([['1245', '6', '13456']])
        unpainted = cv2.cvtColor(cv2.cvtColor(painted, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        worker = AsyncBrailleWorker(YOLOBrailleDetector(dot_color='blue'))
        worker.start()
        try:
            for number, image in enumerate((painted, unpainted, painted), 1):
                worker.submit_frame(image, frame_id=number)
                result = await_condition(lambda: (r if
                    (r := worker.get_latest_results())['frame_id'] == number else None))
                self.assertEqual(result['decoded_text'], '' if number == 2 else 'กญ')
                self.assertIsNone(result['error'])
                self.assertEqual(result['debug_info']['read_source'], 'color')
        finally:
            worker.stop()

    def test_camera_and_image_entry_points_enable_color_filter_by_default(self):
        self.assertEqual(YOLOBrailleDetector().dot_color, 'blue')
        scanner = RealTimeBrailleScanner()
        self.assertEqual(scanner.detector.dot_color, 'blue')
        self.assertIsNone(scanner.detector.model)
        image = mixed_page([['123456']*3], set())
        from main import main
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'unpainted.png'
            cv2.imwrite(str(path), image)
            output = io.StringIO()
            with patch('sys.argv', ['main.py', str(path), '--no-display']), contextlib.redirect_stdout(output):
                main()
            self.assertIn('ไม่พบข้อความ', output.getvalue())
            self.assertNotIn('�', output.getvalue())


if __name__ == '__main__':
    unittest.main()
