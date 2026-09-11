"""Trace recognition decisions independently from text encoding and rendering."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from camera_reader import AsyncBrailleWorker, RealTimeBrailleScanner
from decoder import decode_cells, decode_cells_verbose
from tests.test_camera_recovery import await_condition
from tests.test_yolo_cell_stream import page
from tools.diagnostics.export_yolo_stream import export_stream, text_trace
from yolo_cell_stream import crop_cell, plan_cells, read_cell
from yolo_detector import YOLOBrailleDetector


class RecognitionTraceTests(unittest.TestCase):
    def scene(self):
        image, dots, _ = page([[{1, 2, 3, 4, 5, 6}]])
        return image, plan_cells(dots, image.shape)[0]

    def test_question_mark_and_replacement_character_are_distinct(self):
        trace = text_trace('ญ?�')
        self.assertEqual(trace['codepoints'], ['U+0E0D', 'U+003F', 'U+FFFD'])
        self.assertEqual((trace['question_marks'], trace['replacement_characters']), (1, 1))
        self.assertEqual(bytes.fromhex(trace['utf8_hex']).decode('utf-8'), 'ญ?�')

    def test_no_crop_detections_and_off_grid_detections_have_separate_evidence(self):
        image, cell = self.scene()
        _, inverse = crop_cell(image, cell)
        empty = read_cell(cell, [], inverse)
        outside = read_cell(cell, [dict(center=(-1000, -1000), confidence=.9)], inverse)
        self.assertEqual(empty['crop_diagnostics']['detections'], 0)
        self.assertEqual(outside['crop_diagnostics']['detections'], 1)
        self.assertEqual(outside['crop_diagnostics']['outside_grid'], 1)
        self.assertEqual(empty['dots'], outside['dots'])
        self.assertEqual(decode_cells([empty], 'thai'), '�')
        self.assertEqual(decode_cells_verbose([outside], 'thai')[0]['warning'], 'empty_crop')

    def test_duplicates_are_counted_without_changing_slot_selection(self):
        image, cell = self.scene()
        _, inverse = crop_cell(image, cell)
        center = cv2.perspectiveTransform(
            np.float32([[cell['grid']['slots'][1]]]), np.linalg.inv(inverse))[0, 0]
        result = read_cell(cell, [dict(center=center, confidence=.9),
                                  dict(center=center, confidence=.5)], inverse)
        self.assertEqual(result['dots'], frozenset({1}))
        self.assertEqual(result['crop_diagnostics'],
                         dict(detections=2, matched_slots=1, outside_grid=0, duplicate_slots=1))

    def test_export_keeps_unicode_input_pixels_and_cell_failure_reason(self):
        image, cell = self.scene()
        _, inverse = crop_cell(image, cell)
        empty = read_cell(cell, [], inverse)
        with tempfile.TemporaryDirectory() as folder:
            path = export_stream(image, [empty], dict(method='yolo_cell_stream'), folder,
                                 captured_image=image, metadata=dict(frame_id=7))
            report = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(report['text_trace']['codepoints'], ['U+FFFD'])
            self.assertEqual(report['warning_counts'], {'empty_crop': 1})
            self.assertEqual(report['cells'][0]['crop_diagnostics']['detections'], 0)
            self.assertEqual(report['input_sha256'], hashlib.sha256(image.tobytes()).hexdigest())
            np.testing.assert_array_equal(cv2.imread(str(Path(folder)/report['input_file'])), image)
            np.testing.assert_array_equal(cv2.imread(str(Path(folder)/report['camera_roi_file'])), image)

    def test_renderer_receives_replacement_character_from_decoder_unchanged(self):
        image, cell = self.scene()
        _, inverse = crop_cell(image, cell)
        empty = read_cell(cell, [], inverse)
        text = decode_cells([empty], 'thai')
        tokens = decode_cells_verbose([empty], 'thai')
        detector = object.__new__(YOLOBrailleDetector)
        detector._font_cache = {}
        draw = Mock()
        with patch('yolo_detector.ImageDraw.Draw', return_value=draw):
            detector.annotate_with_text(image, [], [empty], text, tokens, 'thai')
        labels = [call.args[1] for call in draw.text.call_args_list]
        self.assertIn('C1: �', labels)
        self.assertTrue(any('ข้อความ: �' == label for label in labels))
        self.assertFalse(any('?' in label for label in labels))

    def test_worker_diagnostic_images_match_result_and_preprocessing(self):
        observed = []
        detector = Mock()
        detector.detect.side_effect = lambda frame, **kwargs: (observed.append(frame.copy()) or ([], {'dots': []}))
        worker = AsyncBrailleWorker(detector)
        frame = np.zeros((32, 32, 3), np.uint8)
        frame[:, 10:20] = 100
        original = frame.copy()
        worker.start()
        try:
            worker.submit_frame(frame, frame_id=9, sharpness_strength=.7)
            frame[:] = 255
            result = await_condition(lambda: (r if
                (r := worker.get_latest_results(include_frame=True))['frame_id'] == 9 else None))
            np.testing.assert_array_equal(result['camera_roi'], original)
            np.testing.assert_array_equal(result['inference_frame'], observed[0])
            self.assertFalse(np.array_equal(result['inference_frame'], original))
            self.assertNotIn('inference_frame', worker.get_latest_results())
        finally:
            worker.stop()
        self.assertIsNone(worker.get_latest_results(include_frame=True)['inference_frame'])

    def test_diagnostic_key_exports_ai_frame_not_newer_live_preview(self):
        with patch('camera_reader.YOLOBrailleDetector'):
            scanner = RealTimeBrailleScanner()
        inference, raw = np.zeros((20, 20, 3), np.uint8), np.ones((20, 20, 3), np.uint8)
        result = dict(inference_frame=inference, camera_roi=raw, cells=[], debug_info={},
                      lang='english', frame_id=9, result_id=3, status='empty', reason='no cells',
                      stage='detection', error=None, context=('english',), sharpness_strength=.7,
                      decoded_text='')
        scanner.ai_worker = Mock()
        scanner.ai_worker.get_latest_results.return_value = result
        with patch('tools.diagnostics.export_yolo_stream.export_stream',
                   return_value=Path('diagnostic/manifest.json')) as export:
            scanner._handle_key(ord('d'), np.full_like(raw, 5), np.full_like(raw, 10))
        scanner.ai_worker.get_latest_results.assert_called_once_with(include_frame=True)
        self.assertIs(export.call_args.args[0], inference)
        self.assertIs(export.call_args.kwargs['captured_image'], raw)
        self.assertEqual(export.call_args.kwargs['metadata']['frame_id'], 9)
        self.assertEqual(export.call_args.kwargs['lang'], 'english')


if __name__ == '__main__':
    unittest.main()
