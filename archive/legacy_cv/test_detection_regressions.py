"""Headless regressions; fake predictions isolate fusion from model availability."""

import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from decoder import decode_cells
from detector import BrailleDetector
from dot_fusion import merge_dots, tile_windows
from tests.test_accuracy import TEST_DATASET
from yolo_detector import YOLOBrailleDetector


def dot(x, y, radius=2, **extra):
    extra.setdefault('confidence', .9)
    return dict(center=(x, y), area=np.pi * radius ** 2, circularity=1,
                bbox=(x-radius, y-radius, x+radius, y+radius), **extra)


class DetectionRegressions(unittest.TestCase):
    def make_detector(self, **kwargs):
        with patch.object(YOLOBrailleDetector, '_load_model'):
            detector = YOLOBrailleDetector(**kwargs)
        detector.model = object()
        return detector

    def test_sample_sentences_at_small_scales(self):
        for path, color, lang, expected in TEST_DATASET:
            image = cv2.imread(path)
            self.assertIsNotNone(image, path)
            for scale in (1, .5, .25, .2, .15):
                with self.subTest(path=path, scale=scale):
                    small = cv2.resize(image, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
                    cells, _ = BrailleDetector(dot_color=color).detect(small)
                    self.assertEqual(decode_cells(cells, lang=lang), expected)

    def test_blank_and_single_pixel_noise(self):
        image = np.full((100, 200, 3), 255, np.uint8)
        image[::10, ::10] = (255, 0, 0)
        cells, debug = BrailleDetector().detect(image)
        self.assertEqual(cells, [])
        self.assertEqual(debug['dots'], [])

    def test_distinct_small_dots_survive_fusion(self):
        original = dot(10, 10, source='opencv')
        result = merge_dots([original], [dot(10.5, 10, confidence=.9), dot(15, 10)])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['center'], (10, 10))
        self.assertEqual(result[0]['source'], 'both')
        self.assertEqual(original['source'], 'opencv')

    def test_tiles_cover_source_and_obey_budget(self):
        windows = tile_windows(3840, 2160)
        coverage = np.zeros((2160, 3840), bool)
        for x1, y1, x2, y2 in windows:
            coverage[y1:y2, x1:x2] = True
        self.assertTrue(coverage.all())
        self.assertLessEqual(len(windows), 16)
        self.assertIsNone(tile_windows(10000, 10000))
        self.assertEqual(tile_windows(3840, 2160, size=0), [])

    def test_tile_coordinates_and_overlap_deduplication(self):
        detector = self.make_detector(tile_size=64)
        detector.model = lambda image, **kwargs: []
        # Overview misses the dot; both overlapping tiles find it.
        with patch.object(detector, '_extract_yolo_boxes', side_effect=[
                [], [dot(45, 20)], [dot(9, 20)]]):
            dots = detector._predict_dots(np.zeros((64, 100, 3), np.uint8))
        self.assertEqual(len(dots), 1)
        self.assertEqual(dots[0]['center'], (45, 20))
        self.assertEqual(detector._inference_info['tile_count'], 2)

    def test_hybrid_recovers_partial_yolo_miss(self):
        detector = self.make_detector()
        image = cv2.imread('sample_images/test_hello_blue.png')
        with patch.object(detector, '_predict_dots', return_value=[dot(60, 60)]):
            cells, debug = detector.detect(image)
        self.assertEqual(decode_cells(cells, lang='english'), 'hello')
        self.assertGreater(debug['cv_detections'], debug['yolo_detections'])

    def test_hybrid_rejects_unsupported_candidate(self):
        detector = self.make_detector()
        image = np.full((80, 100, 3), 255, np.uint8)
        cv2.circle(image, (30, 30), 5, (0, 0, 0), -1)
        self.assertEqual(detector._refine_dots_with_opencv(image, [dot(30, 30, 6)]), [])

    def test_refinement_keeps_verified_painted_dots(self):
        for color, bgr in [('blue', (255, 0, 0)), ('red', (0, 0, 255)),
                           ('green', (0, 160, 0)), ('black', (0, 0, 0))]:
            with self.subTest(color=color):
                detector = self.make_detector(fallback_color=color)
                image = np.full((80, 100, 3), 255, np.uint8)
                cv2.circle(image, (30, 30), 3, bgr, -1)
                refined = detector._refine_dots_with_opencv(
                    image, [dot(31, 30, 4, confidence=.9)])
                self.assertEqual(len(refined), 1)
                self.assertEqual(refined[0]['center'], (30, 30))
                self.assertTrue(refined[0]['refined'])

    def test_hybrid_can_add_verified_dot_missing_from_cv(self):
        detector = self.make_detector()
        image = np.full((80, 100, 3), 255, np.uint8)
        cv2.circle(image, (30, 30), 3, (255, 0, 0), -1)
        with patch.object(detector, '_predict_dots', return_value=[dot(30, 30, 4, confidence=.9)]), \
                patch.object(detector._opencv_detector, 'detect',
                             return_value=([], {'dots': [], 'mask': np.zeros((80, 100), np.uint8)})):
            _, debug = detector.detect(image)
        self.assertEqual(len(debug['dots']), 1)
        self.assertTrue(debug['dots'][0]['refined'])

    def test_partial_arc_of_large_dot_is_not_an_extra_dot(self):
        detector = self.make_detector()
        image = np.full((80, 100, 3), 255, np.uint8)
        cv2.circle(image, (30, 30), 12, (255, 0, 0), -1)
        candidates = [dot(40, 30, 3, confidence=.95)]
        self.assertEqual(detector._refine_dots_with_opencv(image, candidates), [])

    def test_hybrid_falls_back_when_inference_fails(self):
        detector = self.make_detector()
        image = cv2.imread('sample_images/test_hello_blue.png')
        with patch.object(detector, '_predict_dots', side_effect=RuntimeError('device lost')):
            cells, debug = detector.detect(image)
        self.assertEqual(decode_cells(cells, lang='english'), 'hello')
        self.assertEqual(debug['yolo_error'], 'device lost')
        self.assertEqual(debug['method'], 'hybrid_cv_fallback')

    def test_empty_yolo_result_uses_cv(self):
        detector = self.make_detector()
        with patch.object(detector, '_predict_dots', return_value=[]):
            cells, debug = detector.detect(cv2.imread('sample_images/test_hello_blue.png'))
        self.assertEqual(decode_cells(cells, lang='english'), 'hello')
        self.assertEqual(debug['method'], 'hybrid_cv_fallback')

    def test_worker_result_sequence_and_failure_clears_old_text(self):
        from camera_reader import AsyncBrailleWorker
        from unittest.mock import Mock
        detector = Mock()
        detector.detect.return_value = BrailleDetector().detect(
            cv2.imread('sample_images/test_hello_blue.png'))
        worker = AsyncBrailleWorker(detector, default_lang='english')
        worker.start()

        def await_result(previous):
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                result = worker.get_latest_results()
                if result['result_id'] > previous:
                    return result
                time.sleep(.01)
            self.fail('Worker did not publish a result')

        try:
            worker.submit_frame(np.zeros((20, 20, 3), np.uint8))
            first = await_result(0)
            self.assertEqual(first['decoded_text'], 'hello')
            self.assertEqual(worker.get_latest_results()['result_id'], first['result_id'])
            detector.detect.side_effect = RuntimeError('camera processing failed')
            worker.submit_frame(np.zeros((20, 20, 3), np.uint8))
            failed = await_result(first['result_id'])
            self.assertEqual(failed['decoded_text'], '')
            self.assertEqual(failed['cells'], [])
            self.assertIn('failed', failed['error'])
        finally:
            worker.stop()


if __name__ == '__main__':
    unittest.main()
