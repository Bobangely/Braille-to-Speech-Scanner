"""Handheld motion, color switching, and the optional ROI path use real pixels."""

import contextlib
import io
import json
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from camera_reader import AsyncBrailleWorker, RealTimeBrailleScanner
from colored_braille import ColoredCellStream
from colored_roi import ColoredRoiReader
from decoder import decode_cells, decode_cells_verbose
from frame_motion import estimate_motion, full_size_motion, tracking_gray
from live_preview import LivePreview
from tests.test_camera_recovery import await_condition
from tests.test_yolo_cell_stream import page
from tools.diagnostics.export_yolo_stream import export_stream
from yolo_detector import YOLOBrailleDetector


PATTERNS = [['1245', '6', '13456', '16'], ['123456', '2345', '1256']]


def moved(image, dx=9, dy=5, angle=0):
    transform = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1)
    transform[:, 2] += (dx, dy)
    return cv2.warpAffine(image, transform, image.shape[1::-1], borderValue=(255, 255, 255)), transform


def two_colors():
    image, dots, _ = page([['1245', '6', '13456', '16'], ['1245', '6', '1456', '16']])
    for dot in dots:
        if dot['center'][1] > 160:
            cv2.circle(image, tuple(map(round, dot['center'])), 8, (0, 0, 255), -1)
    return image


class MotionTests(unittest.TestCase):
    def test_short_word_with_fewer_than_eight_features_tracks_small_translation(self):
        image, _, _ = page([['1245', '16']])
        image = cv2.resize(image, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
        reference = tracking_gray(image)
        features = cv2.goodFeaturesToTrack(reference, maxCorners=100, qualityLevel=.02,
                                         minDistance=8, blockSize=5)
        self.assertLess(len(features), 8)
        for dx, dy in ((1, 0), (-1, 1), (2, -1)):
            with self.subTest(translation=(dx, dy)):
                current, expected = moved(image, dx=dx, dy=dy)
                motion = estimate_motion(reference, tracking_gray(current))
                self.assertIsNotNone(motion)
                np.testing.assert_allclose(motion[:2], expected, atol=.15)

    def test_sparse_word_rejects_changed_dots_scene_and_heavy_blur(self):
        image, dots, _ = page([['1245', '16']])
        missing = image.copy()
        cv2.circle(missing, tuple(map(round, dots[0]['center'])), 9, (255, 255, 255), -1)
        added = image.copy()
        cv2.circle(added, (60, 140), 8, (255, 0, 0), -1)
        small = lambda frame: cv2.resize(frame, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
        reference = tracking_gray(small(image))
        changed = [small(missing), small(added), np.full_like(small(image), 255),
                   cv2.GaussianBlur(small(image), (21, 21), 5)]
        for index, frame in enumerate(changed):
            with self.subTest(scene=index):
                current, _ = moved(frame, dx=1, dy=0)
                self.assertIsNone(estimate_motion(reference, tracking_gray(current)))

    def test_sparse_tracking_rejects_large_motion_and_insufficient_geometry(self):
        image, _, _ = page([['1245', '16']])
        image = cv2.resize(image, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
        shifted, _ = moved(image, dx=10, dy=0)
        self.assertIsNone(estimate_motion(tracking_gray(image), tracking_gray(shifted)))
        for locations in ([(20, 20), (40, 40)], [(20, 20), (40, 20), (60, 20)]):
            with self.subTest(locations=locations):
                sparse = np.full((80, 100, 3), 255, np.uint8)
                for center in locations:
                    cv2.circle(sparse, center, 2, (255, 0, 0), -1)
                shifted, _ = moved(sparse, dx=1, dy=0)
                self.assertIsNone(estimate_motion(tracking_gray(sparse), tracking_gray(shifted)))

    def test_short_word_camera_can_lock_after_small_translation(self):
        image, _, _ = page([['1245', '16']])
        image = cv2.resize(image, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
        scanner = RealTimeBrailleScanner()
        cells, debug = scanner.detector.detect(image)
        self.assertEqual(decode_cells(cells, 'thai'), 'กา')
        scanner.ai_worker = Mock()
        result = dict(ai_fps=10., result_id=1, lang='thai', status='ok', reason=None,
            error=None, source_shape=image.shape, decoded_text='กา', cells=cells,
            dots=debug['dots'], verbose_results=decode_cells_verbose(cells, 'thai'),
            context=None, camera_roi=image, result_age=0.)
        scanner.ai_worker.get_latest_results.side_effect = lambda **kwargs: dict(
            result, context=scanner._last_submitted_context)
        current, _ = moved(image, dx=1, dy=0)
        with patch.object(scanner, '_draw_top_hud') as hud:
            for result_id in range(1, scanner.stability_threshold + 1):
                result['result_id'] = result_id
                scanner._render_frame(current, result_id)
            self.assertTrue(scanner._preview.motion_valid)
            self.assertEqual(list(scanner.history), ['กา'] * scanner.stability_threshold)
            self.assertEqual(hud.call_args.args[1:], ('กา', True))

    def test_translation_and_small_rotation_match_known_transform(self):
        image, _, _ = page(PATTERNS)
        for angle in (0, 3, -3):
            with self.subTest(angle=angle):
                current, expected = moved(image, angle=angle)
                reference = tracking_gray(image)
                motion = estimate_motion(reference, tracking_gray(current))
                self.assertIsNotNone(motion)
                actual = full_size_motion(motion, image.shape, reference.shape)
                points = np.float32([[[60, 60], [300, 140], [100, 260]]])
                np.testing.assert_allclose(cv2.perspectiveTransform(points, actual),
                    cv2.transform(points, expected), atol=1.5)

    def test_blank_new_scene_and_heavy_blur_do_not_keep_old_motion(self):
        image, _, _ = page(PATTERNS)
        reference = tracking_gray(image)
        random = np.random.default_rng(5).integers(0, 256, reference.shape, dtype=np.uint8)
        for current in (np.full_like(reference, 255), random,
                        cv2.GaussianBlur(reference, (41, 41), 10)):
            with self.subTest(image=current.shape):
                self.assertIsNone(estimate_motion(reference, current))

    def test_preview_moves_annotations_and_rejects_lost_reference(self):
        image, _, _ = page(PATTERNS)
        actual_reader = YOLOBrailleDetector()
        cells, debug = actual_reader.detect(image)
        result = dict(result_id=1, cells=cells, dots=debug['dots'], decoded_text='test',
                      verbose_results=[], camera_roi=image, reader_name='COLOR BLUE')
        detector = Mock(mode='yolo')

        def annotate(canvas, dots, cells, **options):
            output = np.zeros((canvas.shape[0]+30, canvas.shape[1], 3), np.uint8)
            if cells:
                cv2.circle(output, (300, 100), 2, (0, 255, 0), -1)
                output[-10:, 10:30] = (0, 255, 0)
            return output

        detector.annotate_with_text.side_effect = annotate
        preview = LivePreview()
        current, _ = moved(image, dx=10, dy=6)
        rendered = preview.render(current, detector, result, 'thai')
        self.assertTrue(preview.motion_valid)
        self.assertGreater(rendered[106, 310, 1], 200)
        self.assertEqual(rendered[-5, 15, 1], 255)  # Footer does not move.
        lost = preview.render(np.full_like(image, 255), detector, result, 'thai')
        self.assertFalse(preview.motion_valid)
        self.assertEqual(lost[-5, 15, 1], 0)
        self.assertEqual(detector.annotate_with_text.call_count, 2)  # Reuse both overlays.

    def test_multiline_preview_draws_waiting_footer_after_tracking_loss(self):
        image, _, _ = page(PATTERNS)
        detector = YOLOBrailleDetector()
        cells, debug = detector.detect(image)
        result = dict(result_id=1, cells=cells, dots=debug['dots'],
                      decoded_text='first line\nsecond line', verbose_results=[],
                      camera_roi=image, reader_name='COLOR BLUE')
        preview = LivePreview()
        rendered = preview.render(image, detector, result, 'thai')
        blank = np.full_like(image, 255)
        lost = preview.render(blank, detector, result, 'thai')
        waiting = detector.annotate_with_text(blank, [], [], decoded_text='',
                                              lang='thai', reader_name='COLOR BLUE')
        self.assertFalse(preview.motion_valid)
        self.assertGreater(rendered.shape[0], waiting.shape[0])
        np.testing.assert_array_equal(lost, waiting)


class RoiTests(unittest.TestCase):
    def test_crop_keeps_patterns_and_restores_source_geometry(self):
        image, dots, expected = page(PATTERNS)
        predictor = Mock(return_value=dots)
        reader = ColoredCellStream()
        baseline, _ = reader.detect(image)
        stream = ColoredRoiReader(predictor)
        cells, debug = stream.detect(image, reader)
        self.assertEqual([c['dots'] for c in cells], expected)
        self.assertEqual(decode_cells(cells, 'thai'), decode_cells(baseline, 'thai'))
        self.assertLess(debug['roi_box'][2], image.shape[1])
        self.assertTrue(debug['yolo_inference'])
        for actual, full in zip(cells, baseline):
            np.testing.assert_allclose(actual['center'], full['center'], atol=.01)
            np.testing.assert_allclose(actual['crop_quad'], full['crop_quad'], atol=.01)
            self.assertAlmostEqual(actual['reading_x'], full['reading_x'], places=3)
            for slot in range(1, 7):
                np.testing.assert_allclose(actual['grid']['slots'][slot], full['grid']['slots'][slot], atol=.01)
        # Source coordinates are also consumed by OpenCV drawing in image mode.
        annotated = YOLOBrailleDetector().annotate_with_text(image, debug['dots'], cells,
            decode_cells(cells, 'thai'), decode_cells_verbose(cells, 'thai'))
        self.assertGreater(annotated.shape[0], image.shape[0])

    def test_tracks_small_motion_without_repeated_yolo_and_recovers_scene_loss(self):
        image, dots, expected = page(PATTERNS)
        predictor = Mock(return_value=dots)
        stream, reader = ColoredRoiReader(predictor), ColoredCellStream()
        with patch('colored_roi.time.monotonic', return_value=10):
            stream.detect(image, reader, context=('first',))
            current, _ = moved(image)
            cells, debug = stream.detect(current, reader, context=('first',))
            self.assertEqual([c['dots'] for c in cells], expected)
            self.assertEqual(debug['roi_source'], 'tracked_yolo_dots')
            self.assertFalse(debug['yolo_inference'])
            self.assertEqual(predictor.call_count, 1)
            predictor.return_value = []
            cells, debug = stream.detect(np.full_like(image, 255), reader, context=('first',))
            self.assertEqual(cells, [])
            self.assertEqual(debug['roi_status'], 'not_found')
            self.assertEqual(predictor.call_count, 2)
        predictor.return_value = dots
        with patch('colored_roi.time.monotonic', return_value=11):
            cells, _ = stream.detect(image, reader, context=('first',))
            self.assertEqual([c['dots'] for c in cells], expected)

    def test_empty_roi_is_throttled_and_new_context_forces_detection(self):
        image, _, _ = page(PATTERNS)
        predictor = Mock(return_value=[])
        stream, reader = ColoredRoiReader(predictor), ColoredCellStream()
        with patch('colored_roi.time.monotonic', return_value=10):
            for _ in range(5):
                cells, _ = stream.detect(image, reader, context=0)
                self.assertEqual(cells, [])
            self.assertEqual(predictor.call_count, 1)
            stream.detect(image, reader, context=1)
            self.assertEqual(predictor.call_count, 2)

    def test_yolo_proposals_never_decode_gray_relief(self):
        image, dots, _ = page(PATTERNS)
        gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        cells, debug = ColoredRoiReader(lambda _: dots).detect(gray, ColoredCellStream())
        self.assertEqual(cells, [])
        self.assertEqual(decode_cells(cells), '')
        self.assertEqual(debug['colored_dots'], 0)

    def test_proposal_exception_invalidates_roi_and_next_frame_can_recover(self):
        image, dots, _ = page(PATTERNS)
        predictor = Mock(side_effect=[dots, RuntimeError('injected'), dots])
        stream, reader = ColoredRoiReader(predictor), ColoredCellStream()
        with patch('colored_roi.time.monotonic', return_value=10):
            stream.detect(image, reader)
        with patch('colored_roi.time.monotonic', return_value=11):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                stream.detect(image, reader)
            self.assertIsNone(stream.box)
            self.assertTrue(stream.detect(image, reader)[0])


class LiveColorTests(unittest.TestCase):
    def test_per_frame_color_selection_preserves_detector_configuration(self):
        image = two_colors()
        reader = YOLOBrailleDetector()
        for color, expected in (('blue', 'กญา'), ('red', 'กภา'), ('blue', 'กญา')):
            cells, _ = reader.detect(image, dot_color=color)
            self.assertEqual(decode_cells(cells, 'thai'), expected)
            self.assertEqual(reader.dot_color, 'blue')
            self.assertIsNone(reader.model)

    def test_switch_during_inference_drops_old_color_and_preserves_one_worker(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        image = two_colors()
        real = YOLOBrailleDetector()

        class Detector:
            def detect(self, frame, lang='thai', dot_color=None, context=None):
                calls.append(dot_color)
                if dot_color == 'blue':
                    entered.set()
                    if not release.wait(3):
                        raise TimeoutError('test release')
                return real.detect(frame, lang=lang, dot_color=dot_color)

        worker = AsyncBrailleWorker(Detector())
        worker.start()
        original = worker.thread
        try:
            worker.submit_frame(image, frame_id=1, dot_color='blue', context=1)
            self.assertTrue(entered.wait(2))
            worker.submit_frame(image, frame_id=2, dot_color='red', context=2)
            release.set()
            result = await_condition(lambda: (r if
                (r := worker.get_latest_results())['frame_id'] == 2 else None))
            self.assertEqual((result['decoded_text'], result['dot_color'], result['result_id']), ('กภา', 'red', 1))
            self.assertEqual(calls, ['blue', 'red'])
            self.assertIs(worker.thread, original)
        finally:
            release.set()
            worker.stop()

    def test_hotkey_and_stale_results_cannot_keep_previous_text(self):
        scanner = RealTimeBrailleScanner(res_preset='fhd')
        worker = scanner.ai_worker = Mock()
        image = two_colors()
        cells, debug = scanner.detector.detect(image)
        result = dict(ai_fps=10., result_id=1, lang='thai', status='ok', reason=None,
            error=None, source_shape=image.shape, decoded_text=decode_cells(cells, 'thai'),
            cells=cells, dots=debug['dots'], verbose_results=decode_cells_verbose(cells, 'thai'),
            context=None, camera_roi=image, result_age=0.)
        worker.get_latest_results.return_value = result
        scanner._render_frame(image, 1)
        result['context'] = scanner._last_submitted_context
        scanner._render_frame(image, 1)
        self.assertTrue(scanner.history)
        with contextlib.redirect_stdout(io.StringIO()):
            scanner._handle_key(ord('c'), image, image)
        self.assertEqual(scanner.dot_color, 'red')
        self.assertFalse(scanner.history)
        scanner._render_frame(image, 1)
        self.assertEqual(worker.submit_frame.call_args.kwargs['dot_color'], 'red')
        self.assertFalse(scanner.history)
        worker.start.assert_not_called()
        worker.stop.assert_not_called()
        # An old result with the current context is still rejected by age.
        result['context'] = scanner._last_submitted_context
        result['result_age'] = 2.
        scanner._render_frame(image, 2)
        self.assertFalse(scanner.history)

    def test_diagnostic_records_actual_frame_color_after_switch(self):
        reader = YOLOBrailleDetector(dot_color='blue')
        image = two_colors()
        cells, debug = reader.detect(image, dot_color='red')
        with tempfile.TemporaryDirectory() as folder:
            report = json.loads(export_stream(image, cells, debug, folder, detector=reader).read_text(encoding='utf-8'))
            self.assertEqual(report['reader']['dot_color'], 'red')
            self.assertEqual(report['text'], 'กภา')


if __name__ == '__main__':
    unittest.main()
