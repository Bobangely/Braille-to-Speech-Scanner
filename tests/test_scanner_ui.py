"""Presentation must not change source pixels, recognition state or camera controls."""

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from camera_reader import RealTimeBrailleScanner
from live_preview import LivePreview
from scanner_ui import ScannerUI
from tests.test_yolo_cell_stream import page
from yolo_detector import YOLOBrailleDetector


class ScannerUITests(unittest.TestCase):
    def setUp(self):
        self.reader = YOLOBrailleDetector()
        self.ui = ScannerUI(self.reader._get_font)
        self.state = dict(text='วัฒนธรรม\nนิทานพยัญชนะ 3', lang='thai', color='blue',
            sharp='OFF', zoom=1., status='Detected', resolution='3840 x 2160',
            mode='COLOR / BLUE', cells=55, dots=170, lines=3, confirmed=6, required=6)

    def test_resize_keeps_preview_aspect_and_buttons_inside_controls(self):
        preview = np.full((216, 384, 3), 200, np.uint8)
        for w, h in [(640, 480), (800, 600), (1440, 900), (1920, 1080), (3840, 2160), (900, 1200)]:
            with self.subTest(size=(w, h)):
                self.ui.resize(w, h)
                canvas = self.ui.compose(preview, self.state)
                self.assertLessEqual(canvas.shape[1], 1920)
                self.assertLessEqual(canvas.shape[0], 1200)
                self.assertAlmostEqual(canvas.shape[1]/canvas.shape[0], w/h, delta=.005)
                x1, y1, x2, y2 = self.ui.preview_rect
                self.assertAlmostEqual((x2-x1)/(y2-y1), 384/216, delta=.025)
                self.assertTrue(np.all(canvas[y1:y2, x1:x2] == 200))
                for box, key in self.ui.buttons:
                    self.assertGreaterEqual(box[0], 0)
                    self.assertGreaterEqual(box[1], 0)
                    self.assertLessEqual(box[2], canvas.shape[1])
                    self.assertLessEqual(box[3], canvas.shape[0])
                    self.assertFalse(self.ui._inside(self.ui.preview_rect, box[0], box[1]))
                    self.assertEqual(self.ui.hit_key((box[0]+box[2])/2, (box[1]+box[3])/2), key)
        self.assertTrue(np.all(preview == 200))

    def test_caches_chrome_and_preserves_live_pixels_and_thai_text(self):
        frame = np.full((180, 320, 3), 40, np.uint8)
        state = deepcopy(self.state)
        with patch.object(self.ui, '_build_body', wraps=self.ui._build_body) as build:
            self.ui.compose(frame, state)
            frame[:] = 180
            canvas = self.ui.compose(frame, state)
            self.assertEqual(build.call_count, 1)
        x1, y1, x2, y2 = self.ui.preview_rect
        self.assertTrue(np.all(canvas[y1:y2, x1:x2] == 180))
        self.assertEqual(state, self.state)
        self.assertEqual(''.join(self.ui._lines), state['text'].replace('\n', ''))

    def test_compact_layout_gives_more_image_area_and_keeps_every_control(self):
        for width, height in ((1920, 1080), (3840, 2160)):
            with self.subTest(size=(width, height)):
                self.ui.resize(width, height)
                self.ui.compose(np.zeros((720, 1280, 3), np.uint8), self.state)
                x1, y1, x2, y2 = self.ui.preview_rect
                # Previous dashboard image was 1355 x 762 at the bounded 1080p canvas.
                self.assertGreater((x2-x1)*(y2-y1), 1355*762*1.2)
                self.assertEqual(self.ui.result_card[2]-self.ui.result_card[0], 320)
                self.assertEqual(self.ui.header_h, 110)
                controls = {key for _, key in self.ui.buttons}
                self.assertTrue({ord(k) for k in 'clvexzrpd hq'.replace(' ', '')} <= controls)
                left, top, right, bottom = self.ui.control_card
                for box, key in self.ui.buttons:
                    if key in (ord('['), ord(']')):
                        continue
                    self.assertTrue(left <= box[0] < box[2] <= right)
                    self.assertTrue(top <= box[1] < box[3] <= bottom)

    def test_scroll_can_reach_last_line_without_losing_combining_marks(self):
        text = '\n'.join(f'{i} ศักดิ์ วัฒนธรรม' for i in range(50))
        self.ui.compose(None, dict(self.state, text=text))
        self.assertGreater(self.ui.max_scroll, 0)
        self.assertEqual('\n'.join(self.ui._lines), text)
        self.ui.scroll(1000)
        self.assertEqual(self.ui.scroll_offset, self.ui.max_scroll)
        self.ui.compose(None, dict(self.state, text=text))
        self.assertTrue(self.ui._lines[-1].startswith('49 '))
        self.ui.compose(None, dict(self.state, text='ข้อความใหม่'))
        self.assertEqual(self.ui.scroll_offset, 0)

    def test_mouse_maps_letterboxed_preview_back_to_native_crop(self):
        scanner = RealTimeBrailleScanner(res_preset='4k', initial_zoom=2.)
        frame = np.zeros((2160, 3840, 3), np.uint8)
        crop, box = scanner._apply_zoom(frame)
        scanner._preview_crop_box = box
        scanner._preview_image_size = (1280, 720)
        scanner._compose_dashboard(cv2.resize(crop, (1280, 720)), {})
        x1, y1, x2, y2 = scanner._ui.preview_rect
        x, y = x1+.25*(x2-x1), y1+.75*(y2-y1)
        scanner._on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        np.testing.assert_allclose(scanner.zoom_center,
            [(box[0]+.25*(box[2]-box[0]))/3840, (box[1]+.75*(box[3]-box[1]))/2160])
        prior = scanner.zoom_center.copy()
        scanner._on_mouse(cv2.EVENT_LBUTTONDOWN, 1, 1, 0, None)
        self.assertEqual(scanner.zoom_center, prior)
        with patch.object(scanner, 'zoom_in') as zoom:
            scanner._on_mouse(cv2.EVENT_MOUSEWHEEL, x, y, 120, None)
            zoom.assert_called_once()

    def test_buttons_reuse_shortcuts_and_details_dont_reset_confirmation(self):
        scanner = RealTimeBrailleScanner(res_preset='480p')
        scanner._compose_dashboard(np.zeros((240, 320, 3), np.uint8), {})
        box, key = next(item for item in scanner._ui.buttons if item[1] == ord('c'))
        scanner._on_mouse(cv2.EVENT_LBUTTONDOWN, (box[0]+box[2])/2, (box[1]+box[3])/2, 0, None)
        self.assertEqual(scanner._ui_pending_key, ord('c'))
        with patch.object(scanner, 'cycle_dot_color') as color:
            scanner._handle_key(scanner._ui_pending_key, None, None)
            color.assert_called_once()
        scanner.history.extend(['ก']*6)
        context = scanner._context_generation
        scanner._handle_key(ord('h'), None, None)
        self.assertTrue(scanner.show_details)
        self.assertEqual(list(scanner.history), ['ก']*6)
        self.assertEqual(scanner._context_generation, context)

    def test_compact_buttons_dispatch_existing_actions_and_snapshot_pixels(self):
        scanner = RealTimeBrailleScanner(res_preset='480p')
        frame = np.full((240, 320, 3), 100, np.uint8)
        annotated = scanner._compose_dashboard(frame, {})
        for key, method in [('c', 'cycle_dot_color'), ('v', 'cycle_resolution'),
                            ('e', 'cycle_sharpness'), ('z', 'zoom_in'),
                            ('x', 'zoom_out'), ('r', 'reset_zoom')]:
            box, _ = next(item for item in scanner._ui.buttons if item[1] == ord(key))
            scanner._on_mouse(cv2.EVENT_LBUTTONDOWN, (box[0]+box[2])/2, (box[1]+box[3])/2, 0, None)
            with self.subTest(key=key), patch.object(scanner, method) as action:
                self.assertTrue(scanner._handle_key(scanner._ui_pending_key, frame, annotated))
                action.assert_called_once()
        with patch('camera_reader.os.makedirs'), patch('camera_reader.cv2.imwrite', return_value=True) as save:
            scanner._handle_key(ord('p'), frame, annotated)
            self.assertEqual(save.call_count, 2)
            self.assertIs(save.call_args_list[0].args[1], frame)
            self.assertIs(save.call_args_list[1].args[1], annotated)
        scanner._handle_key(ord('l'), frame, annotated)
        self.assertEqual(scanner.lang, 'english')
        scanner._handle_key(ord('l'), frame, annotated)
        self.assertEqual(scanner.lang, 'thai')
        self.assertFalse(scanner._handle_key(ord('q'), frame, annotated))
        self.assertFalse(scanner._handle_key(27, frame, annotated))

    def test_compact_overlay_is_presentation_only_and_cli_keeps_full_footer(self):
        image, _, _ = page([['1245', '16']])
        cells, debug = self.reader.detect(image)
        original = deepcopy(cells)
        detailed = self.reader.annotate_with_text(image, debug['dots'], cells)
        compact = self.reader.annotate_with_text(image, debug['dots'], cells, details=False, footer=False)
        self.assertGreater(detailed.shape[0], image.shape[0])
        self.assertEqual(compact.shape, image.shape)
        self.assertEqual(cells, original)
        preview = LivePreview(dashboard=True)
        result = dict(result_id=1, cells=cells, dots=debug['dots'], decoded_text='', verbose_results=[])
        with patch.object(preview._grids, 'update', wraps=preview._grids.update) as track:
            preview.render(image, self.reader, result, 'thai')
            preview.details = True
            preview.render(image, self.reader, result, 'thai')
            self.assertEqual(track.call_count, 1)

    def test_dashboard_does_not_submit_resized_frames_or_reprocess_on_ui_change(self):
        image, _, _ = page([['1245', '16']])
        scanner = RealTimeBrailleScanner()
        scanner.ai_worker = Mock()
        result = dict(result_id=0, ai_fps=0., context=None, lang='thai', status='waiting',
            cells=[], dots=[], decoded_text='', verbose_results=[], source_shape=None, error=None)
        scanner.ai_worker.get_latest_results.return_value = result
        _, submitted = scanner._render_frame(image, 1)
        scanner._ui.resize(800, 600)
        scanner.show_details = True
        scanner._render_frame(image, 1)
        scanner.ai_worker.submit_frame.assert_called_once()
        np.testing.assert_array_equal(scanner.ai_worker.submit_frame.call_args.args[0], image)
        np.testing.assert_array_equal(submitted, image)

    def test_status_and_error_notice_do_not_draw_over_preview(self):
        preview = np.full((180, 320, 3), 210, np.uint8)
        canvas = self.ui.compose(preview, self.state)
        x1, y1, x2, y2 = self.ui.preview_rect
        self.ui.notice(canvas, 'SCAN ERROR - retrying next frame')
        self.assertEqual(self.ui._state['status'], 'Error')
        self.assertTrue(np.all(canvas[y1:y2, x1:x2] == 210))
        self.ui.notice(canvas, 'UNREADABLE FRAME')
        self.assertEqual(self.ui._state['status'], 'Check image')
        self.ui.notice(canvas, 'CONFIRMING - 3/6 matching results')
        self.assertEqual(self.ui._state['status'], 'Scanning')
        self.ui.notice(canvas, 'CAMERA WAITING - retrying capture')
        self.assertEqual(self.ui._state['status'], 'Scanning')


if __name__ == '__main__':
    unittest.main()
