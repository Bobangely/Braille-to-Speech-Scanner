"""Preview may start during mode probing; inference still waits for stable input."""
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import Mock, patch

from camera_reader import AsyncBrailleWorker, RealTimeBrailleScanner, ThreadedCameraCapture
from tests.test_camera_modes import ModeCapture
from tests.test_camera_recovery import FRAME, await_condition


class CameraStartupTests(unittest.TestCase):
    def test_first_frame_is_available_before_slow_resolution_probe_finishes(self):
        cap = ModeCapture([(3840, 2160), (2560, 1440), (1920, 1080)])
        entered, resume = threading.Event(), threading.Event()
        real_read = cap.read
        def read():
            if cap.requested == (3840, 2160) and not resume.is_set():
                entered.set()
                if not resume.wait(3):
                    raise TimeoutError('probe did not resume')
            return real_read()
        cap.read = read
        cap.release = Mock(wraps=cap.release)
        with patch('camera_reader.cv2.VideoCapture', return_value=cap):
            camera = ThreadedCameraCapture(target_width=3840, target_height=2160)
        try:
            camera.start()
            self.assertTrue(entered.wait(2))
            ret, frame = camera.read_latest()
            self.assertTrue(ret, 'valid initial frame was hidden behind resolution probing')
            self.assertEqual(frame.shape[:2], (1080, 1920))
            self.assertEqual(camera.get_status()[0], 'probing')
            with camera.lock:
                camera._last_frame_at = time.monotonic()-3
            self.assertTrue(camera.read_latest()[0], 'probing must hold its preview during a slow driver call')
            resume.set()
            await_condition(lambda: camera.get_status()[0] == 'ready')
            self.assertEqual((camera.actual_width, camera.actual_height), (3840, 2160))
            self.assertEqual(len(camera.available_resolutions), 3)
        finally:
            resume.set()
            camera.release()
        cap.release.assert_called_once()

    def test_release_during_initial_read_does_not_publish_or_reopen_camera(self):
        cap = ModeCapture([(1920, 1080)])
        entered, resume = threading.Event(), threading.Event()
        def read():
            entered.set()
            resume.wait(3)
            return True, FRAME.copy()
        cap.read = read
        cap.release = Mock(wraps=cap.release)
        with patch('camera_reader.cv2.VideoCapture', return_value=cap) as opened:
            camera = ThreadedCameraCapture()
            try:
                camera.start()
                self.assertTrue(entered.wait(2))
                self.assertFalse(camera.release(timeout=.01))
                cap.release.assert_not_called()
                resume.set()
                camera.thread.join(2)
                self.assertFalse(camera.thread.is_alive())
                self.assertFalse(camera.read_latest()[0])
                self.assertEqual(camera.get_status()[0], 'stopped')
                opened.assert_called_once()
                cap.release.assert_called_once()
            finally:
                resume.set()
                camera.release()

    def test_dashboard_renders_probe_frame_without_submitting_it_for_detection(self):
        scanner = RealTimeBrailleScanner(res_preset='480p')
        scanner.camera = SimpleNamespace(_needs_probe=True, available_resolutions=[])
        scanner.ai_worker = AsyncBrailleWorker(Mock())
        scanner.ai_worker.submit_frame = Mock()
        scanner._prev_frame_time = time.monotonic()
        annotated, source = scanner._render_frame(FRAME, 1)
        self.assertTrue(annotated.size)
        self.assertEqual(source.shape, FRAME.shape)
        scanner.ai_worker.submit_frame.assert_not_called()
        scanner.camera._needs_probe = False
        scanner._render_frame(FRAME, 1)
        scanner.ai_worker.submit_frame.assert_called_once()
