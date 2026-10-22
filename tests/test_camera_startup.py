"""Keep the waiting dashboard until mode probing produces stable input."""
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from camera_reader import RealTimeBrailleScanner, ThreadedCameraCapture
from tests.test_camera_modes import ModeCapture
from tests.test_camera_recovery import FRAME, await_condition


class CameraStartupTests(unittest.TestCase):
    def test_preview_waits_until_slow_resolution_probe_finishes(self):
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
            self.assertFalse(camera.read_latest()[0], 'transitional native frames must not be displayed')
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

    def test_dashboard_explains_startup_then_clears_message_for_stable_preview(self):
        scanner = RealTimeBrailleScanner(res_preset='480p')
        scanner.camera = SimpleNamespace(_needs_probe=True, available_resolutions=[])
        with patch.object(scanner._ui, '_label', wraps=scanner._ui._label) as labels:
            scanner._compose_dashboard(None, {})
            self.assertEqual(scanner._ui._state['status'], 'Starting')
            self.assertIn('กำลังเปิดระบบอ่าน', [call.args[2] for call in labels.call_args_list])
            self.assertIsNone(scanner._ui.preview_rect)
            labels.reset_mock()
            scanner.camera._needs_probe = False
            scanner._compose_dashboard(FRAME, {})
            self.assertNotIn('กำลังเปิดระบบอ่าน', [call.args[2] for call in labels.call_args_list])
            self.assertIsNotNone(scanner._ui.preview_rect)
