"""ตรวจโหมดจากเฟรมจริงและ fallback โดยไม่ใช้กล้องใน unit tests"""
import unittest
from unittest.mock import patch, Mock
import cv2
import numpy as np
from camera_reader import ThreadedCameraCapture, AsyncBrailleWorker, RealTimeBrailleScanner
from tests.test_camera_recovery import await_condition


class ModeCapture:
    def __init__(self, modes, broken=()):
        self.modes = modes
        self.broken = set(broken)
        self.requested = (1920, 1080)
        self.opened = True

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        w, h = self.requested
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            w = int(value)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            h = int(value)
        self.requested = w, h
        return True

    def get(self, prop):
        # จงใจรายงานค่าที่ขอ แม้ driver จะส่งเฟรม fallback
        return {cv2.CAP_PROP_FRAME_WIDTH: self.requested[0],
                cv2.CAP_PROP_FRAME_HEIGHT: self.requested[1],
                cv2.CAP_PROP_FPS: 30}.get(prop, 0)

    def read(self):
        if self.requested in self.broken:
            return False, None
        w, h = self.requested if self.requested in self.modes else self.modes[-1]
        return True, np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (h, w, 3))

    def release(self):
        self.opened = False


class CameraModesTests(unittest.TestCase):
    def check_modes(self, modes, expected, broken=()):
        cap = ModeCapture(modes, broken)
        with patch('camera_reader.cv2.VideoCapture', return_value=cap):
            camera = ThreadedCameraCapture()
        try:
            camera.start()
            await_condition(lambda: camera.get_status()[0] == 'ready')
            self.assertEqual((camera.actual_width, camera.actual_height), expected)
            self.assertEqual([(w, h) for _, w, h in getattr(camera, 'available_resolutions', [])],
                             [m for m in modes if m not in broken and m[1] >= 1080])
        finally:
            camera.release()

    def test_highest_verified_mode_is_default(self):
        self.check_modes([(3840, 2160), (2560, 1440), (1920, 1080)], (3840, 2160))

    def test_property_success_cannot_advertise_unsupported_mode(self):
        self.check_modes([(1920, 1080)], (1920, 1080))

    def test_unreadable_highest_mode_falls_back(self):
        self.check_modes([(3840, 2160), (2560, 1440), (1920, 1080)],
                         (2560, 1440), broken=[(3840, 2160)])

    def test_native_lower_mode_is_not_mislabeled(self):
        self.check_modes([(640, 480)], (640, 480))

    def test_mode_setting_exception_falls_back_without_startup_crash(self):
        cap = ModeCapture([(2560, 1440), (1920, 1080)])
        original = cap.set

        def reject_highest(prop, value):
            if prop == cv2.CAP_PROP_FRAME_WIDTH and value == 3840:
                raise RuntimeError('unsupported width')
            return original(prop, value)

        cap.set = reject_highest
        with patch('camera_reader.cv2.VideoCapture', return_value=cap), \
                self.assertLogs('camera_reader', level='WARNING'):
            camera = ThreadedCameraCapture(target_width=3840, target_height=2160)
            try:
                camera.start()
                await_condition(lambda: camera.get_status()[0] == 'ready')
                self.assertEqual((camera.actual_width, camera.actual_height), (2560, 1440))
            finally:
                camera.release()

    def test_sharp_zero_compatibility_and_nonzero_rejection(self):
        worker = AsyncBrailleWorker(Mock())
        frame = np.zeros((10, 10, 3), np.uint8)
        self.assertTrue(worker.submit_frame(frame, sharpness_strength=0))
        with self.assertRaises(ValueError):
            worker.submit_frame(frame, sharpness_strength=.7)

    def test_ui_does_not_offer_unverified_modes_or_sharpness(self):
        with patch('camera_reader.YOLOBrailleDetector'):
            scanner = RealTimeBrailleScanner()
        scanner._compose_dashboard(np.zeros((240, 320, 3), np.uint8), {})
        keys = [key for _, key in scanner._ui.buttons]
        self.assertNotIn(ord('v'), keys)
        self.assertNotIn(ord('e'), keys)

    def test_explicit_mode_override_is_preserved(self):
        cap = ModeCapture([(3840, 2160), (1920, 1080)])
        with patch('camera_reader.cv2.VideoCapture', return_value=cap):
            camera = ThreadedCameraCapture(preferred_resolution=(1920, 1080))
        try:
            camera.start()
            await_condition(lambda: camera.get_status()[0] == 'ready')
            self.assertEqual((camera.actual_width, camera.actual_height), (1920, 1080))
        finally:
            camera.release()

    def test_one_mode_does_not_reset_confirmation_on_resolution_key(self):
        with patch('camera_reader.YOLOBrailleDetector'):
            scanner = RealTimeBrailleScanner()
        scanner.available_resolutions = [('1080p', 1920, 1080)]
        scanner.history.append('test')
        generation = scanner._context_generation
        scanner._handle_key(ord('v'), None, None)
        self.assertEqual(scanner._context_generation, generation)
        self.assertEqual(list(scanner.history), ['test'])

    def test_mode_that_fails_after_probe_is_removed_and_falls_back(self):
        cap = ModeCapture([(3840, 2160), (2560, 1440), (1920, 1080)])
        with patch('camera_reader.cv2.VideoCapture', return_value=cap):
            camera = ThreadedCameraCapture()
        try:
            camera.start()
            await_condition(lambda: camera.get_status()[0] == 'ready')
            camera.set_resolution(1920, 1080)
            await_condition(lambda: camera.actual_width == 1920)
            cap.broken.add((3840, 2160))
            camera.set_resolution(3840, 2160)
            await_condition(lambda: camera.actual_width == 2560)
            self.assertNotIn(('4K', 3840, 2160), camera.available_resolutions)
        finally:
            camera.release()


if __name__ == '__main__':
    unittest.main()
