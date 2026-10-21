"""Fingertip tracking: lock one reading finger and report its pixel position."""
import unittest
from types import SimpleNamespace

import numpy as np

from fingertip import ActiveFingertipLock, FingertipTracker, fingertip_pixels

FRAME = np.zeros((480, 640, 3), np.uint8)


def hand(index_tip=(0.5, 0.25), middle_tip=(0.9, 0.9)):
    points = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
    points[8] = SimpleNamespace(x=index_tip[0], y=index_tip[1])
    points[12] = SimpleNamespace(x=middle_tip[0], y=middle_tip[1])
    return points


class FakeDetector:
    """Stands in for MediaPipe's HandLandmarker: returns queued hand lists."""

    def __init__(self, *hands_per_frame):
        self.frames = list(hands_per_frame)

    def detect_for_video(self, image, timestamp_ms):
        hands = self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]
        labels = [[SimpleNamespace(category_name='Right')] for _ in hands]
        return SimpleNamespace(hand_landmarks=hands, handedness=labels)


class FingertipTrackerTests(unittest.TestCase):
    def test_fingertip_pixels_maps_index_and_middle(self):
        self.assertEqual(fingertip_pixels([hand()], 640, 480), [(0, 8, 320, 120), (0, 12, 576, 432)])

    def test_tracker_returns_none_without_hands(self):
        tracker = FingertipTracker(detector=FakeDetector([]))
        self.assertIsNone(tracker.update(FRAME, 1))
        self.assertFalse(tracker.hand_present)

    def test_tracker_returns_locked_tip_pixels(self):
        tracker = FingertipTracker(detector=FakeDetector([hand(middle_tip=(0.1, 0.1))]))
        self.assertIsNone(tracker.update(FRAME, 1))  # not stable for the dwell yet
        self.assertTrue(tracker.hand_present)
        self.assertEqual(tracker.update(FRAME, 300), (320, 120))

    def test_locked_tip_follows_the_finger(self):
        moving = [[hand(index_tip=(0.5, 0.25))]] * 2 + [[hand(index_tip=(0.6, 0.25))]]
        tracker = FingertipTracker(detector=FakeDetector(*moving))
        tracker.update(FRAME, 1)
        tracker.update(FRAME, 300)
        self.assertEqual(tracker.update(FRAME, 333), (384, 120))

    def test_lock_is_released_when_finger_stays_gone(self):
        tracker = FingertipTracker(detector=FakeDetector([hand()], [hand()], [], [], [hand(index_tip=(0.2, 0.5))]))
        tracker.update(FRAME, 1)
        self.assertIsNotNone(tracker.update(FRAME, 300))
        self.assertIsNone(tracker.update(FRAME, 333))
        self.assertIsNone(tracker.update(FRAME, 700))  # gone longer than the grace period
        self.assertIsNone(tracker.update(FRAME, 733))  # a new finger must dwell again

    def test_one_missed_frame_keeps_the_lock(self):
        # MediaPipe drops single frames; the moving reading finger must not dwell again.
        frames = [[hand()], [hand()], [], [hand(index_tip=(0.6, 0.25))]]
        tracker = FingertipTracker(detector=FakeDetector(*frames))
        tracker.update(FRAME, 1)
        tracker.update(FRAME, 300)
        self.assertIsNone(tracker.update(FRAME, 333))
        self.assertEqual(tracker.update(FRAME, 366), (384, 120))

    def test_lock_waits_for_dwell(self):
        lock = ActiveFingertipLock(dwell_ms=250, radius_px=20)
        self.assertIsNone(lock.update([('Right', 8, 10, 10)], 0))
        self.assertEqual(lock.update([('Right', 8, 12, 10)], 250), ('Right', 8))


if __name__ == '__main__':
    unittest.main()
