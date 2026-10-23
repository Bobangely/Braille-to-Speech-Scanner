"""Fingertip tracking for the finger reader (MediaPipe Hand Landmarker).

Only the index fingertip (landmark 8) reads: braille is read with the index finger,
and also tracking the middle finger made the lock flip between the two.
The first one that stays still for the dwell becomes the reading finger and
is followed until it disappears.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2

TIP_LANDMARKS = (8,)
PAD_FRACTION = 0.5  # reading point: this far from the tip toward the first joint (the pad under the nail)
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def fingertip_pixels(hand_landmarks, width: int, height: int,
                     pad_fraction: float = PAD_FRACTION) -> list[tuple[int, int, int, int]]:
    """Map the reading point of each reading finger to clamped frame pixels.

    Braille is felt with the pad, not the tip the camera sees: the point moves
    ``pad_fraction`` of the way from the tip toward the next joint down the finger.
    """
    points = []
    for hand_index, landmarks in enumerate(hand_landmarks):
        for landmark_index in TIP_LANDMARKS:
            tip, joint = landmarks[landmark_index], landmarks[landmark_index - 1]
            x = min(width - 1, max(0, round((tip.x + pad_fraction * (joint.x - tip.x)) * width)))
            y = min(height - 1, max(0, round((tip.y + pad_fraction * (joint.y - tip.y)) * height)))
            points.append((hand_index, landmark_index, x, y))
    return points


class ActiveFingertipLock:
    """Lock the first fingertip that remains stable for the dwell period."""

    def __init__(self, dwell_ms: int, radius_px: int):
        self.dwell_ms = dwell_ms
        self.radius_px = radius_px
        self.active: tuple[str, int] | None = None
        self._observed: dict[tuple[str, int], tuple[int, int, int]] = {}

    def update(
        self, candidates: list[tuple[str, int, int, int]], now_ms: int
    ) -> tuple[str, int] | None:
        if self.active is not None:
            return self.active

        present = {(hand, landmark) for hand, landmark, _, _ in candidates}
        self._observed = {
            key: value for key, value in self._observed.items() if key in present
        }
        for hand, landmark, x, y in candidates:
            key = (hand, landmark)
            previous = self._observed.get(key)
            moved = (
                previous is not None
                and math.hypot(x - previous[0], y - previous[1]) > self.radius_px
            )
            if previous is None or moved:
                self._observed[key] = (x, y, now_ms)
                continue
            if now_ms - previous[2] >= self.dwell_ms:
                self.active = key
                self._observed.clear()
                return self.active
        return None

    def reset(self) -> None:
        self.active = None
        self._observed.clear()


def labelled_fingertips(result, width: int, height: int,
                        pad_fraction: float = PAD_FRACTION) -> list[tuple[str, int, int, int]]:
    """(hand label, landmark, x, y) for every reading fingertip in a detection."""
    candidates = []
    for hand_index, landmark_index, x, y in fingertip_pixels(result.hand_landmarks, width, height, pad_fraction):
        handedness = result.handedness[hand_index]
        hand = handedness[0].category_name if handedness else f"Hand {hand_index + 1}"
        candidates.append((hand, landmark_index, x, y))
    return candidates


class FingertipTracker:
    """Return the locked reading fingertip in frame pixels, or None."""

    def __init__(self, model_path: Path = DEFAULT_MODEL, dwell_ms: int = 250, radius_px: int = 20,
                 detector=None, grace_ms: int = 300, pad_fraction: float = PAD_FRACTION):
        self._detector = detector or _create_detector(model_path)
        self._lock = ActiveFingertipLock(dwell_ms, radius_px)
        self.pad_fraction = pad_fraction
        self.grace_ms = grace_ms  # keep the lock through short detection drop-outs
        self._seen_ms = 0
        self.hand_present = False

    def update(self, frame_bgr, now_ms: int) -> tuple[int, int] | None:
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), max(1, int(now_ms)))
        self.hand_present = bool(result.hand_landmarks)
        height, width = frame_bgr.shape[:2]
        candidates = labelled_fingertips(result, width, height, self.pad_fraction)
        active = self._lock.update(candidates, now_ms)
        if active is None:
            return None
        for hand, landmark, x, y in candidates:
            if (hand, landmark) == active:
                self._seen_ms = now_ms
                return x, y
        if now_ms - self._seen_ms > self.grace_ms:
            self._lock.reset()  # the reading finger left the frame
        return None


def _create_detector(model_path: Path):
    import mediapipe as mp

    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Hand Landmarker model not found: {model_path}\nDownload: {MODEL_URL}")
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)
