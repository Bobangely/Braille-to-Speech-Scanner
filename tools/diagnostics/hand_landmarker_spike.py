"""Throwaway MediaPipe hand-tracking probe for the mounted camera."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp


TIP_LANDMARKS = (8, 12)
LOCK_DWELL_MS = 250
LOCK_RADIUS_PX = 20
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def fingertip_pixels(hand_landmarks, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Map index and middle fingertips to clamped frame pixels."""
    points = []
    for hand_index, landmarks in enumerate(hand_landmarks):
        for landmark_index in TIP_LANDMARKS:
            landmark = landmarks[landmark_index]
            x = min(width - 1, max(0, round(landmark.x * width)))
            y = min(height - 1, max(0, round(landmark.y * height)))
            points.append((hand_index, landmark_index, x, y))
    return points


def draw_hand_skeletons(frame, hand_landmarks) -> None:
    """Draw every detected hand with the complete MediaPipe hand topology."""
    for landmarks in hand_landmarks:
        mp.tasks.vision.drawing_utils.draw_landmarks(
            frame,
            landmarks,
            mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
        )


def resize_preview(frame):
    return cv2.resize(
        frame,
        (PREVIEW_WIDTH, PREVIEW_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def detection_rates(detections: list[bool]) -> tuple[float, float]:
    """Return overall and first-to-last-detection rates."""
    if not detections or not any(detections):
        return 0.0, 0.0
    first = detections.index(True)
    last = len(detections) - 1 - detections[::-1].index(True)
    active_rate = sum(detections[first : last + 1]) / (last - first + 1)
    return sum(detections) / len(detections), active_rate


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


def _open_camera(camera_id: int, width: int, height: int, fps: int):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_id, backend)
    if not capture.isOpened() and backend != cv2.CAP_ANY:
        capture.release()
        capture = cv2.VideoCapture(camera_id)
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


def run_probe(
    camera_id: int,
    model_path: Path,
    seconds: float,
    headless: bool,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
) -> int:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Hand Landmarker model not found: {model_path}\nDownload: {MODEL_URL}"
        )

    capture = _open_camera(camera_id, width, height, fps)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    frame_count = 0
    detections = []
    inference_times = []
    started = time.perf_counter()
    selector = ActiveFingertipLock(LOCK_DWELL_MS, LOCK_RADIUS_PX)

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                timestamp_ms = max(1, int((time.perf_counter() - started) * 1000))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                before_inference = time.perf_counter()
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    timestamp_ms,
                )
                inference_times.append((time.perf_counter() - before_inference) * 1000)
                frame_count += 1
                detections.append(bool(result.hand_landmarks))

                height, width = frame.shape[:2]
                candidates = []
                for hand_index, landmark_index, x, y in fingertip_pixels(
                    result.hand_landmarks, width, height
                ):
                    handedness = result.handedness[hand_index]
                    hand = (
                        handedness[0].category_name
                        if handedness
                        else f"Hand {hand_index + 1}"
                    )
                    candidates.append((hand, landmark_index, x, y))
                # shortcut: spatial stability stands in for cell occupancy until page mapping exists.
                active = selector.update(candidates, timestamp_ms)

                if not headless:
                    draw_hand_skeletons(frame, result.hand_landmarks)
                    for hand, landmark_index, x, y in candidates:
                        is_active = (hand, landmark_index) == active
                        active_color = (
                            (0, 255, 0) if landmark_index == 8 else (0, 165, 255)
                        )
                        color = active_color if is_active else (160, 160, 160)
                        finger = "INDEX" if landmark_index == 8 else "MIDDLE"
                        label = (
                            f"ACTIVE {hand} {finger}"
                            if is_active
                            else f"{hand} {finger}"
                        )
                        cv2.circle(frame, (x, y), 9, color, -1)
                        cv2.putText(
                            frame,
                            label,
                            (x + 12, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )
                    elapsed = max(time.perf_counter() - started, 1e-6)
                    cv2.putText(
                        frame,
                        f"FPS {frame_count / elapsed:.1f} | inference {inference_times[-1]:.1f} ms",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )
                    status = (
                        f"LOCKED {active[0]} {'INDEX' if active[1] == 8 else 'MIDDLE'}"
                        if active
                        else "WAITING FOR STABLE FINGERTIP"
                    )
                    cv2.putText(
                        frame,
                        f"{status} | R reset",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow(
                        "MediaPipe Hand Landmarker Spike",
                        resize_preview(frame),
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key in (ord("r"), ord("R")):
                        selector.reset()

                if seconds > 0 and time.perf_counter() - started >= seconds:
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - started, 1e-6)
    detection_rate, active_detection_rate = detection_rates(detections)
    median_inference = statistics.median(inference_times) if inference_times else 0.0
    print(
        f"frames={frame_count} fps={frame_count / elapsed:.1f} "
        f"detection_rate={detection_rate:.1%} active_detection_rate={active_detection_rate:.1%} "
        f"median_inference_ms={median_inference:.1f}"
    )
    return 0 if frame_count else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = parser.parse_args()
    if args.headless and args.seconds <= 0:
        parser.error("--headless requires --seconds greater than zero")
    return run_probe(
        args.camera,
        args.model,
        args.seconds,
        args.headless,
        args.width,
        args.height,
        args.fps,
    )


if __name__ == "__main__":
    raise SystemExit(main())
