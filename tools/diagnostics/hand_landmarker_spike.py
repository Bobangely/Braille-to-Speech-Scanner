"""Throwaway MediaPipe hand-tracking probe for the mounted camera."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp


TIP_LANDMARKS = (8, 12)
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


def detection_rates(detections: list[bool]) -> tuple[float, float]:
    """Return overall and first-to-last-detection rates."""
    if not detections or not any(detections):
        return 0.0, 0.0
    first = detections.index(True)
    last = len(detections) - 1 - detections[::-1].index(True)
    active_rate = sum(detections[first : last + 1]) / (last - first + 1)
    return sum(detections) / len(detections), active_rate


def _open_camera(camera_id: int):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_id, backend)
    if not capture.isOpened() and backend != cv2.CAP_ANY:
        capture.release()
        capture = cv2.VideoCapture(camera_id)
    return capture


def run_probe(camera_id: int, model_path: Path, seconds: float, headless: bool) -> int:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Hand Landmarker model not found: {model_path}\nDownload: {MODEL_URL}"
        )

    capture = _open_camera(camera_id)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        # shortcut: spike tracks one hand only, revisit after mounted-camera validation.
        num_hands=1,
    )
    frame_count = 0
    detections = []
    inference_times = []
    started = time.perf_counter()

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

                if not headless:
                    height, width = frame.shape[:2]
                    for _, landmark_index, x, y in fingertip_pixels(
                        result.hand_landmarks, width, height
                    ):
                        color = (0, 255, 0) if landmark_index == 8 else (0, 165, 255)
                        label = "INDEX" if landmark_index == 8 else "MIDDLE"
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
                    cv2.imshow("MediaPipe Hand Landmarker Spike", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

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
    args = parser.parse_args()
    if args.headless and args.seconds <= 0:
        parser.error("--headless requires --seconds greater than zero")
    return run_probe(args.camera, args.model, args.seconds, args.headless)


if __name__ == "__main__":
    raise SystemExit(main())
