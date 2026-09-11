"""
Braille Reader - Real-time Webcam Scanner
===========================================
ระบบสแกนอักษรเบรลล์แบบสดผ่านกล้อง Webcam (YOLO cell stream)
รองรับความละเอียดสูง 4K UHD และ Full HD (FHD) พร้อมระบบปรับระดับความคมชัด (Multi-level Sharpness) & Digital Zoom

ฟีเจอร์เด่น:
- สลับความละเอียดแบบสดได้ทันทีระหว่าง 4K UHD (3840x2160), Full HD 1080p (1920x1080) และ HD 720p (กดปุ่ม V หรือ F)
- ปรับระดับความคมชัดของภาพได้ 5 ระดับ (OFF -> LOW -> MED -> HIGH -> ULTRA) ผ่านปุ่ม E
- ใช้ MJPG FourCC Codec เพื่อปลดล็อก Bandwidth สูงสุดของกล้อง 4K/FHD USB
- ระบบ Digital Zoom In / Zoom Out (1.0x - 4.0x) สำหรับขยายอักษรเบรลล์ขนาดเล็ก
- สแกนเฟรมวิดีโอแบบสดพร้อม 2x3 Virtual Grid Overlay และแถบคำแปลภาษาไทย/อังกฤษ
- ระบบ Frame Stabilization ตรวจจับความนิ่งของคำก่อนตัดสินใจ
- สลับภาษา (ไทย/อังกฤษ) ได้ทันทีผ่านคีย์ลัด
- บันทึกภาพ Snapshot พร้อมคำแปลลงโฟลเดอร์ output/

คีย์ลัด (Hotkeys):
  [V] / [F]     : สลับความละเอียดกล้อง (4K UHD <-> Full HD 1080p <-> HD 720p)
  [E]           : ปรับระดับความคมชัด (Sharpness: OFF -> LOW -> MED -> HIGH -> ULTRA)
  [Z] / [+] / [=] : ซูมเข้า (Zoom In +0.2x)
  [X] / [-] / [_] : ซูมออก (Zoom Out -0.2x)
  [R] / [0]     : รีเซ็ตการซูม (Reset Zoom 1.0x)
  [L]           : สลับภาษา (Thai <-> English)
  [P]           : ถ่ายภาพ Snapshot บันทึกลง output/
  [D]           : บันทึกภาพที่ AI อ่านจริง พร้อมเหตุผลรายเซลล์และรหัส Unicode
  [Q] / [ESC]   : ออกจากโปรแกรม

การใช้เมาส์ (Mouse Controls):
  • หมุนล้อเมาส์ขึ้น (Scroll Up)   : ซูมเข้า (Zoom In เล็งตรงตำแหน่งเมาส์)
  • หมุนล้อเมาส์ลง (Scroll Down) : ซูมออก (Zoom Out)
  • คลิกซ้ายบนภาพ                : เลื่อนจุดโฟกัส (Pan) ไปยังจุดที่คลิก
  • ดับเบิ้ลคลิก หรือ คลิกกลาง     : รีเซ็ตการซูมกลับ 1.0x
"""

import argparse
import inspect
import logging
import os
import sys
import time
import threading
from collections import Counter, deque

import cv2
import numpy as np
from PIL import Image, ImageDraw
from yolo_cell_stream import UnreadableBrailleFrame

# ปรับ encoding สำหรับ Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from yolo_detector import YOLOBrailleDetector
from live_preview import LivePreview
from decoder import decode_cells, decode_cells_verbose

logger = logging.getLogger(__name__)


def _valid_frame(frame):
    return (isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and
            frame.ndim == 3 and frame.shape[2] == 3 and frame.size > 0)


# รายการความละเอียดมาตรฐานที่สามารถสลับใช้งานได้
RESOLUTION_LIST = [
    ("4K UHD", 3840, 2160),
    ("Full HD", 1920, 1080),
    ("HD 720p", 1280, 720),
]

RESOLUTION_PRESETS = {
    '4k': (3840, 2160),
    '2k': (2560, 1440),
    '1080p': (1920, 1080),
    'fhd': (1920, 1080),
    '720p': (1280, 720),
    'hd': (1280, 720),
    '480p': (640, 480),
}

# ระดับความคมชัด (Level, Display Name, Unsharp Strength)
SHARPNESS_LEVELS = [
    (0, "OFF", 0.0),
    (1, "LOW (1.2x)", 0.35),
    (2, "MED (1.6x)", 0.70),
    (3, "HIGH (2.2x)", 1.15),
    (4, "ULTRA (3.0x)", 1.65),
]


class ThreadedCameraCapture:
    """Latest-frame capture. After start, only the capture thread touches the handle."""

    STALE_AFTER = 2.0
    RETRY_AFTER = 3.0
    MAX_RECONNECTS = 3

    def __init__(self, camera_id=0, target_width=1920, target_height=1080, target_fps=60):
        self.camera_id = camera_id
        self.target_width, self.target_height = target_width, target_height
        self.target_fps = target_fps
        self.actual_width, self.actual_height, self.actual_fps = target_width, target_height, target_fps
        self.cap = None
        self.frame, self.ret, self.frame_id = None, False, 0
        self.lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pending_resolution = None
        self._last_frame_at = None
        self._status, self._error = 'starting', None
        self.running, self.thread = False, None
        try:
            self._init_camera()
        except Exception:
            self._close_handle()
            raise

    def _close_handle(self):
        cap, self.cap = self.cap, None
        if cap is not None:
            cap.release()

    def _configure(self, width, height, fps):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        with self.lock:
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

    def _init_camera(self):
        backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_id, backend)
        if not self.cap.isOpened():
            self._close_handle()
            self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self._close_handle()
            with self.lock:
                self._status, self._error = 'unavailable', 'Unable to open camera'
            return False
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._configure(self.target_width, self.target_height, self.target_fps)
        # First read belongs to the worker too: a slow driver must not block UI setup.
        with self.lock:
            self._status, self._error = 'waiting', None
        return True

    def start(self):
        with self._lifecycle_lock:
            if self.thread is not None and self.thread.is_alive():
                return self.running and not self._stop_event.is_set()
            if self.cap is None or not self.cap.isOpened():
                return False
            self._stop_event.clear()
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, name="CameraGrabberThread", daemon=True)
            self.thread.start()
            return True

    def _capture_loop(self):
        failed_since, reconnects, failures = None, 0, 0
        last_error = None
        try:
            while self.running and not self._stop_event.is_set():
                with self.lock:
                    resolution, self._pending_resolution = self._pending_resolution, None
                if resolution is not None and self.cap is not None:
                    try:
                        self._configure(*resolution)
                    except Exception:
                        logger.exception("Camera resolution change failed")
                if self._stop_event.is_set() or not self.running:
                    break
                try:
                    if self.cap is None or not self.cap.isOpened():
                        raise RuntimeError('Camera handle is not open')
                    ret, frame = self.cap.read()
                    if not ret or not _valid_frame(frame):
                        raise UnreadableBrailleFrame('Camera returned an empty or invalid frame')
                except Exception as exc:
                    with self.lock:
                        self.ret, self.frame = False, None
                        self._status, self._error = 'waiting', str(exc)
                    signature = (type(exc), str(exc))
                    if signature != last_error:
                        logger.warning("Camera read failed: %s", exc,
                                       exc_info=not isinstance(exc, UnreadableBrailleFrame))
                        last_error = signature
                    now = time.monotonic()
                    failed_since = now if failed_since is None else failed_since
                    failures += 1
                    if now - failed_since >= self.RETRY_AFTER:
                        if reconnects >= self.MAX_RECONNECTS:
                            with self.lock:
                                self._status = 'unavailable'
                            break
                        reconnects += 1
                        logger.warning("Camera reconnect %s/%s after capture failure",
                                       reconnects, self.MAX_RECONNECTS)
                        with self.lock:
                            self._status = 'reconnecting'
                        try:
                            self._close_handle()
                            if not self._stop_event.is_set():
                                self._init_camera()
                        except Exception:
                            logger.exception("Camera reconnect failed")
                            self._close_handle()
                        failed_since = time.monotonic()
                    self._stop_event.wait(min(.25, .01 * failures))
                    continue

                with self.lock:
                    if self._stop_event.is_set() or not self.running:
                        break
                    if self._pending_resolution is not None:
                        continue  # This read began before the requested resolution change.
                    self.ret, self.frame = True, frame
                    self.frame_id += 1
                    self._last_frame_at = time.monotonic()
                    self._status, self._error = 'ready', None
                failed_since, reconnects, failures, last_error = None, 0, 0, None
        finally:
            try:
                self._close_handle()
            finally:
                with self.lock:
                    self.ret, self.frame, self.running = False, None, False
                    if self._stop_event.is_set():
                        self._status = 'stopped'

    def read_latest(self, with_id=False):
        with self.lock:
            stale = (self._last_frame_at is not None and
                     time.monotonic() - self._last_frame_at > self.STALE_AFTER)
            if not self.ret or self.frame is None or stale:
                return (False, None, self.frame_id) if with_id else (False, None)
            return (True, self.frame, self.frame_id) if with_id else (True, self.frame.copy())

    def get_status(self):
        with self.lock:
            if (self._status == 'ready' and self._last_frame_at is not None and
                    time.monotonic() - self._last_frame_at > self.STALE_AFTER):
                return 'waiting', 'Camera read is delayed; waiting for the driver'
            return self._status, self._error

    def set_resolution(self, width, height, fps=60):
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError('Camera dimensions and FPS must be positive')
        with self.lock:
            self.target_width, self.target_height, self.target_fps = width, height, fps
            self._pending_resolution = (width, height, fps)
            self.ret, self.frame = False, None
            return self.actual_width, self.actual_height

    def is_opened(self):
        with self._lifecycle_lock:
            if self.thread is not None:
                return self.thread.is_alive() and not self._stop_event.is_set()
            return self.cap is not None and self.cap.isOpened()

    def release(self, timeout=1.0):
        with self._lifecycle_lock:
            self.running = False
            self._stop_event.set()
            with self.lock:
                self.ret, self.frame, self._status = False, None, 'stopping'
            thread = self.thread
            if thread is None:
                self._close_handle()
                with self.lock:
                    self.ret, self.frame, self._status = False, None, 'stopped'
                return True
        if thread is not threading.current_thread():
            thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning("Camera read still in progress; its owner will release the handle when it returns")
            return False
        return True


class AsyncBrailleWorker:
    """
    Worker Thread แยกอิสระสำหรับ AI Detection & Decoder
    - นำเฟรมล่าสุดจากกล้องไปประมวลผล (YOLO cell stream)
    - แยกงาน AI ออกจากภาพกล้อง; ความเร็วจริงขึ้นกับกล้องและเครื่อง
    - รายงาน ai_fps ควบคู่ไปกับ display_fps
    """
    def __init__(self, detector, default_lang='thai'):
        self.detector = detector
        self.lang = default_lang
        parameters = inspect.signature(detector.detect).parameters
        self._detect_accepts_lang = ('lang' in parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()))

        # Shared input
        self._new_frame_event = threading.Event()
        self._input_lock = threading.Lock()
        self._pending_frame = None
        self._pending_strength = 0.0
        self._pending_context = None
        self._pending_frame_id = None
        self._last_submission = None
        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._last_error_log = (None, 0.0)

        # Shared output
        self._output_lock = threading.Lock()
        self.result_id = 0
        self.result_lang = default_lang
        self.error = None
        self.status, self.reason, self.stage = 'waiting', None, None
        self.busy_since = None
        self.result_frame_id = None
        self.result_frame = self.result_camera_roi = None
        self.result_strength = 0.0
        self.source_shape = None
        self.result_context = None
        self.debug_info = {}
        self.cells = []
        self.dots = []
        self.decoded_text = ""
        self.verbose_results = []
        self.ai_fps = 0.0

        self.running = False
        self.thread = None

    def start(self):
        with self._lifecycle_lock:
            if self.thread is not None and self.thread.is_alive():
                return self.running and not self._stop_event.is_set()
            self._stop_event.clear()
            self.running = True
            self.thread = threading.Thread(target=self._worker_loop, name="AIInferenceWorker", daemon=True)
            self.thread.start()
            return True

    def submit_frame(self, frame, lang=None, sharpness_strength=0.0, context=None, frame_id=None):
        """Keep one owned pending snapshot; never run concurrent model calls."""
        with self._input_lock:
            if self._stop_event.is_set():
                return False
            frame_lang = lang or self.lang
            identity = (frame_id, frame_lang, sharpness_strength, context)
            if frame_id is not None and identity == self._last_submission:
                return False
            self.lang = frame_lang
            self._pending_frame = frame.copy() if isinstance(frame, np.ndarray) else frame
            self._pending_strength = sharpness_strength
            self._pending_context = context
            self._pending_frame_id = frame_id
            self._last_submission = identity
            self._new_frame_event.set()
            return True

    def get_latest_results(self, include_frame=False):
        """ดึงผลลัพธ์การตรวจจับล่าสุดออกมาวาดบนหน้าจอกล้อง"""
        with self._output_lock:
            result = {
                'cells': list(self.cells),
                'dots': list(self.dots),
                'decoded_text': self.decoded_text,
                'verbose_results': list(self.verbose_results),
                'ai_fps': self.ai_fps,
                'result_id': self.result_id,
                'lang': self.result_lang,
                'error': self.error,
                'source_shape': self.source_shape,
                'context': self.result_context,
                'debug_info': dict(self.debug_info),
                'status': self.status,
                'reason': self.reason,
                'stage': self.stage,
                'frame_id': self.result_frame_id,
                'sharpness_strength': self.result_strength,
                'busy_seconds': (time.monotonic() - self.busy_since) if self.busy_since is not None else 0.0,
            }
            if include_frame:
                # The worker owns these arrays and never mutates them after publication.
                result['inference_frame'] = self.result_frame
                result['camera_roi'] = self.result_camera_roi
            return result

    def _worker_loop(self):
        try:
            while not self._stop_event.is_set():
                if not self._new_frame_event.wait(timeout=0.1):
                    continue
                with self._input_lock:
                    self._new_frame_event.clear()
                    if self._stop_event.is_set():
                        break
                    if self._pending_frame is None:
                        continue
                    frame_to_process = self._pending_frame
                    camera_roi = frame_to_process
                    frame_lang, strength = self.lang, self._pending_strength
                    frame_context, frame_id = self._pending_context, self._pending_frame_id
                    self._pending_frame = None

                started = time.monotonic()
                with self._output_lock:
                    self.busy_since = started
                stage = 'preprocessing'
                cells, dots, verbose_results, debug_info = [], [], [], {}
                decoded_text, error, reason, status, source_shape = '', None, None, 'ok', None
                try:
                    if not _valid_frame(frame_to_process):
                        raise UnreadableBrailleFrame('Expected a nonempty BGR uint8 camera frame')
                    source_shape = frame_to_process.shape
                    if strength:
                        h, w = frame_to_process.shape[:2]
                        small = cv2.resize(frame_to_process, (max(1, w//2), max(1, h//2)))
                        blur = cv2.resize(cv2.GaussianBlur(small, (5, 5), 0), (w, h))
                        frame_to_process = cv2.addWeighted(frame_to_process, 1 + strength, blur, -strength, 0)
                    stage = 'detection'
                    detect_options = {'lang': frame_lang} if self._detect_accepts_lang else {}
                    cells, debug_info = self.detector.detect(frame_to_process, **detect_options)
                    dots = debug_info.get('dots', [])
                    stage = 'decoding'
                    if cells:
                        decoded_text = decode_cells(cells, lang=frame_lang)
                        verbose_results = decode_cells_verbose(cells, lang=frame_lang)
                    warnings = [str(item['warning']) for item in verbose_results if item.get('warning')]
                    if warnings or '�' in decoded_text:
                        status, reason = 'uncertain', ', '.join(dict.fromkeys(warnings)) or 'incomplete_symbol'
                    elif not decoded_text.strip():
                        status, reason = 'empty', 'No readable Braille cells'
                except UnreadableBrailleFrame as exc:
                    status, reason = 'unreadable', str(exc)
                    cells, dots, verbose_results, debug_info, decoded_text = [], [], [], {}, ''
                except Exception as exc:
                    status, error, reason = 'error', str(exc) or type(exc).__name__, str(exc)
                    cells, dots, verbose_results, debug_info, decoded_text = [], [], [], {}, ''
                    source_shape = None
                    signature = (stage, type(exc), str(exc))
                    previous, logged_at = self._last_error_log
                    if signature != previous or time.monotonic() - logged_at >= 5:
                        logger.exception("Scan frame %s failed during %s", frame_id, stage)
                        self._last_error_log = (signature, time.monotonic())

                with self._output_lock:
                    self.busy_since = None
                    if self._stop_event.is_set():
                        break
                    inst_fps = 1.0 / max(1e-5, time.monotonic() - started)
                    self.ai_fps = .85*self.ai_fps + .15*inst_fps if self.ai_fps > 0 else inst_fps
                    self.cells, self.dots, self.verbose_results = cells, dots, verbose_results
                    self.decoded_text = decoded_text
                    self.result_id += 1
                    self.result_lang, self.result_frame_id = frame_lang, frame_id
                    self.result_frame = frame_to_process if _valid_frame(frame_to_process) else None
                    self.result_camera_roi = camera_roi if _valid_frame(camera_roi) else None
                    self.result_strength = strength
                    self.error, self.status, self.reason, self.stage = error, status, reason, stage
                    self.source_shape, self.result_context = source_shape, frame_context
                    self.debug_info = {key: debug_info[key] for key in (
                        'method', 'line_count', 'crop_count', 'crop_disagreements',
                        'overview_detections', 'tile_count', 'tile_budget_exceeded') if key in debug_info}
        finally:
            with self._output_lock:
                self.busy_since = None
            self.running = False

    def stop(self, timeout=1.0):
        with self._lifecycle_lock:
            # Serialize stop with publication and pending-frame ownership.
            with self._input_lock, self._output_lock:
                self.running = False
                self._stop_event.set()
                self._pending_frame, self._last_submission = None, None
                self.result_frame = self.result_camera_roi = None
                self._new_frame_event.set()
            thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("Inference still in progress; no replacement worker will be started")
                return False
        return True


class RealTimeBrailleScanner:
    """ตัวควบคุมการสแกนอักษรเบรลล์จากกล้องแบบ Real-time พร้อมระบบ YOLO / Async Preview, 4K/FHD, Zoom & Sharpening"""

    def __init__(
        self,
        camera_id=0,
        lang='thai',
        stability_threshold=6,
        res_preset='4k',
        width=None,
        height=None,
        initial_zoom=1.0,
        sharpness_level=2,
        detector_mode='yolo',
        yolo_conf=0.35,
        yolo_imgsz=1280,
        yolo_tile_size=1280,
        yolo_pipeline='stream',
        crop_batch=8,
        max_cells=256,
        proposal_confidence=None,
        model_path=None,
    ):
        self.camera_id = camera_id
        self.lang = lang.lower()
        self.stability_threshold = stability_threshold
        self.detector_mode = detector_mode.lower()
        self.yolo_conf = yolo_conf

        # กำหนดระดับความคมชัดเริ่มต้น (0 ถึง 4)
        self.sharpness_idx = max(0, min(len(SHARPNESS_LEVELS) - 1, int(sharpness_level)))

        self.available_resolutions = list(RESOLUTION_LIST)
        if (width is None) != (height is None):
            raise ValueError('Provide both camera width and height')
        if width is not None:
            if int(width) <= 0 or int(height) <= 0:
                raise ValueError('Camera dimensions must be positive')
            self.target_width, self.target_height = int(width), int(height)
        else:
            preset = (res_preset or '4k').lower()
            if preset not in RESOLUTION_PRESETS:
                raise ValueError(f'Unknown camera resolution: {res_preset}')
            self.target_width, self.target_height = RESOLUTION_PRESETS[preset]
        dimensions = (self.target_width, self.target_height)
        self.curr_res_idx = next((i for i, (_, w, h) in enumerate(RESOLUTION_LIST)
                                  if (w, h) == dimensions), 0)
        self.res_name = next((name for name, w, h in RESOLUTION_LIST
                              if (w, h) == dimensions), f'{dimensions[0]}x{dimensions[1]}')

        self.actual_width = self.target_width
        self.actual_height = self.target_height

        # Digital Zoom state
        self.zoom_level = max(1.0, min(4.0, float(initial_zoom)))
        self.zoom_center = [0.5, 0.5]  # [center_x, center_y] normalized (0.0 - 1.0)

        # State tracking
        self.history = deque(maxlen=stability_threshold)
        self._last_stability_result_id = 0
        self._last_submitted_frame_id = -1
        self._last_submitted_context = None
        self._last_error = None
        self._session_lock = threading.Lock()
        self._context_generation = 0
        self._rejected_result_id = None
        self._last_ui_error = (None, 0.0)
        self._preview = LivePreview(max_width=1280)

        # ตัวตรวจจับ YOLO
        initial_mode = self.detector_mode
        self.detector = YOLOBrailleDetector(
            model_path=model_path,
            confidence=self.yolo_conf,
            mode=initial_mode,
            imgsz=yolo_imgsz,
            tile_size=yolo_tile_size,
            yolo_pipeline=yolo_pipeline,
            crop_batch=crop_batch,
            max_cells=max_cells,
            proposal_confidence=proposal_confidence,
        )

        # Threaded components
        self.camera = None
        self.ai_worker = None

        # Dual FPS calculation (Display 60Hz vs AI Inference)
        self.display_fps = 0.0
        self.ai_fps = 0.0
        self._prev_frame_time = time.time()


    def cycle_sharpness(self):
        """สลับระดับความคมชัด: OFF -> LOW -> MED -> HIGH -> ULTRA"""
        self.sharpness_idx = (self.sharpness_idx + 1) % len(SHARPNESS_LEVELS)
        lvl_num, lvl_name, _ = SHARPNESS_LEVELS[self.sharpness_idx]
        self.history.clear()
        print(f"  ✨ ปรับระดับความคมชัด (Sharpness): Level {lvl_num} [{lvl_name}]")

    def cycle_resolution(self):
        """สลับความละเอียดแบบสดระหว่าง 4K UHD <-> Full HD 1080p <-> HD 720p"""
        self.curr_res_idx = (self.curr_res_idx + 1) % len(self.available_resolutions)
        name, target_w, target_h = self.available_resolutions[self.curr_res_idx]

        if self.camera:
            actual_w, actual_h = self.camera.set_resolution(target_w, target_h, fps=60)
            self.actual_width = actual_w
            self.actual_height = actual_h
        else:
            self.actual_width = target_w
            self.actual_height = target_h

        self.res_name = name
        self.history.clear()
        print(f"  📹 ขอเปลี่ยนความละเอียดกล้องเป็น: {name} ({target_w}x{target_h}); กำลังรอเฟรมใหม่")

    def zoom_in(self, step=0.2, center_norm=None):
        """ขยายภาพ (Zoom In)"""
        new_zoom = min(4.0, round(self.zoom_level + step, 2))
        if new_zoom != self.zoom_level:
            self.zoom_level = new_zoom
            if center_norm is not None:
                self.zoom_center = [
                    max(0.1, min(0.9, center_norm[0])),
                    max(0.1, min(0.9, center_norm[1])),
                ]
            self.history.clear()
            print(f"  🔍 ZOOM IN: {self.zoom_level:.1f}x")

    def zoom_out(self, step=0.2):
        """ลดการขยาย (Zoom Out)"""
        new_zoom = max(1.0, round(self.zoom_level - step, 2))
        if new_zoom != self.zoom_level:
            self.zoom_level = new_zoom
            if self.zoom_level <= 1.001:
                self.zoom_center = [0.5, 0.5]
            self.history.clear()
            print(f"  🔍 ZOOM OUT: {self.zoom_level:.1f}x")

    def reset_zoom(self):
        """รีเซ็ตการซูมกลับเป็น 1.0x"""
        if self.zoom_level != 1.0:
            self.zoom_level = 1.0
            self.zoom_center = [0.5, 0.5]
            self.history.clear()
            print("  🔍 RESET ZOOM: 1.0x")

    def _apply_zoom(self, frame):
        """
        ตัดภาพตามอัตราซูมและตำแหน่ง zoom_center โดยรักษาพิกเซลต้นฉบับสำหรับ YOLO
        """
        if self.zoom_level <= 1.001:
            return frame, None

        h, w = frame.shape[:2]
        crop_w = max(1, int(w / self.zoom_level))
        crop_h = max(1, int(h / self.zoom_level))

        cx = int(self.zoom_center[0] * w)
        cy = int(self.zoom_center[1] * h)

        # คำนวณขอบเขต Crop
        x1 = max(0, min(w - crop_w, cx - crop_w // 2))
        y1 = max(0, min(h - crop_h, cy - crop_h // 2))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        cropped = frame[y1:y2, x1:x2]
        # Keep native crop pixels for AI; only the display renderer resizes.
        return cropped, (x1, y1, x2, y2)


    def _draw_mini_viewfinder(self, image, orig_frame, crop_box):
        """Fit the overview inside the preview and map ROI from source coordinates."""
        if crop_box is None or self.zoom_level <= 1.001:
            return
        h, w = image.shape[:2]
        source_h, source_w = orig_frame.shape[:2]
        margin, vy = 5, 50
        available_w, available_h = w - 2*margin, h - vy - margin
        if available_w < 1 or available_h < 1:
            return
        scale = min(160/source_w, available_w/source_w, available_h/source_h)
        vw, vh = max(1, int(source_w*scale)), max(1, int(source_h*scale))
        vx = w - vw - margin
        mini = cv2.resize(orig_frame, (vw, vh))
        x1, y1, x2, y2 = crop_box
        cv2.rectangle(mini, (round(x1*vw/source_w), round(y1*vh/source_h)),
                      (round(x2*vw/source_w), round(y2*vh/source_h)), (0, 255, 255), 2)
        cv2.rectangle(mini, (0, 0), (vw-1, vh-1), (120, 120, 120), 1)
        cv2.putText(mini, f"ZOOM {self.zoom_level:.1f}x", (5, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 255, 255), 1, cv2.LINE_AA)
        image[vy:vy+vh, vx:vx+vw] = mini

    def _draw_top_hud(self, image, current_text, is_locked):
        """วาดแถบเมนูควบคุมและสถานะด้านบน (Top HUD)"""
        h, w = image.shape[:2]
        hud_h = 42

        # พื้นหลังแถบ HUD ด้านบน (Semi-transparent dark bar)
        hud = image[:hud_h]
        cv2.addWeighted(np.full_like(hud, 20), 0.75, hud, 0.25, 0, hud)
        cv2.line(image, (0, hud_h), (w, hud_h), (60, 80, 100), 1)

        # ข้อมูลสถานะ
        lang_text = "THAI [L]" if self.lang == 'thai' else "ENG [L]"

        # Detector Mode badge
        mode_badge = 'YOLO'
        mode_color = (0, 210, 255)

        # Resolution badge
        if self.actual_width >= 3840:
            res_badge = f"RES: 4K [{self.actual_width}x{self.actual_height}] [V/F]"
            res_color = (0, 255, 120)  # Bright green
        elif self.actual_width >= 1920:
            res_badge = f"RES: FHD [1080p] [V/F]"
            res_color = (100, 230, 255)  # Cyan
        else:
            res_badge = f"RES: {self.actual_width}x{self.actual_height} [V/F]"
            res_color = (180, 180, 180)

        # Sharpness badge
        lvl_num, lvl_name, _ = SHARPNESS_LEVELS[self.sharpness_idx]
        if lvl_num == 0:
            sharp_text = "SHARP: OFF [E]"
            sharp_color = (170, 170, 170)
        else:
            sharp_text = f"SHARP: LV.{lvl_num} ({lvl_name.split()[0]}) [E]"
            sharp_color = (255, 140, 255)  # Magenta/Pink

        fps_text = f"FPS: {self.display_fps:.1f} | AI: {self.ai_fps:.1f}"

        # Zoom badge
        if self.zoom_level > 1.001:
            zoom_text = f"ZOOM: {self.zoom_level:.1f}x [Z/X/R]"
            zoom_color = (0, 255, 255)  # Yellow
        else:
            zoom_text = "ZOOM: 1.0x [Z/X]"
            zoom_color = (200, 200, 200)

        # Status badge
        if '�' in current_text:
            status_text = "UNCERTAIN: ADJUST CAMERA / ROI"
            status_color = (0, 200, 255)
        elif is_locked and current_text:
            status_text = f"LOCKED: {current_text}"
            status_color = (0, 255, 120)  # Bright Green
        elif current_text:
            status_text = "DETECTING..."
            status_color = (0, 200, 255)  # Orange/Yellow
        else:
            status_text = "SCANNING..."
            status_color = (180, 180, 180)  # Gray

        # วาดข้อความ HUD
        cv2.putText(image, fps_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image, f"| {mode_badge}", (150, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 2, cv2.LINE_AA)
        cv2.putText(image, f"| {res_badge}", (280, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, res_color, 1, cv2.LINE_AA)
        cv2.putText(image, f"| {sharp_text}", (490, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sharp_color, 1, cv2.LINE_AA)
        cv2.putText(image, f"| {zoom_text}", (660, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, zoom_color, 1 if self.zoom_level <= 1.001 else 2, cv2.LINE_AA)
        cv2.putText(image, f"| {lang_text}", (925, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 255), 1, cv2.LINE_AA)
        if status_text.isascii():
            cv2.putText(image, f"| {status_text}", (1025, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 2, cv2.LINE_AA)
        elif w > 1025:
            # Reuse the detector's Thai font. Only convert the small HUD strip.
            strip = Image.fromarray(cv2.cvtColor(image[:hud_h, 1025:], cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(strip).text((0, 5), f"| {status_text}",
                                      font=self.detector._get_font(size=18, bold=True),
                                      fill=status_color[::-1])
            image[:hud_h, 1025:] = cv2.cvtColor(np.asarray(strip), cv2.COLOR_RGB2BGR)

        # วาดคำแนะนำปุ่มกดด้านล่างขวา
        tip = "[V/F] Res | [E] Sharp | [Z/X] Zoom | [P] Save | [D] Trace | [Q] Quit"
        cv2.putText(image, tip, (w - 535, hud_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    def _on_mouse(self, event, x, y, flags, param):
        """Event handler สำหรับการควบคุม Zoom และ Pan ด้วยเมาส์"""
        w, h = getattr(self, '_preview_image_size', (self.actual_width, self.actual_height))
        if y >= h:
            return
        crop = getattr(self, '_preview_crop_box', None)
        if crop is None:
            crop = (0, 0, self.actual_width, self.actual_height)
        # Mouse positions are preview pixels, then mapped through the native crop.
        x = crop[0] + x / max(1, w) * (crop[2] - crop[0])
        y = crop[1] + y / max(1, h) * (crop[3] - crop[1])
        h, w = self.actual_height, self.actual_width
        # ลูกกลิ้งเมาส์ -> ซูมเข้า/ออก
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.zoom_in(center_norm=(x / w, y / h))
            else:
                self.zoom_out()
        # ดับเบิ้ลคลิก หรือ คลิกกลาง -> รีเซ็ต Zoom
        elif event in (cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_MBUTTONDOWN):
            self.reset_zoom()
        # คลิกซ้ายเพื่อเลื่อนจุดโฟกัส (Pan) เมื่ออยู่ในโหมดซูม
        elif event == cv2.EVENT_LBUTTONDOWN and self.zoom_level > 1.0:
            self.zoom_center = [max(0.1, min(0.9, x / w)), max(0.1, min(0.9, y / h))]
            self.history.clear()

    def _render_frame(self, frame, frame_id):
        if not _valid_frame(frame):
            raise UnreadableBrailleFrame('Invalid camera frame')
        self.actual_height, self.actual_width = frame.shape[:2]
        now = time.monotonic()
        self.display_fps = .9*self.display_fps + .1/max(now-self._prev_frame_time, 1e-5)
        self._prev_frame_time = now
        enhanced_frame, crop_box = self._apply_zoom(frame)
        context = (self._context_generation, self.lang, self.detector_mode, self.sharpness_idx,
                   tuple(crop_box) if crop_box is not None else None, enhanced_frame.shape)
        if frame_id != self._last_submitted_frame_id or context != self._last_submitted_context:
            self.ai_worker.submit_frame(enhanced_frame, lang=self.lang,
                sharpness_strength=SHARPNESS_LEVELS[self.sharpness_idx][2],
                context=context, frame_id=frame_id)
            self._last_submitted_frame_id, self._last_submitted_context = frame_id, context
        result = self.ai_worker.get_latest_results()
        self.ai_fps = result['ai_fps']
        slow = result.get('busy_seconds', 0) >= 5
        if (result.get('context') != context or result['lang'] != self.lang or
                result['result_id'] == self._rejected_result_id or slow):
            result = dict(result, result_id=-1, cells=[], dots=[], decoded_text='',
                          verbose_results=[], error=None, status='waiting', reason=None)
            self.history.clear()
        signature = (result.get('status'), result.get('reason'))
        if signature != self._last_error:
            self._last_error = signature
            if result.get('status') in ('unreadable', 'uncertain'):
                logger.warning("Frame %s: %s", result.get('frame_id'), result.get('reason'))
        decoded_text = result['decoded_text']
        if result['result_id'] != self._last_stability_result_id:
            self._last_stability_result_id = result['result_id']
            if result.get('status') == 'ok' and decoded_text and '�' not in decoded_text:
                self.history.append(decoded_text)
            else:
                self.history.clear()
        is_locked = (len(self.history) >= self.stability_threshold and bool(decoded_text.strip())
                     and all(t == decoded_text for t in self.history))
        preview_result = result
        if result.get('source_shape') != enhanced_frame.shape:
            preview_result = dict(result, result_id=-1, cells=[], dots=[], decoded_text='', verbose_results=[])
        try:
            annotated = self._preview.render(enhanced_frame, self.detector, preview_result, self.lang)
            self._preview_image_size = (annotated.shape[1],
                round(enhanced_frame.shape[0]*annotated.shape[1]/enhanced_frame.shape[1]))
            self._preview_crop_box = crop_box
            self._draw_mini_viewfinder(annotated, frame, crop_box)
            self._draw_top_hud(annotated, decoded_text, is_locked)
        except Exception:
            # Do not repeatedly render the same broken result on every display tick.
            self._rejected_result_id = result['result_id']
            self._preview = LivePreview(max_width=1280)
            raise
        status = result.get('status')
        if slow:
            message = 'AI BUSY - waiting for inference; camera remains live'
        elif result['error']:
            message = 'SCAN ERROR - retrying next frame; see console'
        elif status == 'unreadable':
            message = 'UNREADABLE FRAME - adjust focus / lighting / ROI'
        elif status == 'uncertain':
            counts = Counter(token['warning'] for token in result['verbose_results']
                             if token.get('warning') and not token.get('consumed'))
            other = sum(counts.values()) - counts['empty_crop'] - counts['ambiguous_row_grid']
            message = (f"UNCERTAIN - empty crops: {counts['empty_crop']} | "
                       f"row grid: {counts['ambiguous_row_grid']} | other: {other} | [D] trace")
        else:
            message = ''
        if message:
            self._draw_notice(annotated, message)
        return annotated, enhanced_frame

    @staticmethod
    def _draw_notice(image, message):
        cv2.putText(image, message, (10, min(60, image.shape[0]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 160, 255), 1, cv2.LINE_AA)

    def _report_ui_error(self, stage, exc):
        signature = (stage, type(exc), str(exc))
        previous, logged_at = self._last_ui_error
        if signature != previous or time.monotonic() - logged_at >= 5:
            logger.exception("Camera session %s failed; keeping session open", stage)
            self._last_ui_error = (signature, time.monotonic())

    def _handle_key(self, key, enhanced_frame, annotated):
        if key in (ord('q'), ord('Q'), 27):
            return False
        if key in (ord('v'), ord('V'), ord('f'), ord('F')):
            self.cycle_resolution()
            self._context_generation += 1
        elif key in (ord('e'), ord('E')):
            self.cycle_sharpness()
        elif key in (ord('z'), ord('Z'), ord('+'), ord('='), ord('i'), ord('I')):
            self.zoom_in()
        elif key in (ord('x'), ord('X'), ord('-'), ord('_'), ord('o'), ord('O')):
            self.zoom_out()
        elif key in (ord('r'), ord('R'), ord('0')):
            self.reset_zoom()
        elif key in (ord('l'), ord('L')):
            self.lang = 'english' if self.lang == 'thai' else 'thai'
            self.history.clear()
            print(f"  🌐 สลับภาษาเป็น: {self.lang.upper()}")
        elif key in (ord('d'), ord('D')):
            # Live preview pixels can be newer than AI output. Export one atomic AI result.
            result = self.ai_worker.get_latest_results(include_frame=True)
            inference_frame = result.pop('inference_frame')
            camera_roi = result.pop('camera_roi')
            if inference_frame is None:
                raise ValueError('No completed inference frame available for diagnosis')
            from tools.diagnostics.export_yolo_stream import export_stream
            destination = os.path.join('output', f"diagnostic_{time.time_ns()}")
            manifest = export_stream(inference_frame, result['cells'],
                dict(result['debug_info'], method='yolo_cell_stream'), destination,
                lang=result['lang'], source=f"camera:{self.camera_id}", captured_image=camera_roi,
                metadata={key: result[key] for key in ('frame_id', 'result_id', 'status', 'reason',
                    'stage', 'error', 'context', 'sharpness_strength', 'decoded_text')}, detector=self.detector)
            print(f"  🔎 Diagnostic frame {result['frame_id']}: {manifest.resolve()}")
        elif key in (ord('p'), ord('P')):
            if enhanced_frame is None:
                raise ValueError('No live frame available for snapshot')
            os.makedirs('output', exist_ok=True)
            snap_path = f"output/snapshot_{time.time_ns()}.png"
            raw_path = snap_path.replace('.png', '_raw.png')
            saved_raw = cv2.imwrite(raw_path, enhanced_frame)
            saved_preview = cv2.imwrite(snap_path, annotated)
            if not (saved_raw and saved_preview):
                raise OSError('Snapshot could not be saved completely; check output storage')
            print(f"  📸 ภาพดิบ: {raw_path} | ภาพผลลัพธ์: {snap_path}")
        return True

    def run(self):
        if not self._session_lock.acquire(blocking=False):
            raise RuntimeError('This scanner session is already running')
        try:
            # A native call may outlive a timed join. Never reuse its model/handle concurrently.
            for owner in (self.camera, self.ai_worker):
                if owner is not None and owner.thread is not None and owner.thread.is_alive():
                    raise RuntimeError('Previous camera/inference shutdown is still pending')
            self._run_session()
        finally:
            self._session_lock.release()

    def _run_session(self):
        lvl_num, lvl_name, _ = SHARPNESS_LEVELS[self.sharpness_idx]
        print("=" * 70)
        print("   Braille-to-Speech Real-Time Scanner (YOLO / Async Preview)")
        print("=" * 70)
        print(f"  เปิดกล้อง ID:        Camera Index {self.camera_id}")
        print(f"  ความละเอียดเป้าหมาย:  {self.target_width}x{self.target_height} ({self.res_name})")
        print(f"  ระดับความคมชัด:      Level {lvl_num} [{lvl_name}]")
        print(f"  ภาษาเริ่มต้น:        {self.lang.upper()}")
        print(f"  อัตราการซูมเริ่มต้น:  {self.zoom_level:.1f}x")
        print()
        print("  คีย์ลัดและควบคุม:")
        print("    [V] / [F]       - สลับความละเอียด (4K UHD <-> Full HD 1080p <-> HD 720p)")
        print("    [E]             - ปรับระดับความคมชัด (OFF -> LOW -> MED -> HIGH -> ULTRA)")
        print("    [Z] / [+]       - ซูมเข้า (Zoom In)")
        print("    [X] / [-]       - ซูมออก (Zoom Out)")
        print("    [R] / [0]       - รีเซ็ตการซูม (Reset Zoom 1.0x)")
        print("    [Wheel Up/Down] - ซูมเข้า/ออกด้วยล้อเมาส์")
        print("    [L]             - สลับภาษา (Thai <-> English)")
        print("    [P]             - ถ่ายภาพ Snapshot ลง output/")
        print("    [Q]/[ESC]       - ออกจากโปรแกรม")
        print("=" * 70)
        print()

        # 1. เริ่มต้น Threaded Camera Grabber
        window_name = "Braille Real-Time Scanner [YOLO / Async Preview | 4K/FHD/Zoom]"
        window_created = False
        self.history.clear()
        self._context_generation += 1
        self._last_submitted_frame_id = -1
        self._last_stability_result_id = 0
        self._rejected_result_id = None
        self._preview = LivePreview(max_width=1280)
        try:
            self.camera = ThreadedCameraCapture(camera_id=self.camera_id,
                target_width=self.target_width, target_height=self.target_height, target_fps=60)
            if not self.camera.is_opened():
                print(f"[ERR] ไม่สามารถเปิดกล้อง ID: {self.camera_id}; ตรวจสอบการเชื่อมต่อหรือ --camera")
                return
            self.actual_width, self.actual_height = self.camera.actual_width, self.camera.actual_height
            self.ai_worker = AsyncBrailleWorker(self.detector, default_lang=self.lang)
            if not self.ai_worker.start():
                raise RuntimeError('Inference worker could not start')
            if not self.camera.start():
                raise RuntimeError('Camera capture thread could not start')
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            window_created = True
            disp_w = min(1600, self.actual_width)
            disp_h = int(disp_w * self.actual_height/max(1, self.actual_width))
            cv2.resizeWindow(window_name, disp_w, disp_h)
            cv2.setMouseCallback(window_name, self._on_mouse)
            self._prev_frame_time = time.monotonic()
            waiting, action_error = False, None

            while True:
                enhanced_frame, frame = None, None
                try:
                    ret, frame, frame_id = self.camera.read_latest(with_id=True)
                    if ret and _valid_frame(frame):
                        annotated, enhanced_frame = self._render_frame(frame, frame_id)
                        waiting = False
                    else:
                        if not waiting:
                            self._context_generation += 1
                            self.history.clear()
                            waiting = True
                        status, detail = self.camera.get_status()
                        annotated = np.zeros((240, 960, 3), np.uint8)
                        message = ('CAMERA UNAVAILABLE - check connection; Q to quit' if status == 'unavailable'
                                   else 'CAMERA WAITING - retrying capture; Q to quit')
                        self._draw_notice(annotated, message)
                        if detail:
                            cv2.putText(annotated, detail[:115], (10, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, (180, 180, 180), 1, cv2.LINE_AA)
                except Exception as exc:
                    self._report_ui_error('frame processing / preview', exc)
                    self.history.clear()
                    if _valid_frame(frame):
                        scale = min(1., 1280/frame.shape[1])
                        annotated = cv2.resize(frame, (max(1, round(frame.shape[1]*scale)),
                                                      max(1, round(frame.shape[0]*scale))))
                    else:
                        annotated = np.zeros((240, 960, 3), np.uint8)
                    self._draw_notice(annotated, 'FRAME ERROR - skipping frame; see console')
                if action_error:
                    self._draw_notice(annotated, action_error)
                cv2.imshow(window_name, annotated)
                # Pump events on success AND failure, including an unplugged camera.
                key = cv2.waitKey(10 if waiting else 1) & 0xFF
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                try:
                    if not self._handle_key(key, enhanced_frame, annotated):
                        break
                    if key != 255:
                        action_error = None
                except (cv2.error, OSError, ValueError, TypeError) as exc:
                    self._report_ui_error('keyboard action', exc)
                    action_error = 'ACTION FAILED - camera remains live; see console'
        finally:
            for owner, method in ((self.ai_worker, 'stop'), (self.camera, 'release')):
                if owner is not None:
                    try:
                        getattr(owner, method)()
                    except Exception:
                        logger.exception("Camera session cleanup failed: %s", method)
            if window_created:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Braille Reader — ระบบสแกนอักษรเบรลล์ Real-Time รองรับ 4K UHD / Full HD พร้อมปรับระดับความคมชัด & Digital Zoom',
    )
    parser.add_argument(
        '--camera', '--source', type=int, default=0, dest='camera',
        help='ID ของกล้อง Webcam (default: 0)',
    )
    parser.add_argument(
        '--res', type=str, default='4k',
        choices=['4k', '2k', '1080p', 'fhd', '720p', 'hd', '480p'],
        help='ความละเอียดเริ่มต้น: 4k (3840x2160), fhd/1080p (1920x1080), 720p (default: 4k)',
    )
    parser.add_argument(
        '--sharp', type=int, default=2,
        choices=[0, 1, 2, 3, 4],
        help='ระดับความคมชัดเริ่มต้น: 0=OFF, 1=LOW, 2=MED, 3=HIGH, 4=ULTRA (default: 2)',
    )
    parser.add_argument(
        '--lang', type=str, default='thai',
        choices=['thai', 'english'],
        help='ภาษาที่ต้องการอ่าน (default: thai)',
    )
    parser.add_argument(
        '--zoom', type=float, default=1.0,
        help='อัตราการซูมเริ่มต้น (เช่น 1.0, 1.5, 2.0; default: 1.0)',
    )
    parser.add_argument(
        '--stability', type=int, default=6,
        help='จำนวนเฟรมที่ข้อความต้องนิ่งก่อนออกเสียงอัตโนมัติ (default: 6)',
    )
    parser.add_argument(
        '--width', type=int, default=None,
        help='ความกว้างวิดีโอแบบระบุเจาะจง (px)',
    )
    parser.add_argument(
        '--height', type=int, default=None,
        help='ความสูงวิดีโอแบบระบุเจาะจง (px)',
    )
    parser.add_argument(
        '--detector', type=str, default='yolo',
        choices=['yolo'],
        help='YOLO เป็นตัวตรวจจับเดียว',
    )
    parser.add_argument(
        '--conf', type=float, default=0.35,
        help='Confidence threshold สำหรับ YOLO (default: 0.35)',
    )

    parser.add_argument('--imgsz', type=int, default=1280, help='YOLO input size (multiple of 32)')
    parser.add_argument('--tile-size', type=int, default=1280, help='Tile size; 0 disables extra inference')
    parser.add_argument('--yolo-pipeline', choices=['stream'], default='stream')
    parser.add_argument('--crop-batch', type=int, default=8, help='Cells per YOLO crop batch (1..32)')
    parser.add_argument('--max-cells', type=int, default=256, help='Maximum cells per frame')
    parser.add_argument('--proposal-conf', type=float, default=None, help='Overview threshold (default: min(0.3, conf))')
    parser.add_argument('--model', default=None, help='Local YOLO weights')
    args = parser.parse_args()

    scanner = RealTimeBrailleScanner(
        camera_id=args.camera,
        lang=args.lang,
        stability_threshold=args.stability,
        res_preset=args.res,
        width=args.width,
        height=args.height,
        initial_zoom=args.zoom,
        sharpness_level=args.sharp,
        detector_mode=args.detector,
        yolo_conf=args.conf,
        yolo_imgsz=args.imgsz,
        yolo_tile_size=args.tile_size,
        yolo_pipeline=args.yolo_pipeline,
        crop_batch=args.crop_batch,
        max_cells=args.max_cells,
        proposal_confidence=args.proposal_conf,
        model_path=args.model,
    )
    scanner.run()


if __name__ == '__main__':
    main()
