"""
Braille Reader - Real-time Webcam Scanner
===========================================
ระบบสแกนอักษรเบรลล์ที่แต้มสีแบบสดผ่านกล้อง Webcam

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
from scanner_ui import ScannerUI
from yolo_cell_stream import UnreadableBrailleFrame

# ปรับ encoding สำหรับ Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from yolo_detector import YOLOBrailleDetector
from colored_braille import DOT_COLORS
from live_preview import LivePreview
from line_confirmation import LineConfirmation, reading_lines
from decoder import decode_cells, decode_cells_verbose
from finger_braille_reader import FingerBrailleTouchReader

logger = logging.getLogger(__name__)


def _uncertainty_notice(result):
    tokens = result.get('verbose_results', [])
    cells = result.get('cells', [])
    warnings = []
    for i, t in enumerate(tokens):
        w = t.get('warning')
        if not w or t.get('consumed'):
            continue
        # The grid uses persistent track IDs, which can have gaps after a loss.
        # Keep this notice on the same cell rather than its current list index.
        cell_id = cells[i].get('track_id', i + 1) if i < len(cells) else i + 1
        c_name = f'C{cell_id}'
        dots = t.get('dots', [])
        dots_str = ''.join(map(str, dots)) if dots else 'empty'
        
        w_text = w
        if w == 'unknown_or_incomplete_symbol':
            w_text = 'unknown/incomplete symbol'
        elif w == 'empty_crop':
            w_text = 'empty crop'
            
        warnings.append(f"{c_name} [{dots_str}]: {w_text}")

    # A text-level boundary review is not an unknown dot pattern in a cell.
    warnings.extend(f"text segmentation: {t['composition_warning']}"
                    for t in tokens if t.get('composition_warning'))
        
    if not warnings:
        return 'text assembly incomplete'
        
    msg = warnings[0]
    if len(warnings) > 1:
        msg += f' (+{len(warnings)-1} more)'
    return msg


def _valid_frame(frame):
    return (isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and
            frame.ndim == 3 and frame.shape[2] == 3 and frame.size > 0)


# รายการความละเอียดมาตรฐานที่สามารถสลับใช้งานได้
RESOLUTION_LIST = [
    ("4K", 3840, 2160),
    ("2K", 2560, 1440),
    ("1080p", 1920, 1080),
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



class ThreadedCameraCapture:
    """Latest-frame capture. After start, only the capture thread touches the handle."""

    STALE_AFTER = 2.0
    RETRY_AFTER = 3.0
    MAX_RECONNECTS = 3

    def __init__(self, camera_id=0, target_width=1920, target_height=1080, target_fps=60,
                 preferred_resolution=None):
        self.camera_id = camera_id
        self.target_width, self.target_height = target_width, target_height
        self.target_fps = target_fps
        self.preferred_resolution = preferred_resolution
        self.actual_width, self.actual_height, self.actual_fps = target_width, target_height, target_fps
        self.cap = None
        self.frame, self.ret, self.frame_id = None, False, 0
        self.lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pending_resolution = None
        self.available_resolutions = []
        self._needs_probe = False
        self._last_probe_frame = None
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
        self._needs_probe = True
        # First read belongs to the worker too: a slow driver must not block UI setup.
        with self.lock:
            self._status, self._error = 'waiting', None
        return True

    def _read_resolution(self, width, height, fps):
        """ยืนยันสองเฟรมติดกัน ป้องกันเฟรมเก่าหลัง driver เปลี่ยนโหมด"""
        if self._stop_event.is_set():
            return None
        try:
            self._configure(width, height, fps)
            consecutive = 0
            # shortcut: จำกัด 5 reads; กล้องที่ warm-up นานต้องปรับ budget จากการทดสอบจริง
            for _ in range(5):
                if self._stop_event.is_set():
                    return None
                ret, frame = self.cap.read()
                if ret and _valid_frame(frame):
                    if self._needs_probe:
                        self._last_probe_frame = frame
                    consecutive = consecutive + 1 if frame.shape[:2] == (height, width) else 0
                    if consecutive == 2:
                        with self.lock:
                            self.actual_width, self.actual_height = width, height
                        return frame
                else:
                    consecutive = 0
        except Exception:
            logger.warning('Camera mode %sx%s failed', width, height, exc_info=True)
        return None

    def _activate_resolution(self, width, height, fps):
        choices = [(width, height)] + [(w, h) for _, w, h in self.available_resolutions
                                      if w*h < width*height]
        for w, h in choices:
            if self._read_resolution(w, h, fps) is not None:
                return True
            with self.lock:
                self.available_resolutions = [m for m in self.available_resolutions
                                              if m[1:] != (w, h)]
        return False

    def _probe_resolutions(self):
        self._last_probe_frame = None
        with self.lock:
            self.available_resolutions = []
        supported = []
        for mode in RESOLUTION_LIST:
            if self._read_resolution(mode[1], mode[2], self.target_fps) is not None:
                supported.append(mode)
        with self.lock:
            self.available_resolutions = supported
        if self.preferred_resolution is not None:
            self._activate_resolution(*self.preferred_resolution, self.target_fps)
        elif supported:
            _, width, height = supported[0]
            self._activate_resolution(width, height, self.target_fps)
        elif self._last_probe_frame is not None:
            # กล้องต่ำกว่า 1080p ใช้เฟรม native แต่ไม่แสดงชื่อโหมดที่ไม่รองรับ
            height, width = self._last_probe_frame.shape[:2]
            self._read_resolution(width, height, self.target_fps)
        self._needs_probe = False
        self._last_probe_frame = None

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
                if self._needs_probe and self.cap is not None:
                    self._probe_resolutions()
                with self.lock:
                    resolution, self._pending_resolution = self._pending_resolution, None
                if resolution is not None and self.cap is not None:
                    try:
                        self._activate_resolution(*resolution)
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
                    self.actual_height, self.actual_width = frame.shape[:2]
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
        self._detect_options = {name for name in ('dot_color', 'context') if name in parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())}

        # Shared input
        self._new_frame_event = threading.Event()
        self._input_lock = threading.Lock()
        self._pending_frame = None
        self._pending_context = None
        self._pending_frame_id = None
        self._pending_color = None
        self._pending_submitted_at = None
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
        self.source_shape = None
        self.result_context = None
        self.result_color = None
        self.result_submitted_at = None
        self.debug_info = {}
        self.cells = []
        self.dots = []
        self.decoded_text = ""
        self.verbose_results = []
        self.line_results = []
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

    def submit_frame(self, frame, lang=None, sharpness_strength=0.0, context=None, frame_id=None,
                     dot_color=None):
        """Keep one owned pending snapshot; never run concurrent model calls."""
        if sharpness_strength != 0:
            raise ValueError('Sharpness has been removed; use sharpness_strength=0')
        with self._input_lock:
            if self._stop_event.is_set():
                return False
            frame_lang = lang or self.lang
            identity = (frame_id, frame_lang, context, dot_color)
            if frame_id is not None and identity == self._last_submission:
                return False
            self.lang = frame_lang
            self._pending_frame = frame.copy() if isinstance(frame, np.ndarray) else frame
            self._pending_context = context
            self._pending_frame_id = frame_id
            self._pending_color = dot_color
            self._pending_submitted_at = time.monotonic()
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
                'line_results': list(self.line_results),
                'ai_fps': self.ai_fps,
                'result_id': self.result_id,
                'lang': self.result_lang,
                'error': self.error,
                'source_shape': self.source_shape,
                'context': self.result_context,
                'dot_color': self.result_color,
                'result_age': (time.monotonic() - self.result_submitted_at)
                              if self.result_submitted_at is not None else 0.0,
                'debug_info': dict(self.debug_info),
                'status': self.status,
                'reason': self.reason,
                'stage': self.stage,
                'frame_id': self.result_frame_id,
                'sharpness_strength': 0.0,  # คง schema ของ Diagnostic เดิม ไม่มีการเพิ่มความคม
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
                    frame_lang = self.lang
                    frame_context, frame_id = self._pending_context, self._pending_frame_id
                    frame_color, submitted_at = self._pending_color, self._pending_submitted_at
                    self._pending_frame = None

                started = time.monotonic()
                with self._output_lock:
                    self.busy_since = started
                stage = 'preprocessing'
                cells, dots, verbose_results, debug_info = [], [], [], {}
                line_results = []
                decoded_text, error, reason, status, source_shape = '', None, None, 'ok', None
                try:
                    if not _valid_frame(frame_to_process):
                        raise UnreadableBrailleFrame('Expected a nonempty BGR uint8 camera frame')
                    source_shape = frame_to_process.shape
                    stage = 'detection'
                    detect_options = {'lang': frame_lang} if self._detect_accepts_lang else {}
                    options = dict(dot_color=frame_color, context=frame_context)
                    detect_options.update({key: options[key] for key in self._detect_options})
                    cells, debug_info = self.detector.detect(frame_to_process, **detect_options)
                    dots = debug_info.get('dots', [])
                    stage = 'decoding'
                    if cells:
                        decoded_text = decode_cells(cells, lang=frame_lang)
                        verbose_results = decode_cells_verbose(cells, lang=frame_lang)
                        line_results = reading_lines(cells, verbose_results, decoded_text)
                    warnings = [str(item[key]) for item in verbose_results
                                for key in ('warning', 'composition_warning') if item.get(key)]
                    if debug_info.get('grid_pending'):
                        warnings.append('unstable_grid')
                    if warnings or '�' in decoded_text:
                        status, reason = 'uncertain', ', '.join(dict.fromkeys(warnings)) or 'incomplete_symbol'
                    elif not decoded_text.strip():
                        status, reason = 'empty', 'No readable Braille cells'
                except UnreadableBrailleFrame as exc:
                    status, reason = 'unreadable', str(exc)
                    cells, dots, verbose_results, debug_info, decoded_text = [], [], [], {}, ''
                    line_results = []
                except Exception as exc:
                    status, error, reason = 'error', str(exc) or type(exc).__name__, str(exc)
                    cells, dots, verbose_results, debug_info, decoded_text = [], [], [], {}, ''
                    line_results = []
                    source_shape = None
                    signature = (stage, type(exc), str(exc))
                    previous, logged_at = self._last_error_log
                    if signature != previous or time.monotonic() - logged_at >= 5:
                        logger.exception("Scan frame %s failed during %s", frame_id, stage)
                        self._last_error_log = (signature, time.monotonic())

                with self._input_lock, self._output_lock:
                    self.busy_since = None
                    if self._stop_event.is_set():
                        break
                    if frame_context != self._pending_context or frame_color != self._pending_color:
                        continue  # A setting changed while this frame was being read.
                    inst_fps = 1.0 / max(1e-5, time.monotonic() - started)
                    self.ai_fps = .85*self.ai_fps + .15*inst_fps if self.ai_fps > 0 else inst_fps
                    self.cells, self.dots, self.verbose_results = cells, dots, verbose_results
                    self.line_results = line_results
                    self.decoded_text = decoded_text
                    self.result_id += 1
                    self.result_lang, self.result_frame_id = frame_lang, frame_id
                    self.result_frame = frame_to_process if _valid_frame(frame_to_process) else None
                    self.result_camera_roi = camera_roi if _valid_frame(camera_roi) else None
                    self.error, self.status, self.reason, self.stage = error, status, reason, stage
                    self.source_shape, self.result_context = source_shape, frame_context
                    self.result_color, self.result_submitted_at = frame_color, submitted_at
                    self.debug_info = {key: debug_info[key] for key in (
                        'method', 'line_count', 'crop_count', 'crop_disagreements',
                        'overview_detections', 'tile_count', 'tile_budget_exceeded', 'color',
                        'colored_dots', 'color_components', 'rejected_color_components', 'color_config',
                        'candidate_source', 'read_source', 'yolo_inference',
                        'roi_source', 'roi_box', 'roi_status', 'grid_tracking', 'grid_pending',
                        'grid_candidate_count', 'grid_planning_error', 'refined_colored_dots',
                        'split_color_components') if key in debug_info}
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
        res_preset='auto',
        width=None,
        height=None,
        initial_zoom=1.0,
        sharpness_level=0,
        detector_mode='yolo',
        yolo_conf=0.35,
        yolo_imgsz=1280,
        yolo_tile_size=1280,
        yolo_pipeline='stream',
        crop_batch=8,
        max_cells=256,
        proposal_confidence=None,
        model_path=None,
        dot_color='blue',
        roi_mode='off',
        scan_roi=None,
        tts_models=None,
        tts_python=None,
        tts_thai_backend='piper',
        tts_backend='local',
    ):
        self._speech = None
        if tts_models or tts_backend == 'edge':
            from piper_tts import PiperSpeech
            self._speech = PiperSpeech(tts_models or {}, python_executable=tts_python, thai_backend=tts_thai_backend, backend=tts_backend)
        self.camera_id = camera_id
        self.lang = lang.lower()
        self.stability_threshold = stability_threshold
        self.detector_mode = detector_mode.lower()
        self.yolo_conf = yolo_conf
        if dot_color not in DOT_COLORS:
            raise ValueError('Camera dot color must be blue or red')
        self.dot_color = dot_color
        if scan_roi is not None:
            if (len(scan_roi) != 4 or not np.isfinite(scan_roi).all() or
                    not (0 <= scan_roi[0] < scan_roi[2] <= 1 and
                         0 <= scan_roi[1] < scan_roi[3] <= 1)):
                raise ValueError('scan_roi must be normalized x1 y1 x2 y2 within 0..1')
            scan_roi = tuple(map(float, scan_roi))
        self.scan_roi = scan_roi

        if sharpness_level != 0:
            raise ValueError('Sharpness has been removed; use sharpness_level=0')

        self.available_resolutions = []
        self._auto_resolution = width is None and res_preset in (None, 'auto')
        if (width is None) != (height is None):
            raise ValueError('Provide both camera width and height')
        if width is not None:
            if int(width) <= 0 or int(height) <= 0:
                raise ValueError('Camera dimensions must be positive')
            self.target_width, self.target_height = int(width), int(height)
        else:
            preset = ('1080p' if res_preset in (None, 'auto') else res_preset).lower()
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
        self._line_confirmation = LineConfirmation(stability_threshold)
        self._last_stability_result_id = 0
        self._confirmed_text = ''
        self._confirmed_context = None
        self._last_submitted_frame_id = -1
        self._last_submitted_context = None
        self._last_error = None
        self._session_lock = threading.Lock()
        self._context_generation = 0
        self._rejected_result_id = None
        self._last_ui_error = (None, 0.0)
        self._preview = LivePreview(max_width=1280, dashboard=True)
        self.show_details = False
        self._ui_pending_key = None
        self._window_name = None
        self._window_checked_at = 0.

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
            dot_color=dot_color,
            roi_mode=roi_mode,
        )
        self._ui = ScannerUI(lambda size, bold: self.detector._get_font(size, bold)
                             if hasattr(self.detector, '_get_font') else None)

        # Threaded components
        self.camera = None
        self.ai_worker = None

        # Dual FPS calculation (Display 60Hz vs AI Inference)
        self.display_fps = 0.0
        self.ai_fps = 0.0
        self._prev_frame_time = time.time()
        self.finger_reader = FingerBrailleTouchReader(external_tts=self._speech)
        self._last_touched_info = None


    def _reset_confirmation(self):
        self.history.clear()
        self._line_confirmation.reset()
        if hasattr(self, 'finger_reader'):
            self.finger_reader.unlock_grid()


    def cycle_dot_color(self):
        self.dot_color = DOT_COLORS[(DOT_COLORS.index(self.dot_color) + 1) % len(DOT_COLORS)]
        self._context_generation += 1
        self._reset_confirmation()
        print(f"  สีจุดที่อ่าน: {self.dot_color.upper()} [C]")

    def cycle_resolution(self):
        """สลับเฉพาะโหมด 4K / 2K / 1080p ที่ตรวจจากเฟรมจริงแล้ว"""
        if len(self.available_resolutions) < 2:
            return
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
        self._reset_confirmation()
        print(f"   ขอเปลี่ยนความละเอียดกล้องเป็น: {name} ({target_w}x{target_h}); กำลังรอเฟรมใหม่")

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
            self._reset_confirmation()
            print(f"   ZOOM IN: {self.zoom_level:.1f}x")

    def zoom_out(self, step=0.2):
        """ลดการขยาย (Zoom Out)"""
        new_zoom = max(1.0, round(self.zoom_level - step, 2))
        if new_zoom != self.zoom_level:
            self.zoom_level = new_zoom
            if self.zoom_level <= 1.001:
                self.zoom_center = [0.5, 0.5]
            self._reset_confirmation()
            print(f"   ZOOM OUT: {self.zoom_level:.1f}x")

    def reset_zoom(self):
        """รีเซ็ตการซูมกลับเป็น 1.0x"""
        if self.zoom_level != 1.0:
            self.zoom_level = 1.0
            self.zoom_center = [0.5, 0.5]
            self._reset_confirmation()
            print("   RESET ZOOM: 1.0x")

    def _apply_zoom(self, frame):
        """
        ตัดภาพตามอัตราซูมและตำแหน่ง zoom_center โดยรักษาพิกเซลต้นฉบับสำหรับ YOLO
        """
        if self.zoom_level <= 1.001 and self.scan_roi is None:
            return frame, None

        h, w = frame.shape[:2]
        bx1, by1, bx2, by2 = (0, 0, w, h)
        if self.scan_roi is not None:
            left, top, right, bottom = self.scan_roi
            bx1, by1 = int(left*w), int(top*h)
            bx2, by2 = min(w, max(bx1+1, int(np.ceil(right*w)))), min(h, max(by1+1, int(np.ceil(bottom*h))))
        crop_w = max(1, int((bx2-bx1) / self.zoom_level))
        crop_h = max(1, int((by2-by1) / self.zoom_level))

        cx = int(self.zoom_center[0] * w)
        cy = int(self.zoom_center[1] * h)

        # คำนวณขอบเขต Crop
        x1 = max(bx1, min(bx2 - crop_w, cx - crop_w // 2))
        y1 = max(by1, min(by2 - crop_h, cy - crop_h // 2))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        cropped = frame[y1:y2, x1:x2]
        # Keep native crop pixels for AI; only the display renderer resizes.
        return cropped, (x1, y1, x2, y2)


    def _compose_scan_roi_preview(self, frame, annotated_roi, crop_box):
        """Show the full camera frame while keeping detection inside the ROI."""
        preview = cv2.convertScaleAbs(frame, alpha=0.35)
        x1, y1, x2, y2 = crop_box
        roi = cv2.resize(
            annotated_roi,
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_AREA,
        )
        preview[y1:y2, x1:x2] = roi
        cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 3)
        cv2.putText(
            preview,
            "SCAN ROI",
            (x1 + 8, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        return preview

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

    def _compose_dashboard(self, preview, result, current_text='', is_locked=False):
        cells = result.get('cells', [])
        starting = preview is None and getattr(self.camera, '_needs_probe', False) is True
        status = ('Starting' if starting else 'Error' if result.get('error') else
                  'Check image' if result.get('status') in ('uncertain', 'unreadable') else
                  'Detected' if is_locked else 'Scanning' if cells else 'Ready')
        display_text = current_text if is_locked else self._confirmed_text
        lines = result.get('line_confirmation', [])
        confirmed_lines = sum(line['confirmed'] for line in lines)
        partial = bool(confirmed_lines) and not is_locked
        if partial:
            display = []
            for index, line in enumerate(lines, 1):
                if line['confirmed']:
                    display.append(line['text'])
                elif self.lang in ('thai', 'th'):
                    label = 'รอตรวจสอบ' if line['status'] == 'uncertain' else f"รอยืนยัน {line['samples']}/{self.stability_threshold}"
                    display.append(f'[บรรทัด {index}: {label}]')
                else:
                    label = 'review required' if line['status'] == 'uncertain' else f"confirming {line['samples']}/{self.stability_threshold}"
                    display.append(f'[Line {index}: {label}]')
            display_text = '\n'.join(display)
        state = dict(text=display_text, result_held=bool(display_text) and not is_locked and not partial, lang=self.lang,
            camera_starting=starting,
            result_partial=partial, confirmed_lines=confirmed_lines, line_total=len(lines),
            color=self.dot_color, resolution_mode=self.res_name,
            resolution_modes=[name for name, _, _ in self.available_resolutions],
            details=self.show_details, zoom=self.zoom_level, status=status,
            resolution=f'{self.actual_width} x {self.actual_height}',
            mode=('YOLO ROI / ' if getattr(self.detector, 'roi_mode', 'off') == 'yolo'
                  else 'COLOR / ') + self.dot_color.upper(),
            cells=len(cells), dots=len(result.get('dots', [])),
            lines=len({cell.get('line_id', 0) for cell in cells}),
            confirmed=len(self.history) if is_locked or current_text else 0,
            required=self.stability_threshold,
            tts_enabled=self._speech is not None,
            tts_status=self._speech.status if self._speech is not None else '',
            finger_enabled=getattr(self.finger_reader, 'enabled', False),
            finger_mode=getattr(self.finger_reader, 'speech_mode', 'word'))
        return self._ui.compose(preview, state)

    def _draw_top_hud(self, image, current_text, is_locked):
        """Refresh cached metrics outside the camera pixels."""
        if image.shape[1::-1] == self._ui.size:
            self._ui.header(image, (self.display_fps, self.ai_fps))

    def _update_viewport(self):
        if self._window_name and time.monotonic() - self._window_checked_at >= .25:
            self._window_checked_at = time.monotonic()
            try:
                _, _, width, height = cv2.getWindowImageRect(self._window_name)
                self._ui.resize(width, height)
            except cv2.error:
                pass  # Some HighGUI backends do not expose a drawable rectangle.

    def _on_mouse(self, event, x, y, flags, param):
        """Event handler สำหรับการควบคุม Zoom และ Pan ด้วยเมาส์"""
        if event == cv2.EVENT_LBUTTONDOWN:
            key = self._ui.hit_key(x, y)
            if key is not None:
                self._ui_pending_key = key  # Dispatch through the existing keyboard path.
                return
        if event == cv2.EVENT_MOUSEWHEEL and self._ui._inside(self._ui.result_card, x, y):
            self._ui.scroll(-1 if flags > 0 else 1)
            return
        box = self._ui.preview_rect
        if box is None or not self._ui._inside(box, x, y):
            return
        w, h = getattr(self, '_preview_image_size', (self.actual_width, self.actual_height))
        x, y = (x-box[0])*w/(box[2]-box[0]), (y-box[1])*h/(box[3]-box[1])
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
            self._reset_confirmation()

    def _render_frame(self, frame, frame_id):
        if not _valid_frame(frame):
            raise UnreadableBrailleFrame('Invalid camera frame')
        self.actual_height, self.actual_width = frame.shape[:2]
        modes = getattr(self.camera, 'available_resolutions', None)
        if isinstance(modes, list):
            self.available_resolutions = list(modes)
            self.curr_res_idx = next((i for i, (_, w, h) in enumerate(modes)
                                      if (w, h) == (self.actual_width, self.actual_height)), 0)
            self.res_name = next((name for name, w, h in modes
                                  if (w, h) == (self.actual_width, self.actual_height)), 'Native')
        now = time.monotonic()
        self.display_fps = .9*self.display_fps + .1/max(now-self._prev_frame_time, 1e-5)
        self._prev_frame_time = now
        enhanced_frame, crop_box = self._apply_zoom(frame)
        context = (self._context_generation, self.lang, self.detector_mode,
                   tuple(crop_box) if crop_box is not None else None, enhanced_frame.shape, self.dot_color)
        if context != self._confirmed_context:
            # Historical text belongs to this language/color/camera crop only.
            self._confirmed_text = ''
            self._confirmed_context = context
            self._reset_confirmation()
            self._last_stability_result_id = 0
        yolo_paused = bool(getattr(self.finger_reader, 'grid_locked', False) and len(getattr(self.finger_reader, 'cached_cells', [])) > 0)
        if not yolo_paused:
            if frame_id != self._last_submitted_frame_id or context != self._last_submitted_context:
                self.ai_worker.submit_frame(enhanced_frame, lang=self.lang,
                    context=context, frame_id=frame_id, dot_color=self.dot_color)
                self._last_submitted_frame_id, self._last_submitted_context = frame_id, context
        result = self.ai_worker.get_latest_results(include_frame=True)
        self.ai_fps = result['ai_fps'] if not yolo_paused else 0.0
        slow = (result.get('busy_seconds', 0) >= 5) and not yolo_paused
        stale = (result.get('result_age', 0) > 1.0) and not yolo_paused
        if not yolo_paused and (result.get('context') != context or result['lang'] != self.lang or
                result['result_id'] == self._rejected_result_id or slow or stale):
            result = dict(result, result_id=-1, cells=[], dots=[], decoded_text='',
                          verbose_results=[], line_results=[], error=None, status='waiting', reason=None)
            self._reset_confirmation()
        signature = (result.get('status'), result.get('reason'))
        if signature != self._last_error:
            self._last_error = signature
            if result.get('status') in ('unreadable', 'uncertain'):
                logger.warning("Frame %s: %s", result.get('frame_id'), result.get('reason'))
        decoded_text = result['decoded_text']
        if result['result_id'] != self._last_stability_result_id:
            self._last_stability_result_id = result['result_id']
            if result.get('status') == 'ok' and decoded_text and '�' not in decoded_text:
                if self.history and self.history[-1] != decoded_text:
                    self.history.clear()
                self.history.append(decoded_text)
            else:
                self.history.clear()
        is_locked = (len(self.history) >= self.stability_threshold and bool(decoded_text.strip())
                     and all(t == decoded_text for t in self.history))
        self._line_confirmation.update(result)
        # Diagnostics keep the raw result; the live text/character labels only
        # publish after confirmation across distinct inference results.
        preview_result = dict(result, reader_name=f'COLOR {self.dot_color.upper()}',
            decoded_text=decoded_text if is_locked else '',
            verbose_results=result['verbose_results'] if is_locked else
                            self._line_confirmation.preview_tokens(result['verbose_results']),
            line_confirmation=list(self._line_confirmation.lines))
        if result.get('source_shape') != enhanced_frame.shape:
            self._reset_confirmation()
            is_locked = False
            preview_result = dict(preview_result, result_id=-1, cells=[], dots=[], decoded_text='', verbose_results=[], line_confirmation=[])
        try:
            self._preview.details = self.show_details
            annotated = self._preview.render(enhanced_frame, self.detector, preview_result, self.lang)
            if not self._preview.motion_valid and not yolo_paused:
                self._reset_confirmation()
                preview_result = dict(preview_result, line_confirmation=[])
                is_locked = False
                decoded_text = ''
            display_crop_box = crop_box
            if self.scan_roi is not None and crop_box is not None:
                annotated = self._compose_scan_roi_preview(frame, annotated, crop_box)
                display_crop_box = None
            self._preview_image_size = (annotated.shape[1],
                round(enhanced_frame.shape[0]*annotated.shape[1]/enhanced_frame.shape[1]))
            if display_crop_box is None:
                self._preview_image_size = annotated.shape[1], annotated.shape[0]
            self._preview_crop_box = display_crop_box
            self._draw_mini_viewfinder(annotated, frame, display_crop_box)
            if is_locked:
                # Publish only after both confirmation and preview motion checks.
                # Transient failed reads must not erase the previous confirmed text.
                self._confirmed_text = decoded_text
            camera_view = annotated[:self._preview_image_size[1]]
            if hasattr(self, 'finger_reader') and self.finger_reader.enabled:
                sx = self._preview_image_size[0] / max(1, enhanced_frame.shape[1])
                sy = self._preview_image_size[1] / max(1, enhanced_frame.shape[0])
                cells_list = result.get('cells', [])

                # Phase 1: สแกนเก็บพิกัดลง Cache จนกว่าข้อความทั้งหน้าจะนิ่ง
                if cells_list and not self.finger_reader.grid_locked and not self.finger_reader.is_hand_present:
                    self.finger_reader.update_grid_cache(
                        cells_list,
                        result.get('verbose_results', []),
                        scale_x=sx,
                        scale_y=sy,
                        decoded_text=decoded_text
                    )
                    # เมื่อยืนยันแล้วว่าสแกนติดครบทั้งหน้า ให้ Freeze Grid และ Pause YOLO ทันที
                    if is_locked and len(self.finger_reader.cached_cells) > 0:
                        self.finger_reader.lock_grid()
                        yolo_paused = True

                # Phase 2: รันเฉพาะ MediaPipe ตรวจจับนิ้วชี้ 100% CPU แบบ Zero-Latency
                if self.finger_reader.grid_locked or self.finger_reader.is_hand_present:
                    clean_preview = cv2.resize(enhanced_frame, (camera_view.shape[1], camera_view.shape[0]))
                    camera_view, self._last_touched_info = self.finger_reader.process(
                        camera_view, clean_frame=clean_preview
                    )
                else:
                    self._last_touched_info = None
            else:
                self._last_touched_info = None
            annotated = self._compose_dashboard(camera_view,
                preview_result, decoded_text, is_locked)
            self._draw_top_hud(annotated, decoded_text if is_locked else '', is_locked)
        except Exception:
            # Do not repeatedly render the same broken result on every display tick.
            self._rejected_result_id = result['result_id']
            self._reset_confirmation()
            self._preview = LivePreview(max_width=1280, dashboard=True)
            raise
        status = result.get('status')
        if yolo_paused:
            message = f'TOUCH READY - YOLO paused ({len(self.finger_reader.cached_cells)} cells, press R for new page)'
        elif slow:
            message = 'AI BUSY - waiting for inference; camera remains live'
        elif stale or not self._preview.motion_valid:
            message = 'TRACKING - waiting for a fresh readable frame'
        elif result['error']:
            message = 'SCAN ERROR - retrying next frame; see console'
        elif status == 'unreadable':
            message = 'UNREADABLE FRAME - adjust focus / lighting / ROI'
        elif result.get('debug_info', {}).get('grid_pending'):
            message = 'TRACKING - confirming changed cell layout'
        elif status == 'uncertain':
            message = _uncertainty_notice(result)
        elif decoded_text and not is_locked:
            message = f'CONFIRMING - {len(self.history)}/{self.stability_threshold} matching results'
        else:
            message = ''
        if getattr(self, '_last_touched_info', None):
            _c = self._last_touched_info.get('char', '')
            _name = self.finger_reader.get_spoken_name(_c) if hasattr(self, 'finger_reader') else ''
            message = f"TOUCH: [{_c}] {_name}" if _name else f"TOUCH: [{_c}]"
        if message:
            self._draw_notice(annotated, message)
        return annotated, enhanced_frame

    def _draw_notice(self, image, message):
        if image.shape[1::-1] == self._ui.size and hasattr(self._ui, 'notice_box'):
            self._ui.notice(image, message)
        else:
            # Last-resort presentation if the dashboard itself failed.
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
            if self._speech is not None:
                self._speech.stop()
            return False
        if key in (ord('s'), ord('S')):
            if self._speech is not None and self._confirmed_text:
                text = self._confirmed_text
                if self.lang in ('thai', 'th'):
                    from tts import prepare_thai_speech_text
                    text = prepare_thai_speech_text(text)
                self._speech.speak(text, self.lang)
        elif key in (ord('b'), ord('B')):
            if self._speech is not None:
                self._speech.stop()
        elif key in (ord('h'), ord('H')):
            self.show_details = not self.show_details
        elif key in (ord('['), ord(']')):
            self._ui.scroll(-1 if key == ord('[') else 1)
        elif key in (ord('c'), ord('C')):
            self.cycle_dot_color()
        elif key in (ord('f'), ord('F'), ord('t'), ord('T')):
            if hasattr(self, 'finger_reader'):
                self.finger_reader.enabled = not self.finger_reader.enabled
                if not self.finger_reader.enabled:
                    self.finger_reader.unlock_grid()
                elif len(self.finger_reader.cached_cells) > 0:
                    self.finger_reader.lock_grid()
                mode_str = "เปิด (ON)" if self.finger_reader.enabled else "ปิด (OFF)"
                print(f"  👉 สลับโหมดตรวจจับนิ้วชี้สัมผัสอ่าน: {mode_str} [F]")
        elif key in (ord('m'), ord('M')):
            if hasattr(self, 'finger_reader'):
                mode_name = self.finger_reader.toggle_mode()
                thai_desc = {'word': 'อ่านรวมเป็นคำ', 'char': 'สะกดทีละตัวอักษร', 'smart': 'อัจฉริยะ (รูด=อ่านคำ / แช่=สะกด)'}.get(mode_name, mode_name)
                print(f"  🔊 รูปแบบการออกเสียงนิ้วสัมผัส [M]: {thai_desc}")
        elif key in (ord('v'), ord('V')):
            if len(self.available_resolutions) > 1:
                self.cycle_resolution()
                self._context_generation += 1
        elif key in (ord('z'), ord('Z'), ord('+'), ord('='), ord('i'), ord('I')):
            self.zoom_in()
        elif key in (ord('x'), ord('X'), ord('-'), ord('_'), ord('o'), ord('O')):
            self.zoom_out()
        elif key in (ord('r'), ord('R'), ord('0')):
            self.reset_zoom()
            if hasattr(self, 'finger_reader'):
                self.finger_reader.unlock_grid()
        elif key in (ord('l'), ord('L')):
            if self._speech is not None:
                self._speech.stop()
            self.lang = 'english' if self.lang == 'thai' else 'thai'
            self._reset_confirmation()
            print(f"  🌐 สลับภาษาเป็น: {self.lang.upper()}")
        elif key in (ord('d'), ord('D')):
            # Live preview pixels can be newer than AI output. Export one atomic AI result.
            result = self.ai_worker.get_latest_results(include_frame=True)
            inference_frame = result.pop('inference_frame')
            camera_roi = result.pop('camera_roi')
            if inference_frame is None:
                raise ValueError('No completed inference frame available for diagnosis')
            from tools.diagnostics.export_yolo_stream import export_stream
            os.makedirs('output', exist_ok=True)
            existing = [d for d in os.listdir('output') if d.startswith('diag_') and d[5:].isdigit()]
            next_num = max((int(d[5:]) for d in existing), default=0) + 1
            destination = os.path.join('output', f"diag_{next_num:03d}")
            manifest = export_stream(inference_frame, result['cells'],
                dict(result['debug_info'], method=result['debug_info'].get('method',
                    'colored_cell_stream' if getattr(self.detector, 'dot_color', None) else 'yolo_cell_stream')), destination,
                lang=result['lang'], source=f"camera:{self.camera_id}", captured_image=camera_roi,
                metadata={key: result[key] for key in ('frame_id', 'result_id', 'status', 'reason',
                    'stage', 'error', 'context', 'sharpness_strength', 'decoded_text')}, detector=self.detector)
            print(f"  🔎 Diagnostic #{next_num:03d} (frame {result['frame_id']}): {manifest.resolve()}")
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
        print("=" * 70)
        print("   Painted Braille Real-Time Scanner (Async Preview)")
        print("=" * 70)
        print(f"  เปิดกล้อง ID:        Camera Index {self.camera_id}")
        print(f"  ความละเอียดเป้าหมาย:  {self.target_width}x{self.target_height} ({self.res_name})")
        print(f"  ภาษาเริ่มต้น:        {self.lang.upper()}")
        print(f"  อัตราการซูมเริ่มต้น:  {self.zoom_level:.1f}x")
        print()
        print("  คีย์ลัดและควบคุม:")
        print("    [V] / [F]       - สลับเฉพาะโหมดที่ตรวจแล้ว: 4K / 2K / 1080p")
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
        window_name = "Braille Scanner"
        window_created = False
        self._reset_confirmation()
        self._confirmed_text = ''
        self._confirmed_context = None
        self._context_generation += 1
        self._last_submitted_frame_id = -1
        self._last_stability_result_id = 0
        self._rejected_result_id = None
        self._preview = LivePreview(max_width=1280, dashboard=True)
        self._ui_pending_key = None
        try:
            self.camera = ThreadedCameraCapture(camera_id=self.camera_id,
            target_width=self.target_width, target_height=self.target_height, target_fps=60,
            preferred_resolution=(self.target_width, self.target_height))
            if not self.camera.is_opened():
                print(f"[ERR] ไม่สามารถเปิดกล้อง ID: {self.camera_id}; ตรวจสอบการเชื่อมต่อหรือ --camera")
                return
            self.actual_width, self.actual_height = self.camera.actual_width, self.camera.actual_height
            self.ai_worker = AsyncBrailleWorker(self.detector, default_lang=self.lang)
            if not self.ai_worker.start():
                raise RuntimeError('Inference worker could not start')
            if not self.camera.start():
                raise RuntimeError('Camera capture thread could not start')
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
            window_created = True
            self._window_name = window_name
            cv2.resizeWindow(window_name, 1440, 900)
            cv2.setMouseCallback(window_name, self._on_mouse)
            self._prev_frame_time = time.monotonic()
            waiting, action_error = False, None

            while True:
                self._update_viewport()
                enhanced_frame, frame = None, None
                try:
                    ret, frame, frame_id = self.camera.read_latest(with_id=True)
                    if ret and _valid_frame(frame):
                        annotated, enhanced_frame = self._render_frame(frame, frame_id)
                        waiting = False
                    else:
                        if not waiting:
                            self._context_generation += 1
                            self._reset_confirmation()
                            waiting = True
                        status, detail = self.camera.get_status()
                        annotated = self._compose_dashboard(None, {})
                        message = ('CAMERA UNAVAILABLE - check connection; Q to quit' if status == 'unavailable'
                                   else 'CAMERA WAITING - retrying capture; Q to quit')
                        self._draw_notice(annotated, message)
                        if detail and self.show_details:
                            self._draw_notice(annotated, message + ': ' + detail)
                except Exception as exc:
                    self._report_ui_error('frame processing / preview', exc)
                    self._reset_confirmation()
                    if _valid_frame(frame):
                        scale = min(1., 1280/frame.shape[1])
                        annotated = cv2.resize(frame, (max(1, round(frame.shape[1]*scale)),
                                                      max(1, round(frame.shape[0]*scale))))
                    else:
                        annotated = np.zeros((240, 960, 3), np.uint8)
                    try:
                        annotated = self._compose_dashboard(annotated, dict(error=True))
                    except Exception:
                        logger.exception('Dashboard fallback failed')
                    self._draw_notice(annotated, 'FRAME ERROR - skipping frame; see console')
                if action_error:
                    self._draw_notice(annotated, action_error)
                cv2.imshow(window_name, annotated)
                # Pump events on success AND failure, including an unplugged camera.
                key = cv2.waitKey(10 if waiting else 1) & 0xFF
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                try:
                    if key == 255 and self._ui_pending_key is not None:
                        key = self._ui_pending_key
                    self._ui_pending_key = None
                    if not self._handle_key(key, enhanced_frame, annotated):
                        break
                    if key != 255:
                        action_error = None
                except (cv2.error, OSError, ValueError, TypeError) as exc:
                    self._report_ui_error('keyboard action', exc)
                    action_error = 'ACTION FAILED - camera remains live; see console'
        finally:
            self._window_name = None
            for owner, method in ((self._speech, 'stop'), (self.ai_worker, 'stop'), (self.camera, 'release'), (getattr(self, 'finger_reader', None), 'close')):
                if owner is not None:
                    try:
                        getattr(owner, method)()
                    except Exception:
                        logger.exception("Camera session cleanup failed: %s", method)
            if window_created:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Braille Reader — ตรวจโหมดกล้องจริงและอ่านจากภาพต้นฉบับ',
    )
    parser.add_argument(
        '--camera', '--source', type=int, default=0, dest='camera',
        help='ID ของกล้อง Webcam (default: 0)',
    )
    parser.add_argument(
        '--res', type=str, default='auto',
        choices=['auto', '4k', '2k', '1080p', 'fhd', '720p', 'hd', '480p'],
        help='auto: เริ่มโหมดสูงสุดที่ตรวจจากเฟรมจริง; ระบุ preset/ขนาดเองเพื่อเลือกโหมด',
    )
    parser.add_argument(
        '--sharp', type=int, default=0,
        choices=[0],
        help='เลิกใช้ Sharpness แล้ว; คง --sharp 0 สำหรับคำสั่งเดิมเท่านั้น',
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
        help='จำนวนผลอ่านใหม่ที่ตรงกันก่อนแสดงข้อความยืนยัน (default: 6)',
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
    parser.add_argument('--tts-thai-backend', choices=['piper', 'mms'], default='piper')
    parser.add_argument('--tts-backend', choices=['local', 'edge'], default='local')
    parser.add_argument('--tts-thai-model', help='Local Thai Piper .onnx or MMS model directory')
    parser.add_argument('--tts-english-model', help='Local Piper English .onnx; requires matching .onnx.json')
    parser.add_argument('--tts-python', help='Python executable in the optional Piper environment')
    parser.add_argument('--dot-color', choices=DOT_COLORS, default='blue', help='Read only painted dots of this color')
    parser.add_argument('--roi', choices=['off', 'yolo'], default='off', help='Optional YOLO-dot ROI with tracking')
    parser.add_argument('--scan-roi', nargs=4, type=float, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='Fixed scan area in normalized camera coordinates 0..1; zoom stays inside it')
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
        dot_color=args.dot_color,
        roi_mode=args.roi,
        scan_roi=args.scan_roi,
        tts_models=({'thai': args.tts_thai_model, 'english': args.tts_english_model}
                    if args.tts_thai_model or args.tts_english_model else None),
        tts_python=args.tts_python,
        tts_thai_backend=args.tts_thai_backend,
        tts_backend=args.tts_backend,
    )
    scanner.run()


if __name__ == '__main__':
    main()
