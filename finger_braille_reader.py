"""
finger_braille_reader.py
===================================================================
Real-time Index Finger Skeleton Tracking and Braille Touch-to-Speech
ระบบตรวจจับเฉพาะปลายนิ้วชี้เพื่อสัมผัสอ่านอักษรเบรลล์และออกเสียงทันที
===================================================================
"""

import math
import logging
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

from thai_speech import THAI_CONSONANT_NAMES

logger = logging.getLogger(__name__)

# ตารางขานชื่อสระ วรรณยุกต์ และสัญลักษณ์ไทยสำหรับการสัมผัสอ่านรายตัว
THAI_VOWEL_TONE_NAMES = {
    'ะ': 'สระอะ',
    'า': 'สระอา',
    'ิ': 'สระอิ',
    'ี': 'สระอี',
    'ึ': 'สระอึ',
    'ื': 'สระอือ',
    'ุ': 'สระอุ',
    'ู': 'สระอู',
    'เ': 'สระเอ',
    'แ': 'สระแอ',
    'โ': 'สระโอ',
    'ใ': 'สระใอม้วน',
    'ไ': 'สระไอไม้มลาย',
    'ำ': 'สระอำ',
    '็': 'ไม้ไต่คู้',
    '่': 'ไม้เอก',
    '้': 'ไม้โท',
    '๊': 'ไม้ตรี',
    '๋': 'ไม้จัตวา',
    '์': 'การันต์',
    'ๆ': 'ไม้ยมก',
    'ฯ': 'ไปยาลน้อย',
    ' ': 'วรรค',
    '0': 'เลขศูนย์',
    '1': 'เลขหนึ่ง',
    '2': 'เลขสอง',
    '3': 'เลขสาม',
    '4': 'เลขสี่',
    '5': 'เลขห้า',
    '6': 'เลขหก',
    '7': 'เลขเจ็ด',
    '8': 'เลขแปด',
    '9': 'เลขเก้า',
}

ENGLISH_NAMES = {
    c: f"Letter {c.upper()}" for c in 'abcdefghijklmnopqrstuvwxyz'
}


class PersistentPiperClient:
    """Thread-safe client to communicate with persistent_piper_worker daemon."""

    def __init__(self, model_path, python_exe):
        self.model_path = str(model_path)
        self.python_exe = str(python_exe)
        self._lock = threading.Lock()
        self._proc = None
        self._start_process()

    def _start_process(self):
        worker_script = str(Path(__file__).parent / 'tools' / 'persistent_piper_worker.py')
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        self._proc = subprocess.Popen(
            [self.python_exe, worker_script, self.model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=flags
        )
        ready = self._proc.stdout.readline().strip()
        if "READY" not in ready:
            raise RuntimeError(f"Piper worker failed to start: {ready}")

    def speak(self, text):
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start_process()
            try:
                self._proc.stdin.write(json.dumps({'cmd': 'speak', 'text': text}) + '\n')
                self._proc.stdin.flush()
                resp = json.loads(self._proc.stdout.readline())
                return resp.get('status') == 'playing'
            except Exception:
                return False

    def precache(self, words):
        words_list = [w for w in words if w and w.strip()]
        if not words_list:
            return
        def _bg():
            with self._lock:
                if self._proc is None or self._proc.poll() is not None:
                    try:
                        self._start_process()
                    except Exception:
                        return
                try:
                    self._proc.stdin.write(json.dumps({'cmd': 'precache', 'words': words_list}) + '\n')
                    self._proc.stdin.flush()
                    self._proc.stdout.readline()
                except Exception:
                    pass
        threading.Thread(target=_bg, daemon=True).start()

    def stop(self):
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.stdin.write(json.dumps({'cmd': 'stop'}) + '\n')
                    self._proc.stdin.flush()
                    self._proc.stdout.readline()
                except Exception:
                    pass

    def close(self):
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.stdin.write(json.dumps({'cmd': 'quit'}) + '\n')
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=1)
                except Exception:
                    try:
                        self._proc.terminate()
                    except Exception:
                        pass
                self._proc = None


class NonBlockingSpeechQueue:
    """Non-blocking background thread for real-time instant character speech."""

    def __init__(self, fallback_tts=None):
        self._queue = queue.Queue(maxsize=4)
        self._stop_event = threading.Event()
        self._external_tts = fallback_tts
        self._persistent_piper = None
        if self._external_tts is not None and hasattr(self._external_tts, 'models') and hasattr(self._external_tts, 'python'):
            model_thai = self._external_tts.models.get('thai')
            python_exe = getattr(self._external_tts, 'python', None)
            if model_thai and python_exe and Path(str(model_thai)).is_file() and Path(str(python_exe)).is_file():
                try:
                    self._persistent_piper = PersistentPiperClient(str(model_thai), str(python_exe))
                except Exception as exc:
                    logger.warning("Could not launch PersistentPiperClient: %s", exc)

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def speak(self, text, interrupt=True):
        if not text:
            return
        if self._persistent_piper is not None:
            if interrupt:
                self._persistent_piper.stop()
            self._persistent_piper.speak(text)
            return

        if interrupt:
            if self._external_tts is not None and hasattr(self._external_tts, 'stop'):
                try:
                    self._external_tts.stop()
                except Exception:
                    pass
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def precache(self, words):
        if self._persistent_piper is not None:
            self._persistent_piper.precache(words)

    def _worker_loop(self):
        engine = None
        has_real_thai_voice = False
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty('rate', 170)
            eng.setProperty('volume', 1.0)
            voices = eng.getProperty('voices')
            for v in voices:
                v_name = v.name.lower()
                if any(k in v_name for k in ('thai', 'pattara', 'th-th', 'th_th')):
                    eng.setProperty('voice', v.id)
                    engine = eng
                    has_real_thai_voice = True
                    break
        except Exception as exc:
            logger.warning("pyttsx3 engine initialization fallback: %s", exc)

        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # 1. ใช้ออฟไลน์ Piper TTS เป็นลำดับแรกเพื่อการออกเสียงภาษาไทยที่ถูกต้องและตอบสนองทันที
                if self._external_tts is not None:
                    self._external_tts.speak(text, 'thai')
                # 2. ใช้ pyttsx3 เฉพาะเมื่อมีเสียงไทยแท้จริงติดตั้งในระบบเท่านั้น
                elif engine is not None and has_real_thai_voice:
                    engine.say(text)
                    engine.runAndWait()
                # 3. Fallback สำหรับตัวอักษรละติน/อังกฤษ
                elif engine is not None and any(ord(c) < 128 for c in text):
                    engine.say(text)
                    engine.runAndWait()
            except Exception as e:
                logger.error("Speech playback error: %s", e)

    def stop(self):
        if getattr(self, '_persistent_piper', None) is not None:
            try:
                self._persistent_piper.stop()
                self._persistent_piper.close()
            except Exception:
                pass
        self._stop_event.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


class IndexFingerSkeletonDetector:
    """
    ตรวจจับเฉพาะโครงกระดูกนิ้วชี้ (Landmark 5, 6, 7, 8) โดยตัดนิ้วอื่นออก
    รองรับทั้ง MediaPipe Tasks (v1.x+) และ MediaPipe Solutions (v0.10.x)

    v2: ลด thresholds 0.5->0.3, num_hands=2, EMA smoothing, miss grace 5 frames
    """

    SMOOTHING_ALPHA = 0.45
    MISS_GRACE_FRAMES = 5

    def __init__(self, model_path=None):
        self.available = False
        self._detector_type = None
        self._landmarker = None
        self._hands_solution = None
        self._smooth_tip = None
        self._smooth_pts = None
        self._miss_counter = 0
        self._last_tip = None
        self._last_pts = None

        default_model = Path(__file__).resolve().parent / 'models' / 'hand_landmarker.task'
        task_path = Path(model_path) if model_path else default_model

        # ลองใช้ MediaPipe Tasks API (v1.0+)
        if task_path.is_file():
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=str(task_path))
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.25,
                    min_hand_presence_confidence=0.25,
                    min_tracking_confidence=0.25
                )
                self._landmarker = vision.HandLandmarker.create_from_options(options)
                self._detector_type = 'tasks'
                self.available = True
                logger.info("IndexFingerSkeletonDetector loaded via MediaPipe Tasks.")
            except Exception as e:
                logger.warning("MediaPipe Tasks load failed: %s; trying legacy solutions...", e)

        # Fallback: ลองใช้ mp.solutions.hands (Legacy)
        if not self.available:
            try:
                import mediapipe as mp
                if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
                    self._hands_solution = mp.solutions.hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.25,
                        min_tracking_confidence=0.25
                    )
                    self._detector_type = 'solutions'
                    self.available = True
                    logger.info("IndexFingerSkeletonDetector loaded via MediaPipe Solutions.")
            except Exception as e:
                logger.warning("MediaPipe Solutions fallback failed: %s", e)


    def _ema_smooth(self, new_tip, new_pts):
        """Apply exponential moving average to reduce jitter."""
        a = self.SMOOTHING_ALPHA
        if self._smooth_tip is None:
            self._smooth_tip = np.array(new_tip, dtype=float)
            self._smooth_pts = [np.array(p, dtype=float) for p in new_pts] if new_pts else None
        else:
            self._smooth_tip = a * np.array(new_tip) + (1 - a) * self._smooth_tip
            if new_pts and self._smooth_pts and len(new_pts) == len(self._smooth_pts):
                self._smooth_pts = [a * np.array(p) + (1 - a) * sp
                                    for p, sp in zip(new_pts, self._smooth_pts)]
            elif new_pts:
                self._smooth_pts = [np.array(p, dtype=float) for p in new_pts]
        tip = (int(round(self._smooth_tip[0])), int(round(self._smooth_tip[1])))
        pts = [(int(round(p[0])), int(round(p[1]))) for p in self._smooth_pts] if self._smooth_pts else new_pts
        return tip, pts

    def _pick_best_index_finger(self, all_hands, w, h):
        """Pick the hand with the most extended index finger."""
        best_hand = None
        best_ext = -1
        for hand in all_hands:
            tip = hand[8]
            mcp = hand[5]
            ext = math.hypot((tip.x - mcp.x) * w, (tip.y - mcp.y) * h)
            if ext > best_ext:
                best_ext = ext
                best_hand = hand
        return best_hand

    def _refine_fingertip_contour(self, bgr_frame, tip_pt, index_pts):
        """
        ปรับตำแหน่งปลายนิ้วให้ตรงกับขอบเล็บ/ปลายเนื้อจริง (Nail Apex)
        โดยไม่ต้องพึ่งรอยพับข้อต่อ (Knuckle creases) ที่มักมองไม่เห็นเวลาหงายเล็บขึ้น
        """
        if tip_pt is None or not index_pts or len(index_pts) < 4:
            return tip_pt

        p_pip = np.array(index_pts[1], dtype=float)
        p_tip = np.array(tip_pt, dtype=float)
        vec = p_tip - p_pip
        dist = np.linalg.norm(vec)
        if dist < 1e-3:
            return tip_pt

        unit_vec = vec / dist
        h, w = bgr_frame.shape[:2]
        pad = int(max(30, min(80, dist * 0.9)))
        x1 = max(0, int(tip_pt[0] - pad))
        y1 = max(0, int(tip_pt[1] - pad))
        x2 = min(w, int(tip_pt[0] + pad))
        y2 = min(h, int(tip_pt[1] + pad))

        roi = bgr_frame[y1:y2, x1:x2]
        if roi.size == 0:
            return tip_pt

        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, (30, 133, 77), (255, 173, 127))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return tip_pt

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 40:
            return tip_pt

        pts = c.reshape(-1, 2) + np.array([x1, y1])
        projections = np.dot(pts - p_pip, unit_vec)
        best_idx = np.argmax(projections)
        refined_pt = (int(pts[best_idx][0]), int(pts[best_idx][1]))

        if math.hypot(refined_pt[0] - tip_pt[0], refined_pt[1] - tip_pt[1]) <= pad:
            return refined_pt
        return tip_pt

    def _detect_contour_apex(self, bgr_frame):
        """
        Fallback ตรวจจับปลายนิ้วชี้แบบ Direction-Aware เมื่อ MediaPipe ตรวจไม่ติด
        คำนวณทิศทางการยื่นของแขน/ข้อมือ (จากล่าง ขึ้นบน หรือจากข้าง) หาปลายนิ้วได้อย่างถูกต้อง
        """
        h, w = bgr_frame.shape[:2]
        ycrcb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, (30, 133, 77), (255, 173, 127))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 350:
            return None, None

        M = cv2.moments(c)
        if M['m00'] == 0:
            return None, None

        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        # วิเคราะห์ทิศทางที่แขนยื่นเข้ามาจากขอบภาพ (ล่าง, บน, ซ้าย หรือ ขวา)
        d_top = cy
        d_bottom = h - cy
        d_left = cx
        d_right = w - cx
        min_border = min(d_top, d_bottom, d_left, d_right)

        pts = c.reshape(-1, 2)
        if min_border == d_bottom:
            # แขนยื่นจากด้านล่าง (มุมมองปกติของการอ่านโต๊ะ) -> ปลายนิ้วคือจุดที่อยู่บนสุด (Y ต่ำสุด)
            tip_idx = int(np.argmin(pts[:, 1]))
        elif min_border == d_top:
            # แขนยื่นจากด้านบน -> ปลายนิ้วคือจุดที่อยู่ล่างสุด (Y สูงสุด)
            tip_idx = int(np.argmax(pts[:, 1]))
        elif min_border == d_right:
            # แขนยื่นจากด้านขวา -> ปลายนิ้วคือจุดที่อยู่ซ้ายสุด (X ต่ำสุด)
            tip_idx = int(np.argmin(pts[:, 0]))
        else:
            # แขนยื่นจากด้านซ้าย -> ปลายนิ้วคือจุดที่อยู่ขวาสุด (X สูงสุด)
            tip_idx = int(np.argmax(pts[:, 0]))

        extreme_pt = pts[tip_idx]
        # ร่นจุดเป้าเล็งลงบนตัวเล็บ 15% เข้าหาศูนย์กลางเนื้อนิ้ว
        nail_x = extreme_pt[0] * 0.85 + cx * 0.15
        nail_y = extreme_pt[1] * 0.85 + cy * 0.15
        apex = (int(round(nail_x)), int(round(nail_y)))
        return apex, [apex]

    def detect_index_finger(self, bgr_frame):
        """
        ตรวจจับเฉพาะปลายนิ้วชี้ โดยล็อกพิกัดจุดสัมผัสให้อยู่บนตัวเล็บนิ้วชี้พอดี

        v2: เพิ่ม multi-hand selection, EMA smoothing, miss grace period
        Returns:
            tip_pt: (x, y) พิกัดเป้าเล็งบนเล็บนิ้วชี้ หรือ None
            index_pts: [(x, y), ...] พิกัดข้อนิ้ว หรือ None
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return None, None

        h, w = bgr_frame.shape[:2]
        tip_pt, index_pts = None, None

        if self.available:
            if self._detector_type == 'tasks':
                try:
                    import mediapipe as mp
                    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = self._landmarker.detect(mp_image)
                    if result.hand_landmarks:
                        # เลือกมือที่นิ้วชี้ยื่นมากที่สุด
                        hand = self._pick_best_index_finger(result.hand_landmarks, w, h)
                        if hand is None:
                            hand = result.hand_landmarks[0]
                        index_pts = [
                            (int(hand[idx].x * w), int(hand[idx].y * h))
                            for idx in (5, 6, 7, 8)
                        ]
                        p7 = np.array(index_pts[2], dtype=float)
                        p8 = np.array(index_pts[3], dtype=float)
                        nail_center = p7 + 0.80 * (p8 - p7)
                        tip_pt = (int(round(nail_center[0])), int(round(nail_center[1])))
                except Exception as exc:
                    logger.error("Task inference error: %s", exc)

            elif self._detector_type == 'solutions':
                try:
                    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    result = self._hands_solution.process(rgb)
                    if result.multi_hand_landmarks:
                        hand = result.multi_hand_landmarks[0]
                        index_pts = [
                            (int(hand.landmark[idx].x * w), int(hand.landmark[idx].y * h))
                            for idx in (5, 6, 7, 8)
                        ]
                        p7 = np.array(index_pts[2], dtype=float)
                        p8 = np.array(index_pts[3], dtype=float)
                        nail_center = p7 + 0.80 * (p8 - p7)
                        tip_pt = (int(round(nail_center[0])), int(round(nail_center[1])))
                except Exception as exc:
                    logger.error("Solutions inference error: %s", exc)

        # Apply EMA smoothing if MediaPipe returned a result
        if tip_pt is not None:
            tip_pt, index_pts = self._ema_smooth(tip_pt, index_pts)
            self._last_tip = tip_pt
            self._last_pts = index_pts
            self._miss_counter = 0
            return tip_pt, index_pts

        # Miss grace: keep showing last known position for a few frames
        if self._last_tip is not None and self._miss_counter < self.MISS_GRACE_FRAMES:
            self._miss_counter += 1
            return self._last_tip, self._last_pts

        # Reset smoothing state after grace period expires
        self._smooth_tip = None
        self._smooth_pts = None
        self._last_tip = None
        self._last_pts = None

        # Fallback: หาก MediaPipe ตรวจไม่ติด ให้ใช้ Contour ร่นพิกัดเข้ามาบนตัวเล็บ
        tip_pt, index_pts = self._detect_contour_apex(bgr_frame)
        return tip_pt, index_pts

    def draw_skeleton(self, image, index_pts, tip_pt=None):
        """วาดโครงกระดูกเฉพาะนิ้วชี้และเป้าเล็งล็อกบนเล็บนิ้วชี้พอดี"""
        target_pt = tip_pt or (index_pts[3] if index_pts and len(index_pts) >= 4 else None)

        if index_pts and len(index_pts) >= 4:
            # วาดเส้นกระดูกนิ้วชี้ 5 -> 6 -> 7 -> 8
            for i in range(len(index_pts) - 1):
                cv2.line(image, index_pts[i], index_pts[i + 1], (0, 255, 130), 2, cv2.LINE_AA)
                cv2.circle(image, index_pts[i], 4, (0, 210, 255), -1, cv2.LINE_AA)

        if target_pt:
            # เป้าเล็งล็อกอยู่บนตัวเล็บนิ้วชี้พอดี (วงแหวนสีแดง + จุดกึ่งกลางสีเหลือง ไม่เลยนิ้ว)
            cv2.circle(image, target_pt, 10, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(image, target_pt, 3, (0, 255, 255), -1, cv2.LINE_AA)

        return image


class FingerBrailleTouchReader:
    """
    ประสานการตรวจจับนิ้วชี้ แมปเข้ากับพิกัดอักษรเบรลล์ และออกเสียงเมื่อแตะ
    รองรับทั้งการอ่านรวมคำ (Word Mode) และอ่านสะกดทีละตัว (Character Mode)
    """

    def __init__(self, external_tts=None, dwell_threshold=6, touch_radius=26, mode='word'):
        self.detector = IndexFingerSkeletonDetector()
        self.speaker = NonBlockingSpeechQueue(fallback_tts=external_tts)
        self.dwell_threshold = dwell_threshold  # จำนวนเฟรมที่ต้องแช่นิ้วค้างเพื่อยืนยัน (~0.2s ที่ 30 FPS)
        self.touch_radius = touch_radius        # รัศมีแตะกระชับพอดีเซลล์
        self.speech_mode = mode                 # 'word' (อ่านเป็นคำ), 'char' (อ่านสะกดทีละตัว), 'smart'

        self.cached_cells = []          # ข้อมูลตำแหน่ง Cell และตัวอักษรที่จำไว้
        self.grid_locked = False        # สถานะแช่แข็งพิกัด (Freeze Grid) เมื่อหน้ากระดาษสแกนสมบูรณ์แล้ว
        self.is_hand_present = False    # ติดตามว่ามีมือ/นิ้วอยู่ในกรอบกล้องหรือไม่
        self.current_candidate_idx = -1
        self.current_candidate_id = None
        self.current_word_id = None
        self.current_cell_idx = -1
        self.no_touch_counter = 0
        self.dwell_counter = 0
        self.cell_dwell_counter = 0
        self.cell_spelled = False
        self.miss_grace_counter = 0
        self.last_active_cell_info = None
        self.last_spoken_cell_idx = -1
        self.last_spoken_word_id = -1
        self.last_spoken_char = ""
        self.last_spoken_word = ""
        self.enabled = True

    def lock_grid(self):
        """แช่แข็งพิกัดตารางเซลล์อักษรเบรลล์ ไม่ให้ YOLO เขียนทับ พร้อม precache เสียงอ่านคำทั้งหมดในหน้า"""
        self.grid_locked = True
        if hasattr(self, 'speaker') and hasattr(self.speaker, 'precache'):
            words = {c.get('word') for c in self.cached_cells if c.get('word')}
            chars = {self.get_spoken_name(c.get('char')) for c in self.cached_cells if c.get('char')}
            self.speaker.precache(words | chars)

    def unlock_grid(self):
        """ปลดล็อกพิกัดตาราง เพื่อให้สแกนหน้าใหม่ได้"""
        self.grid_locked = False

    def toggle_mode(self):
        """สลับระหว่างโหมดอ่านเป็นคำ (Word) และโหมดสะกดทีละตัว (Char)"""
        modes = ['word', 'char', 'smart']
        curr_idx = modes.index(self.speech_mode) if self.speech_mode in modes else 0
        self.speech_mode = modes[(curr_idx + 1) % len(modes)]
        return self.speech_mode

    def update_grid_cache(self, cells, verbose_results, scale_x=1.0, scale_y=1.0, decoded_text="", force=False):
        """
        บันทึกพิกัดและตัวอักษรของแต่ละ Cell ลงใน Cache โดยล็อคขนาด Bounding Box ให้เท่ากับกรอบของตัวเบรลล์ (grid['bbox']) 100%
        จะไม่เขียนทับหาก Grid ถูกล็อกแล้ว
        """
        if not cells:
            return

        if not force:
            if self.grid_locked:
                return
            if self.is_hand_present:
                return
            if len(self.cached_cells) > 0 and len(cells) < len(self.cached_cells) * 0.6:
                return

        # ทำ Word Segmentation เพื่อแมปว่า Cell ไหนสังกัดคำว่าอะไร
        word_by_cell = {}
        word_id_by_cell = {}
        if decoded_text:
            try:
                from pythainlp.tokenize import word_tokenize
                clean_text = decoded_text.replace('\n', ' ')
                words = word_tokenize(clean_text)
                c_idx = 0
                for w_id, w in enumerate(words):
                    for _ in w:
                        word_by_cell[c_idx] = w
                        word_id_by_cell[c_idx] = w_id
                        c_idx += 1
            except Exception:
                pass

        updated = []
        for idx, cell in enumerate(cells):
            center = cell.get('center', (0, 0))
            if not center or center == (0, 0):
                continue

            token = verbose_results[idx] if idx < len(verbose_results) else {}
            char = token.get('char', '')
            dots = cell.get('dots', [])

            # ล็อค Bounding Box ให้เท่ากับกรอบของ Cell ที่ทำไว้กับตัวเบรลล์ (grid['bbox'])
            grid = cell.get('grid')
            if grid and grid.get('bbox'):
                gx1, gy1, gx2, gy2 = grid['bbox']
                x1 = int(round(gx1 * scale_x))
                y1 = int(round(gy1 * scale_y))
                x2 = int(round(gx2 * scale_x))
                y2 = int(round(gy2 * scale_y))
                cx = int(round((x1 + x2) / 2))
                cy = int(round((y1 + y2) / 2))
            elif cell.get('bbox'):
                bx1, by1, bx2, by2 = cell['bbox']
                x1 = int(round(bx1 * scale_x))
                y1 = int(round(by1 * scale_y))
                x2 = int(round(bx2 * scale_x))
                y2 = int(round(by2 * scale_y))
                cx = int(round((x1 + x2) / 2))
                cy = int(round((y1 + y2) / 2))
            else:
                cx = int(round(center[0] * scale_x))
                cy = int(round(center[1] * scale_y))
                width = int(round(cell.get('width', 36) * scale_x))
                height = int(round(cell.get('height', 52) * scale_y))
                x1 = cx - width // 2
                y1 = cy - height // 2
                x2 = cx + width // 2
                y2 = cy + height // 2

            assigned_word = word_by_cell.get(idx, char)
            assigned_word_id = word_id_by_cell.get(idx, idx)

            updated.append({
                'id': idx,
                'char': char,
                'word': assigned_word,
                'word_id': assigned_word_id,
                'dots': dots,
                'center': (cx, cy),
                'bbox': (x1, y1, x2, y2),
                'track_id': cell.get('track_id', idx + 1)
            })

        if updated:
            self.cached_cells = updated

    def find_cell_at_point(self, pt):
        """ค้นหาว่าจุด (x, y) ของปลายนิ้วตรงกับ Cell ไหน โดยล็อคเข้ากรอบเซลล์ตัวเบรลล์เป็นหลัก"""
        if pt is None or not self.cached_cells:
            return -1, None

        tx, ty = pt
        best_idx = -1
        min_dist = float('inf')
        best_in_box = False

        for idx, item in enumerate(self.cached_cells):
            x1, y1, x2, y2 = item['bbox']
            cx, cy = item['center']
            dist = math.hypot(tx - cx, ty - cy)

            # เช็คว่าอยู่ในกรอบเซลล์ของตัวเบรลล์จริง (เพิ่มระยะผ่อนปรนขอบ 4px)
            in_box = (x1 - 4 <= tx <= x2 + 4) and (y1 - 4 <= ty <= y2 + 4)

            if in_box:
                if not best_in_box or dist < min_dist:
                    min_dist = dist
                    best_idx = idx
                    best_in_box = True
            elif not best_in_box:
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx

        if best_idx != -1 and (best_in_box or min_dist <= self.touch_radius):
            return best_idx, self.cached_cells[best_idx]

        return -1, None

    def get_spoken_name(self, char):
        """แปลงอักขระเดี่ยวเป็นคำอ่านภาษาไทยสำหรับการแตะทีละตัว"""
        if not char or char == ' ':
            return ''
        if char in THAI_CONSONANT_NAMES:
            return THAI_CONSONANT_NAMES[char]
        if char in THAI_VOWEL_TONE_NAMES:
            return THAI_VOWEL_TONE_NAMES[char]
        if char.lower() in ENGLISH_NAMES:
            return ENGLISH_NAMES[char.lower()]
        return char

    def _check_reset_last_spoken(self):
        """เมื่อยกนิ้วออกเกิน 5 เฟรม ให้รีเซ็ตประวัติเพื่อให้แตะคำเดิมซ้ำได้"""
        self.no_touch_counter = getattr(self, 'no_touch_counter', 0) + 1
        if self.no_touch_counter > 5:
            self.last_spoken_word_id = -1
            self.last_spoken_cell_idx = -1
            self.cell_spelled = False
            self.cell_dwell_counter = 0

    def process(self, display_frame, clean_frame=None):
        """
        ประมวลผลต่อเฟรมแบบความเร็วสูง Zero-Latency:
        1. หาปลายนิ้วชี้จาก clean_frame (ภาพบริสุทธิ์ไม่มีเส้น UI มารบกวน MediaPipe)
        2. เช็คการแตะ Cell ด้วย Instant Response (ขยับเข้าคำใหม่พูดทันที)
        3. แสดงผล UI (Skeleton + Halo + Tooltip) บน display_frame
        4. สั่งออกเสียงทันทีผ่าน Background Voice Queue
        """
        if not self.enabled:
            return display_frame, None

        detect_source = clean_frame if clean_frame is not None else display_frame
        tip_pt, index_pts = self.detector.detect_index_finger(detect_source)
        self.is_hand_present = (tip_pt is not None)
        touched_cell = None

        # แสดงสถานะ Grid Lock / Scanning บนภาพมุมบนซ้าย (Non-intrusive)
        if self.grid_locked:
            badge_text = f"GRID: LOCKED ({len(self.cached_cells)} cells)"
            badge_color = (0, 255, 120)
        else:
            badge_text = f"GRID: SCANNING ({len(self.cached_cells)} cells)" if self.cached_cells else "GRID: SCANNING..."
            badge_color = (0, 210, 255)
        cv2.putText(display_frame, badge_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(display_frame, badge_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, badge_color, 1, cv2.LINE_AA)

        if tip_pt is not None:
            # วาดกระดูกนิ้วชี้และเป้าเล็งปลายเล็บ
            self.detector.draw_skeleton(display_frame, index_pts, tip_pt)
            cell_idx, cell_info = self.find_cell_at_point(tip_pt)

            # Miss Grace Period (ผ่อนผัน 3 เฟรม ป้องกัน MediaPipe Landmark สั่น/หลุด 1 เฟรม ทำให้ตัวนับรีเซ็ต)
            if cell_idx == -1 and getattr(self, 'miss_grace_counter', 0) < 3 and getattr(self, 'last_active_cell_info', None) is not None:
                self.miss_grace_counter += 1
                cell_info = self.last_active_cell_info
                cell_idx = cell_info['id']
            elif cell_idx != -1:
                self.miss_grace_counter = 0
                self.last_active_cell_info = cell_info

            if cell_idx != -1 and cell_info:
                self.no_touch_counter = 0
                touched_cell = cell_info
                x1, y1, x2, y2 = cell_info['bbox']
                char = cell_info['char']

                word = cell_info.get('word', char)
                word_id = cell_info.get('word_id', cell_idx)

                # Dwell Detection สำหรับคำ (Word Dwell)
                if word_id == getattr(self, 'current_word_id', None):
                    self.dwell_counter += 1
                else:
                    self.current_word_id = word_id
                    self.dwell_counter = 1

                # Dwell Detection สำหรับตัวอักษรเฉพาะตำแหน่ง (Cell Dwell สำหรับการทิ้งนิ้วค้างสะกด)
                if cell_idx == getattr(self, 'current_cell_idx', -1):
                    self.cell_dwell_counter += 1
                else:
                    self.current_cell_idx = cell_idx
                    self.cell_dwell_counter = 1
                    self.cell_spelled = False

                # คำนวณความคืบหน้าการแช่นิ้ว
                dwell_ratio = min(1.0, self.dwell_counter / max(1, self.dwell_threshold))
                is_confirmed_word = (self.dwell_counter >= self.dwell_threshold)
                is_spelling_dwell = (self.cell_dwell_counter >= 18)  # ทิ้งนิ้วค้างเกิน 18 เฟรม (~0.6s) -> สะกดรายตัว

                # สีกรอบและ Halo เปลี่ยนตามสถานะ
                if is_spelling_dwell:
                    border_color = (0, 240, 255)  # สีทองนีออน (สะกดตัวอักษร)
                elif is_confirmed_word:
                    border_color = (0, 255, 120)  # เขียวนีออน (อ่านคำแล้ว)
                else:
                    border_color = (0, 200, 255)  # เหลืองส้ม (กำลังชาร์จ)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), border_color, 3 if is_confirmed_word else 2)

                # วงกลมชาร์จรอบจุดกึ่งกลางเซลล์ ขยายตามความคืบหน้า
                ring_radius = int(14 + 8 * dwell_ratio)
                cv2.circle(display_frame, cell_info['center'], ring_radius, border_color, 2)

                spoken_char_name = self.get_spoken_name(char) or char

                # กำหนด Tooltip ตามโหมดและระยะเวลาที่แช่นิ้ว
                if self.speech_mode == 'char':
                    prefix = "สะกด" if self.cell_dwell_counter >= self.dwell_threshold else f"กำลังแตะ ({int(min(1.0, self.cell_dwell_counter / self.dwell_threshold)*100)}%)"
                    label = f"{prefix}: [{char}] {spoken_char_name}"
                elif is_spelling_dwell:
                    label = f"สะกด: [{char}] {spoken_char_name}"
                elif is_confirmed_word:
                    label = f"คำ: {word} [{char}] (แช่ค้างเพื่อสะกด)"
                else:
                    label = f"กำลังแตะ ({int(dwell_ratio*100)}%): {word} [{char}]"

                tooltip_x = max(10, min(display_frame.shape[1] - 240, tip_pt[0] - 50))
                tooltip_y = max(40, tip_pt[1] - 30)

                # พื้นหลังกล่องข้อความ Tooltip
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display_frame, (tooltip_x - 5, tooltip_y - th - 8),
                              (tooltip_x + tw + 5, tooltip_y + 6), (20, 20, 20), -1)
                cv2.rectangle(display_frame, (tooltip_x - 5, tooltip_y - th - 8),
                              (tooltip_x + tw + 5, tooltip_y + 6), border_color, 1)
                cv2.putText(display_frame, label, (tooltip_x, tooltip_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                # --- ตรรกะการออกเสียง ---
                if self.speech_mode == 'char':
                    # โหมดสะกดรายตัว: แช่ครบ dwell_threshold (~0.27s) พูดชื่อตัวอักษรทันที
                    if self.cell_dwell_counter >= self.dwell_threshold and cell_idx != self.last_spoken_cell_idx:
                        self.last_spoken_cell_idx = cell_idx
                        self.last_spoken_char = char
                        if spoken_char_name:
                            self.speaker.speak(spoken_char_name, interrupt=True)
                else:
                    # โหมด 'word' และ 'smart':
                    # 1. แช่นิ้วครบ dwell_threshold (~0.27s) -> อ่านเป็นคำเต็ม
                    if is_confirmed_word and word_id != getattr(self, 'last_spoken_word_id', -1):
                        self.last_spoken_word_id = word_id
                        self.last_spoken_cell_idx = cell_idx
                        if word and word.strip():
                            self.speaker.speak(word, interrupt=True)

                    # 2. ทิ้งนิ้วค้างไว้ที่ตัวอักษรเดิม (cell_dwell >= 18, ~0.6s) -> สะกดชื่อตัวอักษรนั้นออกมา
                    if is_spelling_dwell and not getattr(self, 'cell_spelled', False):
                        self.cell_spelled = True
                        self.last_spoken_cell_idx = cell_idx
                        if spoken_char_name and spoken_char_name.strip():
                            self.speaker.speak(spoken_char_name, interrupt=True)
            else:
                self.dwell_counter = 0
                self.cell_dwell_counter = 0
                self.cell_spelled = False
                self.current_word_id = None
                self.current_cell_idx = -1
                self.last_active_cell_info = None
                self.miss_grace_counter = 0
                self._check_reset_last_spoken()
        else:
            self.dwell_counter = 0
            self.cell_dwell_counter = 0
            self.cell_spelled = False
            self.current_word_id = None
            self.current_cell_idx = -1
            self.last_active_cell_info = None
            self.miss_grace_counter = 0
            self._check_reset_last_spoken()

        return display_frame, touched_cell

    def close(self):
        if self.speaker:
            self.speaker.stop()
