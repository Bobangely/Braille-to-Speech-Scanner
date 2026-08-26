"""
Braille Reader - Real-time Webcam Scanner
===========================================
ระบบสแกนและอ่านออกเสียงอักษรเบรลล์แบบสดผ่านกล้อง Webcam (Real-time OBR + TTS)
รองรับความละเอียดสูง 4K UHD และ Full HD (FHD) พร้อมระบบปรับระดับความคมชัด (Multi-level Sharpness) & Digital Zoom

ฟีเจอร์เด่น:
- สลับความละเอียดแบบสดได้ทันทีระหว่าง 4K UHD (3840x2160), Full HD 1080p (1920x1080) และ HD 720p (กดปุ่ม V หรือ F)
- ปรับระดับความคมชัดของภาพได้ 5 ระดับ (OFF -> LOW -> MED -> HIGH -> ULTRA) ผ่านปุ่ม E
- ใช้ MJPG FourCC Codec เพื่อปลดล็อก Bandwidth สูงสุดของกล้อง 4K/FHD USB
- ระบบ Digital Zoom In / Zoom Out (1.0x - 4.0x) สำหรับขยายอักษรเบรลล์ขนาดเล็ก
- สแกนเฟรมวิดีโอแบบสดพร้อม 2x3 Virtual Grid Overlay และแถบคำแปลภาษาไทย/อังกฤษ
- ระบบ Frame Stabilization ตรวจจับความนิ่งของคำก่อนตัดสินใจ
- ระบบ Auto-TTS Debounce ออกเสียงอัตโนมัติเมื่อข้อความนิ่ง ไม่บล็อก Video Stream
- สลับสีจุดแต้ม (Blue, Red, Green, Black) และสลับภาษา (ไทย/อังกฤษ) ได้ทันทีผ่านคีย์ลัด
- บันทึกภาพ Snapshot พร้อมคำแปลลงโฟลเดอร์ output/

คีย์ลัด (Hotkeys):
  [V] / [F]     : สลับความละเอียดกล้อง (4K UHD <-> Full HD 1080p <-> HD 720p)
  [E]           : ปรับระดับความคมชัด (Sharpness: OFF -> LOW -> MED -> HIGH -> ULTRA)
  [Z] / [+] / [=] : ซูมเข้า (Zoom In +0.2x)
  [X] / [-] / [_] : ซูมออก (Zoom Out -0.2x)
  [R] / [0]     : รีเซ็ตการซูม (Reset Zoom 1.0x)
  [SPACE] / [S] : ออกเสียงข้อความปัจจุบันทันที (Speak Now)
  [C]           : สลับสีจุดแต้ม (Blue -> Red -> Green -> Black)
  [L]           : สลับภาษา (Thai <-> English)
  [A]           : เปิด/ปิดระบบออกเสียงอัตโนมัติ (Toggle Auto-Speak)
  [P]           : ถ่ายภาพ Snapshot บันทึกลง output/
  [Q] / [ESC]   : ออกจากโปรแกรม

การใช้เมาส์ (Mouse Controls):
  • หมุนล้อเมาส์ขึ้น (Scroll Up)   : ซูมเข้า (Zoom In เล็งตรงตำแหน่งเมาส์)
  • หมุนล้อเมาส์ลง (Scroll Down) : ซูมออก (Zoom Out)
  • คลิกซ้ายบนภาพ                : เลื่อนจุดโฟกัส (Pan) ไปยังจุดที่คลิก
  • ดับเบิ้ลคลิก หรือ คลิกกลาง     : รีเซ็ตการซูมกลับ 1.0x
"""

import argparse
import os
import sys
import time
import threading
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw

# ปรับ encoding สำหรับ Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from detector import BrailleDetector
from decoder import decode_cells, decode_cells_verbose
from tts import speak, TextToSpeech
from config import DetectionConfig


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


class RealTimeBrailleScanner:
    """ตัวควบคุมการสแกนอักษรเบรลล์จากกล้องแบบ Real-time พร้อมระบบ 4K/FHD, Zoom & Multi-level Sharpness"""

    def __init__(
        self,
        camera_id=0,
        color='blue',
        lang='thai',
        auto_speak=True,
        stability_threshold=6,
        cooldown_seconds=3.0,
        res_preset='4k',
        width=None,
        height=None,
        initial_zoom=1.0,
        sharpness_level=2,
    ):
        self.camera_id = camera_id
        self.color = color.lower()
        self.lang = lang.lower()
        self.auto_speak = auto_speak
        self.stability_threshold = stability_threshold
        self.cooldown_seconds = cooldown_seconds

        # กำหนดระดับความคมชัดเริ่มต้น (0 ถึง 4)
        self.sharpness_idx = max(0, min(len(SHARPNESS_LEVELS) - 1, int(sharpness_level)))

        # กำหนดความละเอียดเริ่มต้น
        self.available_resolutions = list(RESOLUTION_LIST)
        self.curr_res_idx = 0

        if width and height:
            self.target_width = int(width)
            self.target_height = int(height)
            self.res_name = f"{self.target_width}x{self.target_height}"
        elif res_preset and res_preset.lower() in ('fhd', '1080p'):
            self.curr_res_idx = 1
            self.target_width, self.target_height = 1920, 1080
            self.res_name = "Full HD"
        elif res_preset and res_preset.lower() in ('hd', '720p'):
            self.curr_res_idx = 2
            self.target_width, self.target_height = 1280, 720
            self.res_name = "HD 720p"
        else:
            self.curr_res_idx = 0
            self.target_width, self.target_height = 3840, 2160
            self.res_name = "4K UHD"

        self.actual_width = self.target_width
        self.actual_height = self.target_height

        # Digital Zoom state
        self.zoom_level = max(1.0, min(4.0, float(initial_zoom)))
        self.zoom_center = [0.5, 0.5]  # [center_x, center_y] normalized (0.0 - 1.0)

        # State tracking
        self.history = deque(maxlen=stability_threshold)
        self.last_spoken_text = ""
        self.last_spoken_time = 0.0
        self.is_speaking = False
        self.supported_colors = ['blue', 'red', 'green', 'black']
        self.detector = BrailleDetector(dot_color=self.color)
        self.tts = TextToSpeech()

        # FPS calculation
        self.fps = 0.0
        self._prev_frame_time = time.time()

    def cycle_sharpness(self):
        """สลับระดับความคมชัด: OFF -> LOW -> MED -> HIGH -> ULTRA"""
        self.sharpness_idx = (self.sharpness_idx + 1) % len(SHARPNESS_LEVELS)
        lvl_num, lvl_name, _ = SHARPNESS_LEVELS[self.sharpness_idx]
        self.history.clear()
        print(f"  ✨ ปรับระดับความคมชัด (Sharpness): Level {lvl_num} [{lvl_name}]")

    def cycle_resolution(self, cap):
        """สลับความละเอียดแบบสดระหว่าง 4K UHD <-> Full HD 1080p <-> HD 720p"""
        self.curr_res_idx = (self.curr_res_idx + 1) % len(self.available_resolutions)
        name, target_w, target_h = self.available_resolutions[self.curr_res_idx]

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        time.sleep(0.05)

        self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.res_name = name
        self.history.clear()
        print(f"  📹 สลับความละเอียดกล้องเป็น: {name} ({self.actual_width}x{self.actual_height})")

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
        ตัด Crop ตามอัตราการซูมและตำแหน่ง zoom_center
        แล้วขยายกลับมาขนาดเดิมเพื่อให้ detector ประมวลผลจุดเล็กได้ชัดเจน
        """
        if self.zoom_level <= 1.001:
            return frame, None

        h, w = frame.shape[:2]
        crop_w = int(w / self.zoom_level)
        crop_h = int(h / self.zoom_level)

        cx = int(self.zoom_center[0] * w)
        cy = int(self.zoom_center[1] * h)

        # คำนวณขอบเขต Crop
        x1 = max(0, min(w - crop_w, cx - crop_w // 2))
        y1 = max(0, min(h - crop_h, cy - crop_h // 2))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        cropped = frame[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return zoomed, (x1, y1, x2, y2)

    def _apply_sharpening(self, frame):
        """เพิ่มความคมชัดของภาพตามระดับที่เลือก (Unsharp Masking Filter)"""
        _, _, strength = SHARPNESS_LEVELS[self.sharpness_idx]
        if strength <= 0.01:
            return frame

        # Unsharp masking: Image + strength * (Image - Blurred)
        gaussian = cv2.GaussianBlur(frame, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(frame, 1.0 + strength, gaussian, -strength, 0)
        return sharpened

    def _draw_mini_viewfinder(self, image, orig_frame, crop_box):
        """วาด Mini Viewfinder แสดงตำแหน่งพื้นที่ที่ถูกซูมบนภาพมุมกว้าง"""
        if crop_box is None or self.zoom_level <= 1.001:
            return

        h, w = image.shape[:2]
        x1, y1, x2, y2 = crop_box

        # ขนาดกล่อง Viewfinder มุมขวาบน
        vw, vh = 160, int(160 * (h / w))
        vx = w - vw - 15
        vy = 50

        # ย่อภาพเต็มต้นฉบับ
        mini = cv2.resize(orig_frame, (vw, vh))

        # คำนวณกรอบสี่เหลี่ยมของ ROI ใน mini map
        scale_x = vw / w
        scale_y = vh / h
        rx1 = int(x1 * scale_x)
        ry1 = int(y1 * scale_y)
        rx2 = int(x2 * scale_x)
        ry2 = int(y2 * scale_y)

        # วาดกรอบสี่เหลี่ยมแสดงพื้นที่ซูม
        cv2.rectangle(mini, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
        cv2.rectangle(mini, (0, 0), (vw - 1, vh - 1), (120, 120, 120), 1)

        # ป้ายกำกับ
        cv2.putText(
            mini, f"ZOOM {self.zoom_level:.1f}x", (5, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA
        )

        # แปะลงบนเฟรม
        image[vy:vy + vh, vx:vx + vw] = mini

    def _speak_async(self, text, lang):
        """ออกเสียงใน Background Thread เพื่อไม่ให้ภาพกระตุก"""
        if self.is_speaking or not text.strip():
            return

        def _worker():
            self.is_speaking = True
            try:
                self.tts.speak(text, lang=lang)
            finally:
                self.is_speaking = False

        threading.Thread(target=_worker, daemon=True).start()

    def _draw_top_hud(self, image, current_text, is_locked):
        """วาดแถบเมนูควบคุมและสถานะด้านบน (Top HUD)"""
        h, w = image.shape[:2]
        hud_h = 42

        # พื้นหลังแถบ HUD ด้านบน (Semi-transparent dark bar)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
        cv2.line(image, (0, hud_h), (w, hud_h), (60, 80, 100), 1)

        # ข้อมูลสถานะ
        color_badges = {
            'blue': 'BLUE [C]',
            'red': 'RED [C]',
            'green': 'GREEN [C]',
            'black': 'BLACK [C]',
        }
        color_text = color_badges.get(self.color, self.color.upper())
        lang_text = "THAI [L]" if self.lang == 'thai' else "ENG [L]"
        auto_text = "AUTO [A]" if self.auto_speak else "MANUAL [A]"

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

        fps_text = f"FPS: {self.fps:.1f}"

        # Zoom badge
        if self.zoom_level > 1.001:
            zoom_text = f"ZOOM: {self.zoom_level:.1f}x [Z/X/R]"
            zoom_color = (0, 255, 255)  # Yellow
        else:
            zoom_text = "ZOOM: 1.0x [Z/X]"
            zoom_color = (200, 200, 200)

        # Status badge
        if is_locked and current_text:
            status_text = f"LOCKED: {current_text}"
            status_color = (0, 255, 120)  # Bright Green
        elif current_text:
            status_text = "DETECTING..."
            status_color = (0, 200, 255)  # Orange/Yellow
        else:
            status_text = "SCANNING..."
            status_color = (180, 180, 180)  # Gray

        # วาดข้อความ HUD
        cv2.putText(image, fps_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image, f"| {res_badge}", (90, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, res_color, 1, cv2.LINE_AA)
        cv2.putText(image, f"| {sharp_text}", (330, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, sharp_color, 1, cv2.LINE_AA)
        cv2.putText(image, f"| {zoom_text}", (530, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, zoom_color, 1 if self.zoom_level <= 1.001 else 2, cv2.LINE_AA)
        cv2.putText(image, f"| {color_text}", (705, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 80), 1, cv2.LINE_AA)
        cv2.putText(image, f"| {lang_text}", (830, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"| {status_text}", (940, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 2, cv2.LINE_AA)

        # วาดคำแนะนำปุ่มกดด้านล่างขวา
        tip = "[V/F] Res | [E] Sharp | [Z/X] Zoom | [SPACE] Speak | [P] Save | [Q] Quit"
        cv2.putText(image, tip, (w - 490, hud_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    def _on_mouse(self, event, x, y, flags, param):
        """Event handler สำหรับการควบคุม Zoom และ Pan ด้วยเมาส์"""
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

    def run(self):
        """เริ่มการทำงานกล้องและลูปประมวลผล"""
        lvl_num, lvl_name, _ = SHARPNESS_LEVELS[self.sharpness_idx]
        print("=" * 70)
        print("   Braille-to-Speech Real-Time Scanner (4K UHD & Full HD Ready)")
        print("=" * 70)
        print(f"  เปิดกล้อง ID:        Camera Index {self.camera_id}")
        print(f"  ความละเอียดเป้าหมาย:  {self.target_width}x{self.target_height} ({self.res_name})")
        print(f"  ระดับความคมชัด:      Level {lvl_num} [{lvl_name}]")
        print(f"  สีจุดแต้มเริ่มต้น:    {self.color.upper()}")
        print(f"  ภาษาเริ่มต้น:        {self.lang.upper()}")
        print(f"  อัตราการซูมเริ่มต้น:  {self.zoom_level:.1f}x")
        print(f"  ระบบ Auto-Speak:    {'เปิดใช้งาน' if self.auto_speak else 'ปิดใช้งาน'}")
        print()
        print("  คีย์ลัดและควบคุม:")
        print("    [V] / [F]       - สลับความละเอียด (4K UHD <-> Full HD 1080p <-> HD 720p)")
        print("    [E]             - ปรับระดับความคมชัด (OFF -> LOW -> MED -> HIGH -> ULTRA)")
        print("    [Z] / [+]       - ซูมเข้า (Zoom In)")
        print("    [X] / [-]       - ซูมออก (Zoom Out)")
        print("    [R] / [0]       - รีเซ็ตการซูม (Reset Zoom 1.0x)")
        print("    [Wheel Up/Down] - ซูมเข้า/ออกด้วยล้อเมาส์")
        print("    [SPACE]         - สั่งอ่านออกเสียงข้อความปัจจุบัน")
        print("    [C]             - สลับสีจุด (Blue -> Red -> Green -> Black)")
        print("    [L]             - สลับภาษา (Thai <-> English)")
        print("    [A]             - เปิด/ปิด Auto-TTS")
        print("    [P]             - ถ่ายภาพ Snapshot ลง output/")
        print("    [Q]/[ESC]       - ออกจากโปรแกรม")
        print("=" * 70)
        print()

        # เปิดกล้องด้วย DirectShow (บน Windows) เพื่อให้รองรับ FourCC และความละเอียดสูง 4K
        backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.camera_id, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print(f"[ERR] ไม่สามารถเปิดกล้อง Webcam ID: {self.camera_id} ได้")
            print("  ลองตรวจสอบการเชื่อมต่อกล้อง หรือเปลี่ยน index เช่น --camera 1")
            return

        # 1. ตั้งค่า FourCC เป็น MJPG เพื่อปลดล็อก Bandwidth ความเร็วสูงสำหรับ 4K / 1080p
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # 2. ตั้งค่าความละเอียดที่ต้องการ
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        # 3. ตรวจสอบความละเอียดจริงที่กล้องเปิดได้
        self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  📷 กล้องเปิดสำเร็จ! ความละเอียดจริงที่ใช้งาน: {self.actual_width}x{self.actual_height}")
        if self.actual_width >= 3840:
            print("  🌟 ทำงานในโหมด 4K Ultra HD คมชัดระดับสูงสุด!")
        elif self.actual_width >= 1920:
            print("  ✨ ทำงานในโหมด Full HD 1080p คมชัดสูง!")

        window_name = "Braille Real-Time Scanner [4K / FHD / Zoom / Sharpness]"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # ปรับขนาดหน้าต่างแสดงผลเริ่มต้นให้พอดีจอ
        disp_w = min(1600, self.actual_width)
        disp_h = int(disp_w * (self.actual_height / max(1, self.actual_width)))
        cv2.resizeWindow(window_name, disp_w, disp_h)
        cv2.setMouseCallback(window_name, self._on_mouse)

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARN] ไม่สามารถอ่านเฟรมจากกล้องได้")
                    time.sleep(0.05)
                    continue

                # อัปเดตขนาดจริงของเฟรม
                self.actual_height, self.actual_width = frame.shape[:2]

                # คำนวณ FPS
                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self._prev_frame_time, 1e-5))
                self._prev_frame_time = now

                # 1. ใช้งาน Sharpening Filter ตามระดับที่เลือก
                enhanced_frame = self._apply_sharpening(frame)

                # 2. ใช้งาน Digital Zoom (ถ้า zoom_level > 1.0)
                zoomed_frame, crop_box = self._apply_zoom(enhanced_frame)

                # 3. ตรวจจับอักษรเบรลล์จากภาพความละเอียดสูง
                cells, debug_info = self.detector.detect(zoomed_frame)
                dots = debug_info['dots']

                # 4. ถอดรหัสข้อความ
                decoded_text = ""
                verbose_results = []
                if cells:
                    decoded_text = decode_cells(cells, lang=self.lang)
                    verbose_results = decode_cells_verbose(cells, lang=self.lang)

                # 5. ตรวจสอบความนิ่งของคำ (Stability Buffer)
                self.history.append(decoded_text)
                is_locked = False

                if len(self.history) == self.stability_threshold:
                    if all(t == decoded_text for t in self.history) and decoded_text.strip():
                        is_locked = True

                        # Auto-TTS Trigger
                        time_since_last_speak = now - self.last_spoken_time
                        is_new_text = (decoded_text != self.last_spoken_text)
                        is_cooldown_expired = (time_since_last_speak >= self.cooldown_seconds)

                        if self.auto_speak and (is_new_text or is_cooldown_expired):
                            self.last_spoken_text = decoded_text
                            self.last_spoken_time = now
                            print(f"  🔊 [AUTO-TTS] \"{decoded_text}\" (lang={self.lang})")
                            self._speak_async(decoded_text, self.lang)

                # 6. วาด 2x3 Grid Overlay และแบนเนอร์แสดงผลลัพธ์
                annotated = self.detector.annotate_with_text(
                    zoomed_frame, dots, cells,
                    decoded_text=decoded_text,
                    verbose_results=verbose_results,
                    lang=self.lang
                )

                # 7. วาด Mini Viewfinder มุมขวาบน (กรณีซูมอยู่)
                self._draw_mini_viewfinder(annotated, frame, crop_box)

                # 8. วาด Top HUD Bar
                self._draw_top_hud(annotated, decoded_text, is_locked)

                # 9. แสดงผลลัพธ์บนหน้าต่าง
                cv2.imshow(window_name, annotated)

                # 10. จัดการคีย์บอร์ด
                key = cv2.waitKey(1) & 0xFF

                # [Q] หรือ [ESC] -> ออก
                if key in (ord('q'), ord('Q'), 27):
                    print("  ปิดโปรแกรม...")
                    break

                # [V] หรือ [F] -> สลับความละเอียด (4K <-> FHD <-> HD)
                elif key in (ord('v'), ord('V'), ord('f'), ord('F')):
                    self.cycle_resolution(cap)

                # [E] -> ปรับระดับความคมชัด (Multi-level Sharpening)
                elif key in (ord('e'), ord('E')):
                    self.cycle_sharpness()

                # [Z] / [+] / [=] / [I] -> Zoom In
                elif key in (ord('z'), ord('Z'), ord('+'), ord('='), ord('i'), ord('I')):
                    self.zoom_in()

                # [X] / [-] / [_] / [O] -> Zoom Out
                elif key in (ord('x'), ord('X'), ord('-'), ord('_'), ord('o'), ord('O')):
                    self.zoom_out()

                # [R] / [0] -> Reset Zoom
                elif key in (ord('r'), ord('R'), ord('0')):
                    self.reset_zoom()

                # [SPACE] หรือ [S] -> ออกเสียงทันที
                elif key in (ord(' '), ord('s'), ord('S')):
                    if decoded_text.strip():
                        print(f"  🔊 [MANUAL-TTS] \"{decoded_text}\"")
                        self.last_spoken_text = decoded_text
                        self.last_spoken_time = time.time()
                        self._speak_async(decoded_text, self.lang)

                # [C] -> สลับสีจุดแต้ม
                elif key in (ord('c'), ord('C')):
                    curr_idx = self.supported_colors.index(self.color)
                    self.color = self.supported_colors[(curr_idx + 1) % len(self.supported_colors)]
                    self.detector = BrailleDetector(dot_color=self.color)
                    self.history.clear()
                    print(f"  🎨 สลับสีจุดเป็น: {self.color.upper()}")

                # [L] -> สลับภาษา
                elif key in (ord('l'), ord('L')):
                    self.lang = 'english' if self.lang == 'thai' else 'thai'
                    self.history.clear()
                    print(f"  🌐 สลับภาษาเป็น: {self.lang.upper()}")

                # [A] -> เปิด/ปิด Auto-TTS
                elif key in (ord('a'), ord('A')):
                    self.auto_speak = not self.auto_speak
                    print(f"  🔊 Auto-TTS: {'เปิด' if self.auto_speak else 'ปิด'}")

                # [P] -> บันทึก Snapshot ความละเอียดสูง
                elif key in (ord('p'), ord('P')):
                    os.makedirs('output', exist_ok=True)
                    timestamp = int(time.time())
                    snap_path = f"output/snapshot_{timestamp}_{self.actual_width}x{self.actual_height}.png"
                    cv2.imwrite(snap_path, annotated)
                    print(f"  📸 บันทึก Snapshot ความละเอียดสูง: {snap_path}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("  กล้องปิดการทำงานเรียบร้อย")


def main():
    parser = argparse.ArgumentParser(
        description='Braille Reader — ระบบสแกนอักษรเบรลล์ Real-Time รองรับ 4K UHD / Full HD พร้อมปรับระดับความคมชัด & Digital Zoom',
    )
    parser.add_argument(
        '--camera', type=int, default=0,
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
        '--color', type=str, default='blue',
        choices=['blue', 'red', 'green', 'black'],
        help='สีของจุดที่แต้ม (default: blue)',
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
        '--no-auto-speak', action='store_true',
        help='ปิดระบบออกเสียงอัตโนมัติ (ใช้กด SPACE เพื่อออกเสียงแทน)',
    )
    parser.add_argument(
        '--stability', type=int, default=6,
        help='จำนวนเฟรมที่ข้อความต้องนิ่งก่อนออกเสียงอัตโนมัติ (default: 6)',
    )
    parser.add_argument(
        '--cooldown', type=float, default=3.0,
        help='ระยะเวลาหน่วงก่อนอ่านคำซ้ำ (วินาที, default: 3.0)',
    )
    parser.add_argument(
        '--width', type=int, default=None,
        help='ความกว้างวิดีโอแบบระบุเจาะจง (px)',
    )
    parser.add_argument(
        '--height', type=int, default=None,
        help='ความสูงวิดีโอแบบระบุเจาะจง (px)',
    )

    args = parser.parse_args()

    scanner = RealTimeBrailleScanner(
        camera_id=args.camera,
        color=args.color,
        lang=args.lang,
        auto_speak=not args.no_auto_speak,
        stability_threshold=args.stability,
        cooldown_seconds=args.cooldown,
        res_preset=args.res,
        width=args.width,
        height=args.height,
        initial_zoom=args.zoom,
        sharpness_level=args.sharp,
    )
    scanner.run()


if __name__ == '__main__':
    main()
