"""
YOLO & Hybrid CV+YOLO Braille Dot Detector
===========================================
ระบบตรวจจับจุดเบรลล์แบบผสมผสาน (Hybrid Computer Vision + YOLOv8 Deep Learning)
เพิ่มความแม่นยำสูงสุด ทนต่อสภาพแสง เงา มุมมอง และพื้นหลัง พร้อมระบบสลับโหมดแบบ Real-time

สถาปัตยกรรม True Hybrid Fusion (v2):
    1. [OpenCV]: ตรวจจับจุดที่ Native Resolution (ไม่ย่อภาพ) → จับจุดเล็กในประโยคยาวได้ครบ
    2. [YOLOv8]: ตรวจจับจุดด้วย Deep Learning (imgsz=1280) → ทนทานต่อแสง เงา มุมมอง
    3. [Fusion]: รวมผลลัพธ์ทั้งสอง (Union) + ลบจุดซ้ำ (Deduplication by distance)
    4. [Sub-pixel Refinement]: ปรับพิกัดจุดด้วย Image Moments สำหรับจุดจาก YOLO
    5. [Grid Clustering]: จัดกลุ่มจุด 2x3 Braille Cell ด้วยตรรกะระยะห่างทางเรขาคณิต
    6. [Decoder]: ถอดรหัสภาษาไทย (สระ/พยัญชนะ/วรรณยุกต์) และภาษาอังกฤษแบบสมบูรณ์

โหมดการทำงาน:
    - 'hybrid' : แนะนำสำหรับการใช้งานจริง (OpenCV + YOLO Parallel Fusion)
    - 'yolo'   : ใช้ YOLO ตรวจจับจุดล้วนๆ
    - 'opencv' : ใช้ OpenCV ดั้งเดิม (Color Mask + Morphology)
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# เพิ่ม path ของโปรเจกต์
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import BrailleDetector
from config import DetectionConfig


class YOLOBrailleDetector:
    """
    ตัวตรวจจับอักษรเบรลล์รองรับ 3 โหมด: HYBRID (CV+YOLO), YOLO ONLY, และ OPENCV ONLY
    พร้อมฟังก์ชันสำหรับใช้งานในกล้อง Real-time (camera_reader.py)
    """

    AVAILABLE_MODES = ['hybrid', 'yolo', 'opencv']

    def __init__(self, model_path=None, confidence=0.35, mode='hybrid', fallback_color='blue'):
        """
        Parameters
        ----------
        model_path : str
            Path ไปยังไฟล์ YOLO model (.pt)
        confidence : float
            Confidence threshold (default: 0.35)
        mode : str
            โหมดการตรวจจับ: 'hybrid', 'yolo', 'opencv' (default: 'hybrid')
        fallback_color : str
            สีที่ใช้เมื่อ fallback ไป OpenCV
        """
        self.confidence = float(confidence)
        self.mode = mode.lower() if mode.lower() in self.AVAILABLE_MODES else 'hybrid'
        self.color = fallback_color.lower()
        self.model = None
        self.model_path = model_path
        self._font_cache = {}

        # YOLO Inference Resolution — เพิ่มจาก 416 เป็น 1280
        # เพื่อให้ YOLO มองเห็นจุดเบรลล์ขนาดเล็กได้ดีขึ้น (ไม่ย่อภาพจนจุดหายไป)
        self.yolo_imgsz = 1280

        # โหลดโมเดล YOLO
        self._load_model(model_path)

        # ตัวตรวจจับ OpenCV ดั้งเดิม (สำหรับ Grid Clustering และโหมด OpenCV)
        self._opencv_detector = BrailleDetector(dot_color=self.color)

    def _load_model(self, model_path):
        """ค้นหาและโหลดไฟล์ YOLO weights"""
        search_paths = [
            model_path,
            'models/braille_yolo.pt',
            'runs/detect/train/weights/best.pt',
            'runs/detect/runs/detect/train/weights/best.pt',
            'braille_yolo.pt',
        ]

        for path in search_paths:
            if path and os.path.exists(path):
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(path)
                    self.model_path = path
                    print(f"  🧠 [YOLO] โหลดโมเดลสำเร็จ: {path}")
                    return
                except Exception as e:
                    print(f"  ⚠️ [YOLO] โหลดโมเดลล้มเหลว ({path}): {e}")

        print("  ℹ️ [YOLO] ไม่พบโมเดล YOLO — จะ fallback ไปใช้ OpenCV อัตโนมัติ")
        self.mode = 'opencv'

    def set_mode(self, mode):
        """ตั้งค่าโหมดการทำงาน ('hybrid', 'yolo', 'opencv')"""
        mode_str = mode.lower()
        if mode_str in self.AVAILABLE_MODES:
            if mode_str in ('hybrid', 'yolo') and self.model is None:
                print("  ⚠️ ไม่พบ YOLO model จึงทำงานในโหมด OPENCV")
                self.mode = 'opencv'
            else:
                self.mode = mode_str
                print(f"  🔄 สลับโหมด Detector เป็น: {self.mode.upper()}")
        return self.mode

    def cycle_mode(self):
        """สลับโหมดถัดไป: HYBRID -> YOLO -> OPENCV -> HYBRID"""
        curr_idx = self.AVAILABLE_MODES.index(self.mode)
        next_idx = (curr_idx + 1) % len(self.AVAILABLE_MODES)
        new_mode = self.AVAILABLE_MODES[next_idx]
        return self.set_mode(new_mode)

    def set_color(self, color):
        """เปลี่ยนสีจุดสำหรับ OpenCV fallback"""
        self.color = color.lower()
        self._opencv_detector = BrailleDetector(dot_color=self.color)
        print(f"  🎨 [Detector] อัปเดตสีจุดเป็น: {self.color.upper()}")

    def is_yolo_ready(self):
        """ตรวจสอบว่า YOLO พร้อมใช้งานหรือไม่"""
        return self.model is not None

    def detect(self, image):
        """
        ตรวจจับจุดเบรลล์ตามโหมดปัจจุบัน
        Returns:
            cells : list of dict (dots, center, x, y, grid)
            debug_info : dict (dots, mask, annotated, method)
        """
        if self.mode == 'opencv' or self.model is None:
            cells, debug_info = self._opencv_detector.detect(image)
            debug_info['method'] = 'opencv'
            return cells, debug_info

        if self.mode == 'yolo':
            return self._detect_yolo_only(image)

        # โหมด Hybrid (CV + YOLO) — True Parallel Fusion
        return self._detect_hybrid(image)

    def _detect_yolo_only(self, image):
        """ตรวจจับด้วย YOLO ล้วนๆ (imgsz=1280 สำหรับจุดเล็ก)"""
        results = self.model(image, verbose=False, conf=self.confidence, imgsz=self.yolo_imgsz)
        dots = self._extract_yolo_boxes(results, image.shape)

        cells = []
        if len(dots) >= 1:
            cells = self._opencv_detector._cluster_into_cells(dots)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for d in dots:
            cx, cy = d['center']
            r = int(np.sqrt(d['area'] / np.pi))
            cv2.circle(mask, (int(cx), int(cy)), max(r, 4), 255, -1)

        debug_info = {
            'mask': mask,
            'dots': dots,
            'method': 'yolo',
            'num_detections': len(dots),
        }
        return cells, debug_info

    def _detect_hybrid(self, image):
        """
        ตรวจจับแบบ Hybrid:
        1. YOLO หา Candidate bounding boxes
        2. OpenCV ตรวจสอบความกลม (circularity) และปรับ centroid ด้วย Sub-pixel moments
        3. Fallback ไป OpenCV Color ถ้า YOLO ไม่พบจุด
        4. จัดกลุ่มเซลล์ 2x3
        """
        results = self.model(image, verbose=False, conf=self.confidence, imgsz=self.yolo_imgsz)
        raw_dots = self._extract_yolo_boxes(results, image.shape)

        # ถ้า YOLO ไม่พบจุดเลย ให้ fallback ไปใช้ OpenCV color detector
        if len(raw_dots) == 0:
            cells, cv_debug = self._opencv_detector.detect(image)
            cv_debug['method'] = 'hybrid_cv_fallback'
            return cells, cv_debug

        # ขั้นตอน Refinement ด้วย OpenCV
        refined_dots = self._refine_dots_with_opencv(image, raw_dots)

        cells = []
        if len(refined_dots) >= 1:
            cells = self._opencv_detector._cluster_into_cells(refined_dots)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for d in refined_dots:
            cx, cy = d['center']
            r = int(np.sqrt(d['area'] / np.pi))
            cv2.circle(mask, (int(cx), int(cy)), max(r, 4), 255, -1)

        debug_info = {
            'mask': mask,
            'dots': refined_dots,
            'method': 'hybrid',
            'num_detections': len(refined_dots),
        }
        return cells, debug_info

    def _extract_yolo_boxes(self, results, img_shape):
        """แปลงผลลัพธ์จาก Ultralytics YOLO เป็น dot dictionary"""
        dots = []
        if len(results) == 0 or results[0].boxes is None:
            return dots

        boxes = results[0].boxes
        h_img, w_img = img_shape[:2]

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            area = w * h

            dots.append({
                'center': (cx, cy),
                'area': area,
                'circularity': 1.0,
                'confidence': conf,
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'refined': False,
            })
        return dots

    def _refine_dots_with_opencv(self, image, yolo_dots):
        """
        ใช้ OpenCV วิเคราะห์ ROI ของแต่ละ bounding box
        เพื่อคำนวณ Centroid ที่แม่นยำ และกรอง False Positives
        """
        h_img, w_img = image.shape[:2]
        refined_dots = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for dot in yolo_dots:
            x1, y1, x2, y2 = dot['bbox']
            bw = x2 - x1
            bh = y2 - y1

            # เพิ่ม Margin เล็กน้อยรอบจุด
            pad_x = max(2, int(bw * 0.2))
            pad_y = max(2, int(bh * 0.2))

            rx1 = max(0, x1 - pad_x)
            ry1 = max(0, y1 - pad_y)
            rx2 = min(w_img, x2 + pad_x)
            ry2 = min(h_img, y2 + pad_y)

            roi_gray = gray[ry1:ry2, rx1:rx2]
            if roi_gray.size == 0:
                refined_dots.append(dot)
                continue

            # ตรวจสอบ contrast ใน ROI
            min_val, max_val, _, _ = cv2.minMaxLoc(roi_gray)
            if max_val - min_val < 15:
                # Contrast ต่ำ ใช้พิกัดเดิมของ YOLO
                refined_dots.append(dot)
                continue

            # ใช้ Otsu thresholding บน patch
            _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # ค้นหา contours ใน patch
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_cnt = None
            best_dist = float('inf')
            patch_cx = (x2 + x1) / 2.0 - rx1
            patch_cy = (y2 + y1) / 2.0 - ry1

            for cnt in contours:
                c_area = cv2.contourArea(cnt)
                if c_area < 5:
                    continue
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    mcx = M['m10'] / M['m00']
                    mcy = M['m01'] / M['m00']
                    dist = (mcx - patch_cx) ** 2 + (mcy - patch_cy) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_cnt = cnt

            if best_cnt is not None:
                M = cv2.moments(best_cnt)
                if M['m00'] > 0:
                    mcx = rx1 + (M['m10'] / M['m00'])
                    mcy = ry1 + (M['m01'] / M['m00'])
                    area = cv2.contourArea(best_cnt)
                    perimeter = cv2.arcLength(best_cnt, True)
                    circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.5

                    refined_dots.append({
                        'center': (mcx, mcy),
                        'area': max(area, dot['area']),
                        'circularity': float(circ),
                        'confidence': dot['confidence'],
                        'bbox': dot['bbox'],
                        'refined': True,
                    })
                    continue

            # Fallback หาก contour ไม่ชัดเจน
            refined_dots.append(dot)

        return refined_dots

    def _get_font(self, size=18, bold=False):
        """โหลด Font สำหรับแสดงภาษาไทยและอังกฤษ"""
        cache_key = (size, bold)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font_candidates = [
            "C:/Windows/Fonts/tahomabd.ttf" if bold else "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/leelawbd.ttf" if bold else "C:/Windows/Fonts/leelawadee.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        ]

        font = None
        for p in font_candidates:
            if os.path.exists(p):
                try:
                    font = ImageFont.truetype(p, size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

        self._font_cache[cache_key] = font
        return font

    def annotate_with_text(self, image, dots, cells, decoded_text="", verbose_results=None, lang="thai"):
        """
        วาด Overlays ครบถ้วน:
        1. Bounding Boxes / Refined Contours
        2. 2x3 Grid Overlay ของแต่ละ Cell
        3. แถบสรุปผลลัพธ์ (Bottom Banner) พร้อม Badge โหมดการตรวจจับ
        """
        h, w = image.shape[:2]
        banner_h = 80
        canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
        canvas[:h, :w] = image.copy()
        canvas[h:, :] = (28, 22, 16)  # Dark navy slate

        # 1. วาด 2x3 Grid รอบแต่ละ Cell
        for idx, cell in enumerate(cells, 1):
            grid = cell.get('grid')
            if not grid:
                continue

            x_min, y_min, x_max, y_max = grid['bbox']
            cols = grid['expected_cols']
            rows = grid['expected_rows']
            col_mid = int((cols[0] + cols[1]) / 2.0)
            row_mid1 = int((rows[0] + rows[1]) / 2.0)
            row_mid2 = int((rows[1] + rows[2]) / 2.0)

            # กรอบ Cell (สีส้มอมทอง)
            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (255, 200, 0), 2)
            cv2.line(canvas, (col_mid, y_min), (col_mid, y_max), (200, 160, 0), 1)
            cv2.line(canvas, (x_min, row_mid1), (x_max, row_mid1), (200, 160, 0), 1)
            cv2.line(canvas, (x_min, row_mid2), (x_max, row_mid2), (200, 160, 0), 1)

            # วาดสัญลักษณ์ช่องว่าง (Empty slot circles)
            for dot_id, (sx, sy) in grid['slots'].items():
                if dot_id not in cell['dots']:
                    cv2.circle(canvas, (int(sx), int(sy)), 4, (140, 140, 140), 1)

            # ป้ายบอกหมายเลขจุด
            dots_str = ','.join(map(str, sorted(cell['dots'])))
            cv2.putText(
                canvas, f"[{dots_str}]",
                (int(cell['center'][0]) - 16, y_max + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 100, 255), 1, cv2.LINE_AA
            )

        # 2. วาดไฮไลท์รอบจุดที่ตรวจพบ
        for dot in dots:
            cx, cy = dot['center']
            conf = dot.get('confidence')
            bbox = dot.get('bbox')
            is_refined = dot.get('refined', False)
            radius = int(np.sqrt(dot['area'] / np.pi))

            # วงกลมไฮไลท์
            if is_refined:
                # สีเขียวมรกตสำหรับ Hybrid Refined Dot
                cv2.circle(canvas, (int(cx), int(cy)), radius + 3, (0, 255, 120), 2)
                cv2.circle(canvas, (int(cx), int(cy)), 2, (0, 0, 255), -1)
            else:
                cv2.circle(canvas, (int(cx), int(cy)), radius + 3, (0, 255, 0), 2)
                cv2.circle(canvas, (int(cx), int(cy)), 2, (0, 0, 255), -1)

            # Bounding Box
            if bbox:
                bx1, by1, bx2, by2 = bbox
                box_color = (255, 180, 0) if not is_refined else (220, 255, 50)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), box_color, 1)

                if conf is not None:
                    cv2.putText(
                        canvas, f"{conf:.0%}",
                        (bx1, max(12, by1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA
                    )

        # 3. วาดข้อความผลลัพธ์และตัวอักษรด้วย PIL
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        font_large = self._get_font(size=24, bold=True)
        font_mid = self._get_font(size=18, bold=True)
        font_small = self._get_font(size=14, bold=False)

        # ตัวอักษรเหนือ Cell
        if verbose_results:
            for idx, item in enumerate(verbose_results, 1):
                if idx - 1 < len(cells):
                    cell = cells[idx - 1]
                    grid = cell.get('grid')
                    if grid:
                        x_min, y_min = grid['bbox'][0], grid['bbox'][1]
                        char_text = f"C{idx}: {item['char']}"
                        draw.text((x_min + 2, y_min - 25), char_text, fill=(255, 190, 0), font=font_mid)
        else:
            for idx, cell in enumerate(cells, 1):
                grid = cell.get('grid')
                if grid:
                    x_min, y_min = grid['bbox'][0], grid['bbox'][1]
                    draw.text((x_min + 2, y_min - 22), f"C{idx}", fill=(255, 190, 0), font=font_small)

        # 4. Bottom Banner
        draw.line([(0, h), (w, h)], fill=(70, 85, 105), width=2)

        # Mode Badge
        if self.mode == 'hybrid':
            mode_badge = "⚡ MODE: HYBRID (CV + YOLO)"
            mode_color = (0, 255, 150)
        elif self.mode == 'yolo':
            mode_badge = "🧠 MODE: YOLO ONLY"
            mode_color = (0, 210, 255)
        else:
            mode_badge = "🔬 MODE: OPENCV ONLY"
            mode_color = (255, 150, 255)

        if decoded_text:
            text_disp = f"ข้อความ: \"{decoded_text}\"" if lang == 'thai' else f"Text: \"{decoded_text}\""
            braille_chars = [item.get('unicode', '·') for item in (verbose_results or [])]
            braille_disp = ' '.join(braille_chars)

            draw.text((20, h + 12), text_disp, fill=(255, 255, 100), font=font_large)

            lang_label = "ไทย (Thai)" if lang == 'thai' else "English"
            sub_text = f"{mode_badge}  |  Braille: {braille_disp}  |  {len(cells)} Cells ({len(dots)} Dots)  |  {lang_label}"
            draw.text((20, h + 48), sub_text, fill=(180, 210, 230), font=font_small)
        else:
            draw.text((20, h + 22), f"{mode_badge}  -  พร้อมสแกน...", fill=mode_color, font=font_mid)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# =============================================================================
# CLI Testing Interface
# =============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLO & Hybrid Braille Dot Detector')
    parser.add_argument('image', type=str, help='Path ของภาพ')
    parser.add_argument('--model', type=str, default=None, help='Path ของ YOLO model (.pt)')
    parser.add_argument('--mode', type=str, default='hybrid', choices=['hybrid', 'yolo', 'opencv'], help='โหมดการตรวจจับ')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold')
    parser.add_argument('--lang', type=str, default='thai', help='ภาษา (thai/english)')
    parser.add_argument('--save', action='store_true', help='บันทึกภาพผลลัพธ์')
    parser.add_argument('--no-show', action='store_true', help='ไม่เปิดหน้าต่างแสดงภาพ GUI')
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"ERR: Cannot read image: {args.image}")
        sys.exit(1)

    detector = YOLOBrailleDetector(model_path=args.model, confidence=args.conf, mode=args.mode)
    cells, debug_info = detector.detect(image)

    from decoder import decode_cells, decode_cells_verbose
    decoded = decode_cells(cells, lang=args.lang) if cells else ""
    verbose = decode_cells_verbose(cells, lang=args.lang) if cells else []

    print(f"\n  🎯 Mode:          {detector.mode.upper()}")
    print(f"  🔍 Method Used:   {debug_info.get('method')}")
    print(f"  ⚪ Dots detected: {len(debug_info.get('dots', []))}")
    print(f"  📦 Cells found:   {len(cells)}")
    print(f"  📝 Decoded text:  \"{decoded}\"")

    annotated = detector.annotate_with_text(
        image, debug_info.get('dots', []), cells,
        decoded_text=decoded, verbose_results=verbose, lang=args.lang
    )

    if args.save:
        out_path = args.image.replace('.png', f'_{detector.mode}.png').replace('.jpg', f'_{detector.mode}.jpg')
        cv2.imwrite(out_path, annotated)
        print(f"  💾 Saved to: {out_path}")

    if not args.no_show:
        cv2.namedWindow('Braille Hybrid Detector', cv2.WINDOW_NORMAL)
        cv2.imshow('Braille Hybrid Detector', annotated)
        print("  ⌨️ กดปุ่มใดก็ได้เพื่อปิดหน้าต่าง...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
