"""
YOLO-based Braille Dot Detector
=================================
ใช้ YOLOv8 ตรวจจับจุดเบรลล์แต่ละจุด (1 class: "braille_dot")
แล้วส่ง bounding box centroids ไปให้ Grid Clustering ของเดิมจัดกลุ่มเป็น Braille cells

สถาปัตยกรรม Hybrid:
    ภาพ → [YOLOv8] → Bounding Boxes → [Grid Clustering] → Cells → [Decoder] → ข้อความ

วิธีการทำงาน:
    1. YOLO รับภาพ input → ส่งออก bounding boxes ของจุดเบรลล์ทุกจุด
    2. แปลง bounding boxes เป็น centroids (จุดศูนย์กลาง) + area
    3. ส่ง centroids ให้ _cluster_into_cells() ของ BrailleDetector (OpenCV) เดิม
    4. Decoder ถอดรหัสเป็นภาษาไทย/อังกฤษตามปกติ

ข้อดีของ YOLO vs OpenCV ล้วน:
    - ไม่ต้องกำหนด HSV threshold ตายตัว → ทำงานได้กับสีพื้นหลังหลากหลาย
    - ทนต่อแสง/เงา/noise ได้ดีกว่า (เรียนรู้จาก training data)
    - Detect จุดนูน (embossed braille) ที่ไม่มีสีได้
    - Real-time 100+ FPS บน GPU (YOLOv8 nano)
"""

import cv2
import numpy as np
import os
import sys

# เพิ่ม path ของโปรเจกต์
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import BrailleDetector
from config import DetectionConfig


class YOLOBrailleDetector:
    """
    ตรวจจับจุดเบรลล์ด้วย YOLO + ใช้ Grid Clustering ของ OpenCV เดิม

    Usage:
        detector = YOLOBrailleDetector(model_path='runs/detect/train/weights/best.pt')
        cells, debug_info = detector.detect(image)
    """

    def __init__(self, model_path=None, confidence=0.5, fallback_color='blue'):
        """
        Parameters
        ----------
        model_path : str
            Path ไปยังไฟล์ YOLO model (.pt)
            ถ้าไม่ระบุ จะลองหาจาก default paths
        confidence : float
            ค่า confidence threshold ขั้นต่ำ (0.0 - 1.0)
            YOLO จะเอาเฉพาะ detection ที่มั่นใจมากกว่าค่านี้
            - ค่าต่ำ (0.3): ได้จุดเยอะ แต่อาจมี false positive
            - ค่าสูง (0.7): ได้เฉพาะจุดที่มั่นใจ แต่อาจพลาดจุดจริงบางจุด
        fallback_color : str
            สีที่ใช้เมื่อ fallback ไป OpenCV detector
        """
        self.confidence = confidence
        self.fallback_color = fallback_color
        self.model = None
        self.model_path = model_path

        # พยายามโหลด YOLO model
        self._load_model(model_path)

        # เก็บ OpenCV detector ไว้ใช้สำหรับ:
        #   1. Grid Clustering (จัดกลุ่มจุดเป็น cell)
        #   2. Fallback เมื่อ YOLO ไม่พร้อม
        self._opencv_detector = BrailleDetector(dot_color=fallback_color)

    def _load_model(self, model_path):
        """
        โหลด YOLO model จากไฟล์ .pt

        ลำดับการหา model:
            1. path ที่ระบุ
            2. runs/detect/train/weights/best.pt (default training output)
            3. models/braille_yolo.pt (custom path)
        """
        search_paths = [
            model_path,
            'runs/detect/train/weights/best.pt',
            'models/braille_yolo.pt',
            'braille_yolo.pt',
        ]

        for path in search_paths:
            if path and os.path.exists(path):
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(path)
                    self.model_path = path
                    print(f"  🧠 YOLO Model loaded: {path}")
                    return
                except Exception as e:
                    print(f"  ⚠️ YOLO load error ({path}): {e}")

        print("  ℹ️ YOLO model not found — จะใช้ OpenCV detector เป็น fallback")
        print("     (รัน train_yolo.py เพื่อ train model ก่อน)")

    def detect(self, image):
        """
        ตรวจจับจุดเบรลล์ด้วย YOLO (หรือ OpenCV fallback)

        Parameters
        ----------
        image : np.ndarray
            ภาพ BGR จาก OpenCV

        Returns
        -------
        cells : list of dict
            เหมือนกับ BrailleDetector.detect()
            แต่ละ cell มี: 'dots' (frozenset), 'center', 'x', 'y', 'grid'
        debug_info : dict
            ข้อมูล debug: 'mask', 'dots', 'annotated', 'method'
        """
        if self.model is None:
            # Fallback ไปใช้ OpenCV
            cells, debug_info = self._opencv_detector.detect(image)
            debug_info['method'] = 'opencv_fallback'
            return cells, debug_info

        return self._detect_with_yolo(image)

    def _detect_with_yolo(self, image):
        """
        Core YOLO detection pipeline

        ขั้นตอน:
        1. ส่งภาพเข้า YOLO → ได้ bounding boxes
        2. แปลง bounding boxes เป็น dot centroids
        3. ส่ง centroids ไป Grid Clustering ของ OpenCV
        """
        # --- 1. YOLO Inference ---
        # verbose=False: ไม่ print ผลลัพธ์ทุก frame (สำหรับ real-time)
        # conf=self.confidence: กรอง detection ที่ confidence ต่ำกว่านี้ออก
        results = self.model(image, verbose=False, conf=self.confidence)

        # --- 2. แปลง bounding boxes เป็น dot list ---
        dots = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                # ดึง bounding box coordinates (x1, y1, x2, y2)
                # xyxy format: พิกัดมุมซ้ายบน (x1,y1) และมุมขวาล่าง (x2,y2)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())

                # คำนวณจุดศูนย์กลาง (centroid) ของ bounding box
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # คำนวณพื้นที่ (area) ของ bounding box
                w = x2 - x1
                h = y2 - y1
                area = w * h

                dots.append({
                    'center': (cx, cy),
                    'area': area,
                    'circularity': 1.0,  # YOLO ไม่มีค่านี้ ให้ 1.0 (สมบูรณ์)
                    'confidence': conf,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                })

        # --- 3. ส่งไป Grid Clustering ---
        cells = []
        if len(dots) >= 1:
            cells = self._opencv_detector._cluster_into_cells(dots)

        # --- 4. สร้าง debug info ---
        # สร้าง mask จำลอง (วาดวงกลมตรงตำแหน่งจุดที่ YOLO detect ได้)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for dot in dots:
            cx, cy = dot['center']
            r = int(np.sqrt(dot['area'] / np.pi))
            cv2.circle(mask, (int(cx), int(cy)), max(r, 3), 255, -1)

        # สร้าง annotated image
        annotated = self._annotate_yolo(image.copy(), dots, cells, results)

        debug_info = {
            'mask': mask,
            'dots': dots,
            'annotated': annotated,
            'method': 'yolo',
            'num_detections': len(dots),
        }

        return cells, debug_info

    def _annotate_yolo(self, image, dots, cells, yolo_results):
        """
        วาด annotation บนภาพ แสดง YOLO detection results

        จะวาด:
        1. Bounding box สีเขียว รอบจุดที่ YOLO detect ได้
        2. Confidence score ของแต่ละจุด
        3. Grid lines ของ Braille cells (เหมือน OpenCV detector)
        4. Banner ด้านล่างแสดง "YOLO Mode"
        """
        from PIL import Image, ImageDraw, ImageFont

        h, w = image.shape[:2]
        banner_h = 80
        canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
        canvas[:h, :w] = image
        canvas[h:, :] = (30, 23, 15)

        # วาด bounding box + confidence ของ YOLO
        for dot in dots:
            cx, cy = dot['center']
            conf = dot.get('confidence', 0)
            bbox = dot.get('bbox')

            # วงกลมเขียวรอบจุด
            r = int(np.sqrt(dot['area'] / np.pi))
            cv2.circle(canvas, (int(cx), int(cy)), r + 3, (0, 255, 0), 2)
            cv2.circle(canvas, (int(cx), int(cy)), 2, (0, 0, 255), -1)

            # Bounding box (สีฟ้า)
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 180, 0), 1)

            # Confidence label
            if conf > 0:
                label = f"{conf:.0%}"
                cv2.putText(
                    canvas, label,
                    (int(cx) - 10, int(cy) - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1,
                )

        # วาด Grid lines ของ cells
        for idx, cell in enumerate(cells, 1):
            grid = cell.get('grid')
            if not grid:
                continue
            x_min, y_min, x_max, y_max = grid['bbox']
            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (255, 200, 0), 2)

            # Cell label
            dots_str = ','.join(map(str, sorted(cell['dots'])))
            cv2.putText(
                canvas, f"C{idx}:[{dots_str}]",
                (x_min, y_max + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1,
            )

        # Banner ด้านล่าง
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # หา font (ใช้ของเดิม)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 22)
            font_small = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

        # วาดข้อความ banner
        from decoder import decode_cells
        decoded = decode_cells(cells, lang='thai')
        y_banner = h + 10

        draw.text((15, y_banner), f"🧠 YOLO Mode", font=font_small, fill=(0, 200, 255))
        draw.text((150, y_banner), f"| {len(dots)} dots | {len(cells)} cells",
                  font=font_small, fill=(200, 200, 200))
        if decoded:
            draw.text((15, y_banner + 25), f"ข้อความ: \"{decoded}\"",
                      font=font, fill=(255, 255, 100))

        canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return canvas

    def is_yolo_ready(self):
        """เช็คว่า YOLO model พร้อมใช้งานหรือไม่"""
        return self.model is not None


# =============================================================================
# CLI: ทดสอบ YOLO detector กับภาพ
# =============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLO Braille Dot Detector')
    parser.add_argument('image', type=str, help='Path ของภาพ')
    parser.add_argument('--model', type=str, default=None, help='Path ของ YOLO model (.pt)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--lang', type=str, default='thai', help='ภาษา (thai/english)')
    parser.add_argument('--save', action='store_true', help='บันทึกภาพผลลัพธ์')
    args = parser.parse_args()

    # โหลดภาพ
    image = cv2.imread(args.image)
    if image is None:
        print(f"ERR: Cannot read image: {args.image}")
        sys.exit(1)

    # สร้าง detector
    detector = YOLOBrailleDetector(model_path=args.model, confidence=args.conf)

    # Detect
    cells, debug_info = detector.detect(image)

    # แสดงผล
    method = debug_info.get('method', 'unknown')
    print(f"\n  Detection method: {method}")
    print(f"  Dots detected: {len(debug_info['dots'])}")
    print(f"  Cells found: {len(cells)}")

    if cells:
        from decoder import decode_cells
        decoded = decode_cells(cells, lang=args.lang)
        print(f"  Decoded text: \"{decoded}\"")

    # แสดงภาพ
    cv2.namedWindow('YOLO Braille Detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow('YOLO Braille Detection', debug_info['annotated'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.save:
        out_path = args.image.replace('.png', '_yolo.png').replace('.jpg', '_yolo.jpg')
        cv2.imwrite(out_path, debug_info['annotated'])
        print(f"  Saved: {out_path}")
