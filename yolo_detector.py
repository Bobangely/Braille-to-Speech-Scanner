"""YOLO-only Braille dot detection, isolated cell reads, and annotation."""

import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# เพิ่ม path ของโปรเจกต์
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dot_fusion import tile_windows, merge_dots
from yolo_cell_stream import CellStream, pair_markers


class YOLOBrailleDetector:
    """
    ตัวตรวจจับอักษรเบรลล์ด้วย YOLO และการอ่าน crop รายเซลล์
    พร้อมฟังก์ชันสำหรับใช้งานในกล้อง Real-time (camera_reader.py)
    """

    AVAILABLE_MODES = ['yolo']

    def __init__(self, model_path=None, confidence=0.35, mode='yolo',
                 imgsz=1280, tile_size=1280, tile_overlap=0.2, max_tiles=16,
                 max_det=3000, yolo_pipeline='stream', crop_batch=8, max_cells=256,
                 proposal_confidence=None):
        """
        Parameters
        ----------
        model_path : str
            Path ไปยังไฟล์ YOLO model (.pt)
        confidence : float
            Confidence threshold (default: 0.35)
        mode : str
            โหมดการตรวจจับ: 'yolo' เท่านั้น
        """
        self.confidence = float(confidence)
        if not 0 < self.confidence <= 1:
            raise ValueError('confidence must be in (0, 1]')
        self.proposal_confidence = (min(.3, self.confidence) if proposal_confidence is None
                                    else float(proposal_confidence))
        if not 0 < self.proposal_confidence <= self.confidence:
            raise ValueError('proposal confidence must be positive and no higher than final confidence')
        if imgsz < 32 or imgsz % 32 or (tile_size != 0 and tile_size < 32):
            raise ValueError('imgsz must be a multiple of 32; tile_size must be 0 or >=32')
        if not 0 <= tile_overlap < 1 or max_tiles < 1 or max_det < 1:
            raise ValueError('Invalid tile overlap, tile budget, or detection limit')
        self.tile_size = int(tile_size)
        self.tile_overlap = float(tile_overlap)
        self.max_tiles = int(max_tiles)
        self.max_det = int(max_det)
        self._inference_info = {}
        if yolo_pipeline != 'stream':
            raise ValueError('Only the YOLO cell stream pipeline is supported')
        self.yolo_pipeline = yolo_pipeline
        self._cell_stream = CellStream(self._predict_crop_batch, crop_batch, max_cells)
        if mode.lower() != 'yolo':
            raise ValueError('Only YOLO detection is supported')
        self.mode = 'yolo'
        self.model = None
        self.model_path = model_path
        self._font_cache = {}

        # YOLO Inference Resolution — เพิ่มจาก 416 เป็น 1280
        # เพื่อให้ YOLO มองเห็นจุดเบรลล์ขนาดเล็กได้ดีขึ้น (ไม่ย่อภาพจนจุดหายไป)
        self.yolo_imgsz = int(imgsz)

        # โหลดโมเดล YOLO
        self._load_model(model_path)

    def _load_model(self, model_path):
        """ค้นหาและโหลดไฟล์ YOLO weights"""
        search_paths = [model_path] if model_path else [str(Path(__file__).resolve().parent/'models/braille_yolo.pt')]

        for path in search_paths:
            if path and path != model_path:
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
            if path and os.path.exists(path):
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(path)
                    self.model_path = path
                    print(f"  🧠 [YOLO] โหลดโมเดลสำเร็จ: {path}")
                    return
                except Exception as e:
                    print(f"  ⚠️ [YOLO] โหลดโมเดลล้มเหลว ({path}): {e}")

        print("  ℹ️ [YOLO] ไม่พบโมเดล YOLO ที่โหลดได้; โหมด YOLO จะรายงานข้อผิดพลาด")
        self.model = None


    def is_yolo_ready(self):
        """ตรวจสอบว่า YOLO พร้อมใช้งานหรือไม่"""
        return self.model is not None

    def detect(self, image, lang='thai'):
        """
        ตรวจจับจุดเบรลล์ตามโหมดปัจจุบัน
        Returns:
            cells : list of dict (dots, center, x, y, grid)
            debug_info : dict (dots, mask, annotated, method)
        """
        if self.model is None:
            raise RuntimeError('YOLO weights are unavailable. Provide a local model file.')
        return self._detect_cell_stream(image, lang)

    def _predict_crop_batch(self, crops):
        results = self.model(crops, verbose=False, conf=self.confidence,
                             imgsz=320, max_det=12, iou=0.2)
        return [self._extract_yolo_boxes([result], crop.shape)
                for result, crop in zip(results, crops)]

    def _detect_cell_stream(self, image, lang):
        # Tiny dots can produce shifted duplicate boxes despite default NMS.
        # Suppress overlaps before geometry estimates the inter-dot spacing.
        overview = self._predict_dots(image, iou=0.2, confidence=self.proposal_confidence)
        inference_info = dict(self._inference_info)
        cells, symbols = [], []
        for group in pair_markers(self._cell_stream.iter_cells(image, overview), lang):
            start = len(cells)
            cells.extend(group)
            symbols.append(dict(cell_start=start, cell_end=len(cells)-1,
                                line_id=group[0]['line_id'], symbol=group[0].get('symbol')))
        dots = [dot for cell in cells for dot in cell['read_dots']]
        mask = np.zeros(image.shape[:2], np.uint8)
        for dot in dots:
            cv2.circle(mask, tuple(int(round(v)) for v in dot['center']), 2, 255, -1)
        return cells, dict(method='yolo_cell_stream', dots=dots, mask=mask,
                           num_detections=len(dots), overview_detections=len(overview),
                           line_count=len({c['line_id'] for c in cells}),
                           crop_count=len(cells), symbols=symbols,
                           crop_disagreements=sum(c['crop_disagreement'] for c in cells),
                           proposal_confidence=self.proposal_confidence,
                           read_confidence=self.confidence,
                           **inference_info)


    def _predict_dots(self, image, iou=0.7, confidence=None):
        """Overview plus sequential overlapping tiles; coordinates stay in source pixels."""
        height, width = image.shape[:2]
        windows = tile_windows(width, height, self.tile_size,
                               self.tile_overlap, self.max_tiles)
        self._inference_info = {'tile_count': 0, 'tile_budget_exceeded': windows is None}
        threshold = self.confidence if confidence is None else confidence

        def predict(patch):
            results = self.model(patch, verbose=False, conf=threshold,
                                 imgsz=self.yolo_imgsz, max_det=self.max_det, iou=iou)
            return self._extract_yolo_boxes(results, patch.shape, confidence=threshold)

        dots = predict(image)
        for x1, y1, x2, y2 in windows or []:
            patch_dots = predict(image[y1:y2, x1:x2])
            translated = []
            for dot in patch_dots:
                bx1, by1, bx2, by2 = dot['bbox']
                # Partial dots at internal tile edges are recovered by overlap.
                if ((x1 > 0 and bx1 <= 1) or (y1 > 0 and by1 <= 1)
                        or (x2 < width and bx2 >= x2 - x1 - 1)
                        or (y2 < height and by2 >= y2 - y1 - 1)):
                    continue
                translated.append(dict(dot,
                    center=(dot['center'][0] + x1, dot['center'][1] + y1),
                    bbox=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)))
            dots = merge_dots(dots, translated)
            self._inference_info['tile_count'] += 1
        return dots

    def _extract_yolo_boxes(self, results, img_shape, confidence=None):
        """แปลงผลลัพธ์จาก Ultralytics YOLO เป็น dot dictionary"""
        dots = []
        threshold = self.confidence if confidence is None else confidence
        if len(results) == 0 or results[0].boxes is None:
            return dots

        boxes = results[0].boxes
        h_img, w_img = img_shape[:2]

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            if not np.isfinite([x1, y1, x2, y2, conf]).all():
                continue
            x1, x2 = np.clip([x1, x2], 0, w_img)
            y1, y2 = np.clip([y1, y2], 0, h_img)
            if x2 <= x1 or y2 <= y1 or conf < threshold:
                continue

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
            })
        return dots


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
        text_lines = decoded_text.splitlines()[:3]
        if len(decoded_text.splitlines()) > 3:
            text_lines[-1] += ' …'
        banner_h = 80 + 32*max(0, len(text_lines)-1)
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
            radius = int(np.sqrt(dot['area'] / np.pi))

            cv2.circle(canvas, (int(cx), int(cy)), radius + 3, (0, 255, 0), 2)
            cv2.circle(canvas, (int(cx), int(cy)), 2, (0, 0, 255), -1)

            # Bounding Box
            if bbox:
                bx1, by1, bx2, by2 = bbox
                box_color = (255, 180, 0)
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
                if item.get('consumed'):
                    continue
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
        mode_badge = 'MODE: YOLO CELL STREAM'
        mode_color = (0, 210, 255)

        if decoded_text:
            braille_chars = [item.get('unicode', '·') for item in (verbose_results or [])]
            braille_disp = ' '.join(braille_chars)

            for line_index, line in enumerate(text_lines):
                label = 'ข้อความ' if lang in ('thai', 'th') else 'Text'
                text_disp = f'{label}: {line}' if line_index == 0 else line
                draw.text((20, h + 12 + line_index*32), text_disp, fill=(255, 255, 100), font=font_large)

            lang_label = "ไทย (Thai)" if lang == 'thai' else "English"
            sub_text = f"{mode_badge}  |  Braille: {braille_disp}  |  {len(cells)} Cells ({len(dots)} Dots)  |  {lang_label}"
            draw.text((20, h + banner_h - 32), sub_text, fill=(180, 210, 230), font=font_small)
        else:
            draw.text((20, h + 22), f"{mode_badge}  -  พร้อมสแกน...", fill=mode_color, font=font_mid)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# =============================================================================
# CLI Testing Interface
# =============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLO Braille Dot Detector')
    parser.add_argument('image', type=str, help='Path ของภาพ')
    parser.add_argument('--model', type=str, default=None, help='Path ของ YOLO model (.pt)')
    parser.add_argument('--mode', type=str, default='yolo', choices=['yolo'], help='โหมดการตรวจจับ')
    parser.add_argument('--yolo-pipeline', choices=['stream'], default='stream')
    parser.add_argument('--crop-batch', type=int, default=8)
    parser.add_argument('--max-cells', type=int, default=256)
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold')
    parser.add_argument('--proposal-conf', type=float, default=None, help='Overview threshold (default: min(0.3, conf))')
    parser.add_argument('--imgsz', type=int, default=1280, help='YOLO input size (multiple of 32)')
    parser.add_argument('--tile-size', type=int, default=1280, help='Source tile size; 0 disables tiles')
    parser.add_argument('--max-tiles', type=int, default=16, help='Maximum extra tile predictions')
    parser.add_argument('--max-det', type=int, default=3000, help='Maximum dots per prediction')
    parser.add_argument('--lang', type=str, default='thai', help='ภาษา (thai/english)')
    parser.add_argument('--save', action='store_true', help='บันทึกภาพผลลัพธ์')
    parser.add_argument('--no-show', action='store_true', help='ไม่เปิดหน้าต่างแสดงภาพ GUI')
    parser.add_argument('--dump-stream', action='store_true', help='Export cell crops and JSON to output/<image>_stream/')
    args = parser.parse_args()
    if args.dump_stream and (args.mode != 'yolo' or args.yolo_pipeline != 'stream'):
        parser.error('--dump-stream requires --mode yolo --yolo-pipeline stream')

    image = cv2.imread(args.image)
    if image is None:
        print(f"ERR: Cannot read image: {args.image}")
        sys.exit(1)

    detector = YOLOBrailleDetector(model_path=args.model, confidence=args.conf, mode=args.mode,
                                  imgsz=args.imgsz,
                                  tile_size=args.tile_size, max_tiles=args.max_tiles,
                                  max_det=args.max_det, yolo_pipeline=args.yolo_pipeline,
                                  crop_batch=args.crop_batch, max_cells=args.max_cells,
                                  proposal_confidence=args.proposal_conf)
    cells, debug_info = detector.detect(image, lang=args.lang)

    from decoder import decode_cells, decode_cells_verbose
    decoded = decode_cells(cells, lang=args.lang) if cells else ""
    verbose = decode_cells_verbose(cells, lang=args.lang) if cells else []
    if args.dump_stream:
        from tools.diagnostics.export_yolo_stream import export_stream
        from pathlib import Path
        destination = Path(__file__).resolve().parent/'output'/(Path(args.image).stem+'_stream')
        manifest = export_stream(image, cells, debug_info, destination, args.lang, source=args.image)
        print(f'  Cell stream report: {manifest}')

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
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.image))[0]
        out_path = os.path.join(output_dir, f'{stem}_{detector.mode}.png')
        if not cv2.imwrite(out_path, annotated):
            raise OSError(f'Cannot save annotation: {out_path}')
        print(f"  💾 Saved to: {out_path}")

    if not args.no_show:
        cv2.namedWindow('Braille YOLO Detector', cv2.WINDOW_NORMAL)
        cv2.imshow('Braille YOLO Detector', annotated)
        print("  ⌨️ กดปุ่มใดก็ได้เพื่อปิดหน้าต่าง...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
