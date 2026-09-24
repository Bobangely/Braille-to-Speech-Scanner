"""Painted-dot Braille reading with a retained YOLO path for model evaluation."""

import os
import math
import unicodedata
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# เพิ่ม path ของโปรเจกต์
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dot_fusion import tile_windows, merge_dots
from yolo_cell_stream import CellStream, pair_markers
from colored_braille import ColoredCellStream, DOT_COLORS
from colored_roi import ColoredRoiReader


class YOLOBrailleDetector:
    """
    ตัวอ่านเบรลล์ที่แต้มสี พร้อมเส้นทาง YOLO สำหรับประเมินโมเดล
    พร้อมฟังก์ชันสำหรับใช้งานในกล้อง Real-time (camera_reader.py)
    """

    AVAILABLE_MODES = ['yolo']

    def __init__(self, model_path=None, confidence=0.35, mode='yolo',
                 imgsz=1280, tile_size=1280, tile_overlap=0.2, max_tiles=16,
                 max_det=3000, yolo_pipeline='stream', crop_batch=8, max_cells=256,
                 proposal_confidence=None, dot_color='blue', roi_mode='off'):
        """
        Parameters
        ----------
        model_path : str
            Path ไปยังไฟล์ YOLO model (.pt)
        confidence : float
            Confidence threshold (default: 0.35)
        mode : str
            โหมดการตรวจจับ: 'yolo' เท่านั้น
        dot_color : str | None
            สีที่อ่าน (ค่าเริ่มต้น blue); None ใช้เส้นทาง YOLO เดิมสำหรับประเมินโมเดล
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
        self.dot_color = dot_color
        self._colored_stream = ColoredCellStream(dot_color, max_cells) if dot_color is not None else None
        self._color_readers = {color: ColoredCellStream(color, max_cells) for color in DOT_COLORS}
        if roi_mode not in ('off', 'yolo') or (roi_mode == 'yolo' and dot_color is None):
            raise ValueError('ROI mode must be off or yolo with a selected dot color')
        self.roi_mode = roi_mode
        self._roi_reader = ColoredRoiReader(self._predict_roi_dots) if roi_mode == 'yolo' else None
        self.reader_name = f'COLOR {dot_color.upper()}' if dot_color is not None else 'YOLO'

        # YOLO Inference Resolution — เพิ่มจาก 416 เป็น 1280
        # เพื่อให้ YOLO มองเห็นจุดเบรลล์ขนาดเล็กได้ดีขึ้น (ไม่ย่อภาพจนจุดหายไป)
        self.yolo_imgsz = int(imgsz)

        # โหลดโมเดล YOLO
        if self._colored_stream is None or self._roi_reader is not None:
            self._load_model(model_path)

    def _load_model(self, model_path):
        """Load a local single-class dot detector; fail before scanning on mismatch."""
        path = Path(model_path) if model_path else Path(__file__).resolve().parent/'models/braille_yolo.pt'
        if not path.is_file():
            raise FileNotFoundError(f'Local YOLO weights are unavailable: {path}')
        from ultralytics import YOLO

        model = YOLO(str(path))
        if model.task != 'detect' or model.names != {0: 'braille_dot'}:
            raise ValueError('Expected a detect model with exactly one class: 0: braille_dot')
        self.model = model
        self.model_path = str(path.resolve())
        print(f"  🧠 [YOLO] โหลดโมเดลสำเร็จ: {self.model_path}")


    def is_yolo_ready(self):
        """ตรวจสอบว่า YOLO พร้อมใช้งานหรือไม่"""
        return self.model is not None

    def detect(self, image, lang='thai', dot_color=None, context=None):
        """
        ตรวจจับจุดเบรลล์ตามโหมดปัจจุบัน
        Returns:
            cells : list of dict (dots, center, x, y, grid)
        debug_info : dict (dots, method, crop and overview diagnostics)
        """
        if self._colored_stream is not None:
            color = self.dot_color if dot_color is None else dot_color
            if color not in self._color_readers:
                raise ValueError(f'Unsupported dot color: {color}')
            reader = self._color_readers[color]
            if self._roi_reader is not None:
                return self._roi_reader.detect(image, reader, lang, context)
            return reader.detect(image, lang, context=context)
        if self.model is None:
            raise RuntimeError('YOLO weights are unavailable. Provide a local model file.')
        return self._detect_cell_stream(image, lang)

    def _predict_roi_dots(self, image):
        # One overview call; ROI mode does not run tile or per-cell model passes.
        results = self.model(image, conf=self.proposal_confidence, iou=.2,
                             imgsz=self.yolo_imgsz, max_det=self.max_det, verbose=False)
        return self._extract_yolo_boxes(results, image.shape, self.proposal_confidence)

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
        return cells, dict(method='yolo_cell_stream', dots=dots,
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

    def _cell_hud_header(self, label, char, spacing, budget, height, warning, muted=False):
        """Cache transparent glyphs, shared across consecutive inference overlays."""
        from scanner_ui import CARD, TEXT

        char_size = max(20, min(26, round(spacing*1.1)))
        id_size = max(9, min(11, round(spacing*.55)))
        width = max(1, int(budget))
        key = (label, char, char_size, id_size, width, height, bool(warning), muted)
        cache = getattr(self, '_hud_label_cache', None)
        if cache is None:
            cache = self._hud_label_cache = {}
        if key not in cache:
            char_font, id_font = self._get_font(char_size, bold=True), self._get_font(id_size)
            char_box, id_box = char_font.getbbox(char), id_font.getbbox(label)
            text_width = max(char_box[2]-char_box[0], id_box[2]-id_box[0])
            char_height = char_box[3]-char_box[1]
            char_area = max(char_size, char_height)
            header = Image.new('RGBA', (max(1, text_width)+8, char_area+id_size+9), (0, 0, 0, 0))
            draw = ImageDraw.Draw(header)
            draw.text(((header.width-(char_box[2]-char_box[0]))/2-char_box[0],
                       2+(char_area-char_height)/2-char_box[1]),
                      char, font=char_font, fill=CARD, stroke_width=1, stroke_fill=TEXT)
            draw.text(((header.width-(id_box[2]-id_box[0]))/2-id_box[0],
                       char_area+5-id_box[1]), label, font=id_font, fill=CARD,
                      stroke_width=1, stroke_fill=TEXT)
            # Fit the complete token into its lane, preserving combining marks.
            ratio = min(1., width/header.width, height/header.height)
            if ratio < 1:
                header = header.resize((max(1, round(header.width*ratio)), max(1, round(header.height*ratio))),
                                       Image.Resampling.LANCZOS)
            if len(cache) >= 512:
                cache.clear()
            cache[key] = cv2.cvtColor(np.asarray(header), cv2.COLOR_RGBA2BGRA)
        return cache[key]

    def _draw_cell_hud(self, canvas, cells, verbose_results=None, details=False):
        """Draw confirmed per-cell tokens and measured dots, without decoding again."""
        from scanner_ui import ACCENT, CARD, MUTED, STATUS, TEXT

        def bgr(color):
            return tuple(int(color[i:i+2], 16) for i in (5, 3, 1))

        font = cv2.FONT_HERSHEY_SIMPLEX
        ink, active, inactive = bgr(CARD), bgr(STATUS['Detected']), bgr(MUTED)
        height, width = canvas.shape[:2]
        tokens = verbose_results or []
        # Pillow supports Thai/combining marks. Composite small cached glyphs
        # only when LivePreview rebuilds its overlay, not on every camera frame.
        headers = []

        def pattern_label(text, cx, top, scale):
            (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
            x = max(0, min(width-tw-4, round(cx-tw/2)-2))
            y = max(0, min(height-th-baseline-4, round(top)))
            # Anti-alias dark text against its light outline, not the black
            # overlay backing. Only the glyph-shaped mask reaches the camera.
            glyph = np.full((th+baseline+4, tw+4, 3), bgr(TEXT), np.uint8)
            mask = np.zeros(glyph.shape[:2], np.uint8)
            cv2.putText(mask, text, (2, th+1), font, scale, 255, 2, cv2.LINE_8)
            cv2.putText(glyph, text, (2, th+1), font, scale, ink, 1, cv2.LINE_AA)
            gh, gw = min(glyph.shape[0], height-y), min(glyph.shape[1], width-x)
            cv2.copyTo(glyph[:gh, :gw], mask[:gh, :gw], canvas[y:y+gh, x:x+gw])
            return th+baseline+4

        for index, cell in enumerate(cells):
            grid = cell.get('grid')
            if not grid or not grid.get('slots'):
                continue
            slots = {int(key): value for key, value in grid['slots'].items()}
            # Use the rotated/perspective-aware slots, not a reconstructed 2x3 box.
            distances = [math.hypot(slots[a][0]-slots[b][0], slots[a][1]-slots[b][1])
                         for a, b in ((1, 2), (2, 3), (4, 5), (5, 6), (1, 4))
                         if a in slots and b in slots]
            spacing = min(distances, default=12.)
            radius = max(2, min(12, round(spacing*.32)))
            empty_radius = max(1, round(radius*.55))
            number_scale = min(.45, max(.15, (2*radius-2)/22))
            (nw, nh), _ = cv2.getTextSize('6', font, number_scale, 1)
            x1, y1, x2, y2 = map(round, grid['bbox'])
            token = tokens[index] if index < len(tokens) else None
            warning = (cell.get('row_ambiguous') or cell.get('crop_status') == 'empty'
                       or (token and token.get('warning')))
            char = token.get('char', '') if token else ''
            muted = token is None or bool(token.get('consumed')) or not char.strip()
            if warning or '�' in char:
                char, warning = '?', True
            elif token is None:
                char = '…'  # The camera withholds tokens until confirmation.
            elif token.get('consumed'):
                char = '+'  # Continuation of a multi-cell symbol; don't duplicate it.
            elif not char.strip():
                char = '·'  # A decoded indicator/blank has no printable character.
            elif unicodedata.category(char[0]).startswith('M'):
                char = '◌' + char  # Make a standalone Thai mark legible in its label.
            edge = bgr(STATUS['Check image']) if warning else bgr(ACCENT)
            # Follow measured slot geometry even when the page is tilted. Only
            # these display separators are extended/clipped to the existing bbox.
            if all(dot in slots for dot in range(1, 7)) and x2 > x1 and y2 > y1:
                def midpoint(a, b):
                    return tuple((slots[a][axis]+slots[b][axis])/2 for axis in (0, 1))

                for start, end in ((midpoint(1, 4), midpoint(3, 6)),
                                   (midpoint(1, 2), midpoint(4, 5)),
                                   (midpoint(2, 3), midpoint(5, 6))):
                    delta = tuple(end[axis]-start[axis] for axis in (0, 1))
                    reach = math.hypot(x2-x1, y2-y1)/max(1., math.hypot(*delta))
                    a = tuple(round(start[axis]-delta[axis]*reach) for axis in (0, 1))
                    b = tuple(round(end[axis]+delta[axis]*reach) for axis in (0, 1))
                    visible, a, b = cv2.clipLine((x1, y1, x2-x1+1, y2-y1+1), a, b)
                    if visible:
                        cv2.line(canvas, a, b, inactive, 1, cv2.LINE_8)
            # The overlay uses a binary copy mask. Avoid anti-aliased edges on
            # its black backing, which would leave dark halos over the camera.
            cv2.rectangle(canvas, (x1, y1), (x2, y2), edge, 2, cv2.LINE_8)
            for dot_id, point in slots.items():
                center = tuple(map(round, point))
                is_active = dot_id in cell['dots']
                if is_active:
                    cv2.circle(canvas, center, radius, active, 2, cv2.LINE_8)
                else:
                    # Hollow markers preserve camera pixels (also in cached overlays).
                    cv2.circle(canvas, center, empty_radius, inactive, 1, cv2.LINE_8)
                if details:
                    cv2.putText(canvas, str(dot_id), (center[0]-nw//2, center[1]+nh//2),
                                font, number_scale, ink, 1, cv2.LINE_AA)

            cx = (x1+x2)/2
            budget = max(8., min(width, (x2-x1)*1.6))
            for neighbor in cells[max(0, index-1):index] + cells[index+1:index+2]:
                if neighbor.get('line_id') == cell.get('line_id') and neighbor.get('grid'):
                    bx1, _, bx2, _ = neighbor['grid']['bbox']
                    distance = abs((bx1+bx2)/2-cx)
                    if distance > 0:
                        budget = min(budget, max(8., distance-3))
            label = f"C{cell.get('track_id', index+1)}"
            header_pixels = self._cell_hud_header(label, char, spacing, min(width, budget), height, warning, muted)
            header_height, header_width = header_pixels.shape[:2]
            label_top = y1-header_height-3
            header_below = label_top < 0
            if header_below:
                label_top = y2+3
            label_top = max(0, min(height-header_height, label_top))
            label_left = max(0, min(width-header_width, round(cx-header_width/2)))
            headers.append((label_left, label_top, header_pixels))

            pattern = '[' + ','.join(map(str, sorted(cell['dots']))) + ']'
            pattern_scale = min(.34, max(.23, spacing*.022))
            # Wrap dense patterns within the cell's horizontal lane rather than
            # drawing over its neighbor. The underlying pattern is never shortened.
            lines = [pattern]
            if cv2.getTextSize(pattern, font, pattern_scale, 1)[0][0]+4 > budget and len(cell['dots']) > 3:
                parts = list(map(str, sorted(cell['dots'])))
                middle = (len(parts)+1)//2
                lines = ['['+','.join(parts[:middle])+',', ','.join(parts[middle:])+']']
            max_width = max(cv2.getTextSize(line, font, pattern_scale, 1)[0][0] for line in lines)
            pattern_scale = min(pattern_scale, pattern_scale*max(4, budget-4)/max(1, max_width))
            line_sizes = [cv2.getTextSize(line, font, pattern_scale, 1) for line in lines]
            block_height = sum(size[0][1]+size[1]+4 for size in line_sizes)
            top = label_top+header_height+3 if header_below else y2+3
            if top+block_height > height:
                # Keep the entire pattern together when zoom puts a cell at the
                # bottom edge; clamping each line separately would erase a line.
                top = max(0, label_top-block_height-2)
            for line in lines:
                top += pattern_label(line, cx, top, pattern_scale)

        for x, y, header in headers:
            h, w = header.shape[:2]
            # Copy only glyph pixels, never the rectangular cache backing.
            cv2.copyTo(header[:, :, :3], header[:, :, 3], canvas[y:y+h, x:x+w])

    def annotate_with_text(self, image, dots, cells, decoded_text="", verbose_results=None, lang="thai",
                           reader_name=None, details=True, footer=True, cell_hud=False):
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
        banner_h = (80 + 32*max(0, len(text_lines)-1)) if footer else 0
        canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
        canvas[:h, :w] = image.copy()
        canvas[h:, :] = (28, 22, 16)  # Dark navy slate

        # 1. วาด 2x3 Grid รอบแต่ละ Cell
        for idx, cell in enumerate(() if cell_hud else cells, 1):
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
            if not details:
                cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (190, 155, 45), 1)
                continue
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
            if cell_hud and not details:
                continue
            if not details:
                cv2.circle(canvas, (round(cx), round(cy)), 2, (190, 155, 45), -1)
                continue
            conf = dot.get('confidence')
            bbox = dot.get('bbox')
            radius = int(np.sqrt(dot['area'] / np.pi))

            if not cell_hud:
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

        if cell_hud:
            self._draw_cell_hud(canvas, cells, verbose_results, details=details)

        if not details and not footer:
            return canvas

        # 3. วาดข้อความผลลัพธ์และตัวอักษรด้วย PIL
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        font_large = self._get_font(size=24, bold=True)
        font_mid = self._get_font(size=18, bold=True)
        font_small = self._get_font(size=14, bold=False)

        # ตัวอักษรเหนือ Cell
        if details and verbose_results and not cell_hud:
            for idx, item in enumerate(verbose_results, 1):
                if item.get('consumed'):
                    continue
                if idx - 1 < len(cells):
                    cell = cells[idx - 1]
                    grid = cell.get('grid')
                    if grid:
                        x_min, y_min = grid['bbox'][0], grid['bbox'][1]
                        char_text = f"C{cell.get('track_id', idx)}: {item['char']}"
                        draw.text((x_min + 2, y_min - 25),
                                  char_text, fill=(255, 190, 0), font=font_mid)
        elif details and not cell_hud:
            for idx, cell in enumerate(cells, 1):
                grid = cell.get('grid')
                if grid:
                    x_min, y_min = grid['bbox'][0], grid['bbox'][1]
                    draw.text((x_min + 2, y_min - 22), f"C{cell.get('track_id', idx)}", fill=(255, 190, 0), font=font_small)

        if not footer:
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 4. Bottom Banner
        draw.line([(0, h), (w, h)], fill=(70, 85, 105), width=2)

        # Mode Badge
        mode_badge = f"MODE: {reader_name or getattr(self, 'reader_name', 'YOLO')} CELL STREAM"
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
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description='Painted Braille Reader')
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
    parser.add_argument('--dot-color', choices=DOT_COLORS, default='blue', help='Read only painted dots')
    parser.add_argument('--roi', choices=['off', 'yolo'], default='off', help='Optional YOLO-dot ROI')
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
                                  proposal_confidence=args.proposal_conf, dot_color=args.dot_color,
                                  roi_mode=args.roi)
    cells, debug_info = detector.detect(image, lang=args.lang)

    from decoder import decode_cells, decode_cells_verbose
    decoded = decode_cells(cells, lang=args.lang) if cells else ""
    verbose = decode_cells_verbose(cells, lang=args.lang) if cells else []
    if args.dump_stream:
        from tools.diagnostics.export_yolo_stream import export_stream
        from pathlib import Path
        destination = Path(__file__).resolve().parent/'output'/(Path(args.image).stem+'_stream')
        manifest = export_stream(image, cells, debug_info, destination, args.lang, source=args.image,
                                 detector=detector)
        print(f'  Cell stream report: {manifest}')

    print(f"\n   Reader:        {detector.reader_name}")
    print(f"   Method Used:   {debug_info.get('method')}")
    print(f"   Dots detected: {len(debug_info.get('dots', []))}")
    print(f"   Cells found:   {len(cells)}")
    print(f"   Decoded text:  \"{decoded}\"")

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
        print(f"   Saved to: {out_path}")

    if not args.no_show:
        cv2.namedWindow('Painted Braille Reader', cv2.WINDOW_NORMAL)
        cv2.imshow('Painted Braille Reader', annotated)
        print("   กดปุ่มใดก็ได้เพื่อปิดหน้าต่าง...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
