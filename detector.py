"""
Braille Reader - Dot Detector
===============================
ตรวจจับจุดสีบนภาพอักษรเบรลล์ด้วย OpenCV

Pipeline:
    1. Preprocess (blur, resize)
    2. Color Segmentation (HSV threshold)
    3. Morphological Cleanup (close + open)
    4. Contour Detection + Filtering (area, circularity)
    5. Grid Alignment (cluster dots into Braille cells)
"""

import cv2
import numpy as np

from config import DetectionConfig


class BrailleDetector:
    """ตรวจจับจุดสีและจัดเป็น Braille cells"""

    def __init__(self, dot_color='blue', config=None):
        """
        Parameters
        ----------
        dot_color : str
            สีของจุดที่แต้ม ('blue', 'red', 'green')
        config : DetectionConfig, optional
            ค่า config สำหรับ detection (ใช้ default ถ้าไม่ระบุ)
        """
        self.config = config or DetectionConfig()
        self.dot_color = dot_color.lower()

        if self.dot_color not in self.config.SUPPORTED_COLORS:
            raise ValueError(
                f"ไม่รองรับสี '{dot_color}' "
                f"(รองรับ: {self.config.SUPPORTED_COLORS})"
            )

    # =================================================================
    # Public API
    # =================================================================

    def detect(self, image):
        """
        ตรวจจับอักษรเบรลล์จากภาพ

        Parameters
        ----------
        image : np.ndarray
            ภาพ BGR จาก cv2.imread()

        Returns
        -------
        cells : list of dict
            แต่ละ cell มี: 'dots' (frozenset), 'center' (x,y), 'x' (float)
        debug_info : dict
            ข้อมูล debug: mask, dot_centers, annotated_image
        """
        # 1. Preprocess
        processed = self._preprocess(image)

        # 2. Color segmentation → binary mask
        mask = self._color_segment(processed)

        # 3. Morphological cleanup
        mask = self._morph_clean(mask)

        # 4. Find dot centroids
        dots = self._find_dots(mask)

        # 5. Cluster into Braille cells
        cells = []
        if len(dots) >= 1:
            cells = self._cluster_into_cells(dots)

        # สร้าง debug info
        debug_info = {
            'mask': mask,
            'dots': dots,
            'annotated': self._annotate(image.copy(), dots, cells),
        }

        return cells, debug_info

    # =================================================================
    # Step 1 — Preprocessing
    # =================================================================

    def _preprocess(self, image):
        """Gaussian blur เพื่อลด noise"""
        return cv2.GaussianBlur(image, (5, 5), 0)

    # =================================================================
    # Step 2 — Color Segmentation
    # =================================================================

    def _color_segment(self, image):
        """สร้าง binary mask จากสีที่กำหนด (ใน HSV space)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if self.dot_color == 'blue':
            lower = np.array(self.config.BLUE_HSV_LOWER)
            upper = np.array(self.config.BLUE_HSV_UPPER)
            mask = cv2.inRange(hsv, lower, upper)

        elif self.dot_color == 'red':
            # สีแดงต้องใช้ 2 range เพราะ Hue wrap around
            mask1 = cv2.inRange(
                hsv,
                np.array(self.config.RED_HSV_LOWER_1),
                np.array(self.config.RED_HSV_UPPER_1),
            )
            mask2 = cv2.inRange(
                hsv,
                np.array(self.config.RED_HSV_LOWER_2),
                np.array(self.config.RED_HSV_UPPER_2),
            )
            mask = cv2.bitwise_or(mask1, mask2)

        elif self.dot_color == 'green':
            lower = np.array(self.config.GREEN_HSV_LOWER)
            upper = np.array(self.config.GREEN_HSV_UPPER)
            mask = cv2.inRange(hsv, lower, upper)

        elif self.dot_color == 'black':
            # จุดสีดำ / กราฟิกเอกสาร (Grayscale thresholding)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(
                gray, self.config.BLACK_INTENSITY_MAX, 255, cv2.THRESH_BINARY_INV
            )

        return mask

    # =================================================================
    # Step 3 — Morphological Cleanup
    # =================================================================

    def _morph_clean(self, mask):
        """ปิดรูเล็ก + ลบ noise ด้วย morphology"""
        k = self.config.MORPH_KERNEL_SIZE
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # Close: ปิดรูเล็กๆ ภายในจุด
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Open: ลบ noise เล็กๆ
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    # =================================================================
    # Step 4 — Dot Detection
    # =================================================================

    def _find_dots(self, mask):
        """
        หา contours ที่เป็นจุดกลม แล้ว return centroids

        Returns
        -------
        list of dict
            'center': (cx, cy), 'area': float, 'circularity': float
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        dots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # กรองตามขนาด
            if area < self.config.MIN_DOT_AREA:
                continue
            if area > self.config.MAX_DOT_AREA:
                continue

            # กรองตามความกลม (circularity)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)

            if circularity < self.config.MIN_CIRCULARITY:
                continue

            # หาจุดศูนย์กลาง (centroid)
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            dots.append({
                'center': (cx, cy),
                'area': area,
                'circularity': circularity,
                'contour': cnt,
            })

        return dots

    # =================================================================
    # Step 5 — Grid Alignment & Cell Clustering
    # =================================================================

    def _cluster_into_cells(self, dots):
        """
        จัดจุดที่ตรวจพบเข้าเป็น Braille cells

        Algorithm (ปรับปรุงใหม่ — Gap-Based Splitting):
        1. หา dot_spacing จาก nearest-neighbor distances
        2. Cluster พิกัด Y → rows (จุดที่อยู่แนวเดียวกัน)
        3. Cluster พิกัด X → columns
        4. แยก columns เป็น cell groups ด้วย gap analysis
           (gap ระหว่าง cells > gap ภายใน cell)
        5. Group rows เป็น cell rows (กลุ่มละ ≤3)
        6. สำหรับแต่ละ (row_group, col_group) หา dots ที่ตกอยู่ภายใน
        """
        centers = np.array([d['center'] for d in dots], dtype=np.float64)

        # 1. ประมาณ within-cell dot spacing
        dot_spacing = self._estimate_dot_spacing(centers)
        if dot_spacing == 0:
            return []

        tol = dot_spacing * self.config.CLUSTER_TOLERANCE

        # 2. Cluster Y → rows, X → columns
        _row_labels, row_centers = self._cluster_1d(centers[:, 1], tol)
        _col_labels, col_centers = self._cluster_1d(centers[:, 0], tol)

        row_centers_sorted = sorted(row_centers)
        col_centers_sorted = sorted(col_centers)

        # 3. Group rows → cell rows (≤3 แถวที่ติดกัน)
        cell_row_groups = self._group_rows_into_cells(
            row_centers_sorted, dot_spacing
        )

        # 4. แยก columns เป็น cell groups ด้วย gap analysis
        cell_col_groups = self._split_columns_into_cells(
            col_centers_sorted, dot_spacing
        )

        # 5. สร้าง cells จากทุกคู่ (row_group × col_group)
        cells = []
        for rg in cell_row_groups:
            for cg in cell_col_groups:
                cell_dots = self._assign_dots_to_cell(
                    dots, rg, cg, dot_spacing
                )
                if cell_dots:
                    cx = np.mean(cg)
                    cy = np.mean(rg)
                    cells.append({
                        'dots': frozenset(cell_dots),
                        'center': (int(cx), int(cy)),
                        'x': float(cx),
                        'y': float(cy),
                    })

        # เรียงตาม text line (Y) ก่อน แล้วค่อย sort X ภายในแต่ละบรรทัด
        # จัดกลุ่ม cells ที่อยู่บรรทัดเดียวกัน (Y ใกล้กัน)
        if cells:
            cells_sorted = []
            line_tol = dot_spacing * 1.5
            cells_by_y = sorted(cells, key=lambda c: c['y'])
            current_line = [cells_by_y[0]]

            for c in cells_by_y[1:]:
                if abs(c['y'] - current_line[0]['y']) <= line_tol:
                    current_line.append(c)
                else:
                    # เรียง cells ในบรรทัดนี้จากซ้ายไปขวา
                    current_line.sort(key=lambda c: c['x'])
                    cells_sorted.extend(current_line)
                    current_line = [c]

            current_line.sort(key=lambda c: c['x'])
            cells_sorted.extend(current_line)
            cells = cells_sorted

        return cells

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------

    def _estimate_dot_spacing(self, centers):
        """
        หาระยะห่างภายใน cell (within-cell spacing) จาก
        nearest-neighbor distances — ใช้ percentile ต่ำ
        เพราะ NN distance ที่สั้นที่สุดจะเป็นระยะภายใน cell
        """
        n = len(centers)
        if n < 2:
            return 0

        nn_dists = []
        for i in range(n):
            min_d = float('inf')
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < min_d:
                        min_d = d
            nn_dists.append(min_d)

        # ใช้ 25th percentile แทน median → จับเฉพาะ within-cell spacing
        return float(np.percentile(nn_dists, 25))

    def _cluster_1d(self, values, tolerance):
        """
        จัดกลุ่มค่า 1 มิติที่อยู่ใกล้กัน (ภายใน tolerance)

        Returns: (labels, centers)
        """
        order = np.argsort(values)
        sorted_vals = values[order]

        labels = np.zeros(len(values), dtype=int)
        centers = []
        cluster_id = 0
        cluster_start = 0

        for i in range(1, len(sorted_vals)):
            if sorted_vals[i] - sorted_vals[i - 1] > tolerance:
                center = np.mean(sorted_vals[cluster_start:i])
                centers.append(center)
                for idx in order[cluster_start:i]:
                    labels[idx] = cluster_id
                cluster_id += 1
                cluster_start = i

        center = np.mean(sorted_vals[cluster_start:])
        centers.append(center)
        for idx in order[cluster_start:]:
            labels[idx] = cluster_id

        return labels, centers

    def _group_rows_into_cells(self, row_centers, dot_spacing):
        """
        แบ่ง row centers เป็นกลุ่มของ ≤3 rows ที่เป็น cell เดียวกัน
        (ภายใน cell จะห่างกัน ≈ dot_spacing,
         ระหว่าง cell lines จะห่างกันมากกว่า)
        """
        if not row_centers:
            return []

        # ทุก row ที่ห่างกัน ≤ 1.8 * dot_spacing ถือว่าอยู่ cell เดียวกัน
        groups = [[row_centers[0]]]
        for i in range(1, len(row_centers)):
            if row_centers[i] - groups[-1][-1] <= dot_spacing * 1.8:
                if len(groups[-1]) < 3:          # จำกัดไม่เกิน 3 rows / cell
                    groups[-1].append(row_centers[i])
                else:
                    groups.append([row_centers[i]])
            else:
                groups.append([row_centers[i]])
        return groups

    def _split_columns_into_cells(self, col_centers, dot_spacing):
        """
        แยก column centers ออกเป็น cell groups (≤2 columns ต่อ cell)
        โดยใช้ **gap analysis**:
          - gap ภายใน cell ≈ dot_spacing
          - gap ระหว่าง cells > dot_spacing * 1.5

        ถ้ามีแค่ 1-2 columns → ถือว่าเป็น cell เดียว
        """
        if len(col_centers) <= 2:
            return [col_centers]

        # คำนวณ gaps ระหว่าง column ที่เรียงแล้ว
        gaps = []
        for i in range(1, len(col_centers)):
            gaps.append(col_centers[i] - col_centers[i - 1])

        # หา threshold เพื่อแยก within-cell vs between-cell gaps
        # within-cell gap ≈ dot_spacing, between-cell gap >> dot_spacing
        # ใช้ 1.5 * dot_spacing เป็นเส้นแบ่ง
        split_threshold = dot_spacing * 1.5

        # แบ่ง columns ตรง gaps ที่ > threshold
        groups = [[col_centers[0]]]
        for i in range(len(gaps)):
            if gaps[i] > split_threshold:
                # gap ใหญ่ → เริ่ม cell ใหม่
                groups.append([col_centers[i + 1]])
            else:
                # gap เล็ก → ยังอยู่ cell เดิม
                if len(groups[-1]) < 2:       # จำกัดไม่เกิน 2 cols / cell
                    groups[-1].append(col_centers[i + 1])
                else:
                    groups.append([col_centers[i + 1]])

        return groups

    def _assign_dots_to_cell(self, dots, row_group, col_group, dot_spacing):
        """
        หาว่า dot ไหนตกอยู่ใน cell (row_group × col_group)
        แล้ว return set ของ dot numbers (1-6)

        Braille cell layout:
            left_col  right_col
              (1)       (4)      ← row 0 (top)
              (2)       (5)      ← row 1 (mid)
              (3)       (6)      ← row 2 (bottom)

        ใช้ absolute lattice fitting:
        - สร้าง grid 3 แถว × 2 คอลัมน์ จาก row_group/col_group + dot_spacing
        - จับ dot เข้า slot ที่ใกล้ที่สุดใน grid
        """
        margin = dot_spacing * 0.6
        rg = sorted(row_group)
        cg = sorted(col_group)

        # --- สร้าง expected row positions (3 แถว) ---
        if len(rg) >= 3:
            expected_rows = [rg[0], rg[1], rg[2]]
        elif len(rg) == 2:
            gap = rg[1] - rg[0]
            if gap > dot_spacing * 1.5:
                # เป็น row 0 และ row 2 (ข้าม row 1)
                expected_rows = [rg[0], (rg[0] + rg[1]) / 2.0, rg[1]]
            else:
                # เป็น 2 แถวติดกัน
                expected_rows = [rg[0], rg[1], rg[1] + dot_spacing]
        else:
            row_top = rg[0]
            expected_rows = [
                row_top,
                row_top + dot_spacing,
                row_top + dot_spacing * 2,
            ]

        # --- สร้าง expected column positions (2 คอลัมน์) ---
        # ใช้ตำแหน่งซ้ายสุดของ col_group เป็นฐาน
        col_left = cg[0]
        if len(cg) >= 2:
            # มี 2 คอลัมน์ → ใช้ค่าจริง
            expected_cols = [cg[0], cg[-1]]
        else:
            # มีแค่ 1 คอลัมน์ → สร้างทั้ง 2 ตำแหน่ง
            expected_cols = [
                col_left,                   # left column (dot 1,2,3)
                col_left + dot_spacing,     # right column (dot 4,5,6)
            ]

        # --- bounding box ครอบ expected grid ทั้งหมด ---
        y_min = expected_rows[0] - margin
        y_max = expected_rows[-1] + margin
        x_min = expected_cols[0] - margin
        x_max = expected_cols[-1] + margin

        cell_dots = set()

        for dot in dots:
            cx, cy = dot['center']

            if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
                continue

            # --- column index (left=0, right=1) ---
            # ใช้ midpoint ระหว่าง 2 expected columns เป็นเส้นแบ่ง
            col_mid = (expected_cols[0] + expected_cols[1]) / 2.0
            col_idx = 0 if cx < col_mid else 1

            # --- row index (0=top, 1=mid, 2=bottom) ---
            # หา row ที่ใกล้ที่สุดใน expected grid (absolute positions)
            row_dists = [abs(cy - er) for er in expected_rows]
            row_idx = int(np.argmin(row_dists))

            # ตรวจสอบว่า dot ไม่ห่างจาก expected position มากเกินไป
            if row_dists[row_idx] > dot_spacing * 0.7:
                continue

            # dot number: col 0 → 1,2,3  |  col 1 → 4,5,6
            dot_num = (row_idx + 1) + (col_idx * 3)
            if 1 <= dot_num <= 6:
                cell_dots.add(dot_num)

        return cell_dots

    # =================================================================
    # Visualization
    # =================================================================

    def _annotate(self, image, dots, cells):
        """วาด annotation ลงบนภาพสำหรับ debug"""
        # วาดวงกลมรอบจุดที่ตรวจพบ
        for dot in dots:
            cx, cy = dot['center']
            radius = int(np.sqrt(dot['area'] / np.pi))
            cv2.circle(image, (cx, cy), radius + 4, (0, 255, 0), 2)
            cv2.circle(image, (cx, cy), 2, (0, 0, 255), -1)

        # วาด label ของแต่ละ cell
        from config import BRAILLE_TO_CHAR

        for cell in cells:
            cx, cy = cell['center']
            char = BRAILLE_TO_CHAR.get(cell['dots'], '?')
            dots_str = ','.join(map(str, sorted(cell['dots'])))

            label = f"{char} [{dots_str}]"
            cv2.putText(
                image, label,
                (cx - 20, cy - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2,
            )

        return image
