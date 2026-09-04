"""
Braille Reader - Dot Detector
===============================
ตรวจจับจุดสีบนภาพอักษรเบรลล์ OpenCV

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

    
    # 1 — Preprocessing

    def _preprocess(self, image):
        """Gaussian blur เพื่อลด noise"""
        return cv2.GaussianBlur(image, (5, 5), 0)

    # 2 — Color Segmentation

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
            # จุดสีดำ / กราฟิกเอกสาร — ใช้เทคนิคหลายระดับเพื่อรองรับพื้นหลังหลากหลาย
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
            #    เพื่อเพิ่มคอนทราสต์เฉพาะจุด ช่วยให้จุดดำโดดเด่นจากพื้นม่วง/เทา/ครีม
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)

            # 2. Otsu's Thresholding — หาค่า threshold อัตโนมัติจาก histogram
            #    ดีกว่าค่า fixed เพราะปรับตามพื้นหลังจริงของแต่ละภาพ
            _, mask_otsu = cv2.threshold(
                gray_eq, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )

            # 3. Fixed threshold fallback (สำหรับภาพที่ Otsu ไม่ดี)
            _, mask_fixed = cv2.threshold(
                gray, self.config.BLACK_INTENSITY_MAX, 255, cv2.THRESH_BINARY_INV
            )

            # 4. รวม mask ทั้งสอง — ใช้ intersection (AND) เพื่อลด false positive
            #    จุดที่ "ดำจริงๆ" จะผ่านทั้ง 2 threshold
            mask = cv2.bitwise_and(mask_otsu, mask_fixed)

            # 5. หากผลลัพธ์น้อยเกินไป (Otsu aggressive เกินไป) ให้ใช้ Otsu อย่างเดียว
            if cv2.countNonZero(mask) < cv2.countNonZero(mask_otsu) * 0.3:
                mask = mask_otsu
        else:
            # Bug 5 fix: สีที่ไม่รู้จัก → return empty mask แทน crash
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        return mask

    # Step 3 — Morphological Cleanup

    def _morph_clean(self, mask):
        """ปิดรูเล็ก + ลบ noise ด้วย morphology"""
        k = self.config.MORPH_KERNEL_SIZE
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # Close: ปิดรูเล็กๆ ภายในจุด
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Open: ลบ noise เล็กๆ
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    # Step 4 — Dot Detection

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
    # Step 5 — Grid Alignment & Cell Clustering (Line-Adaptive Robust Clustering)
    # =================================================================

    def _estimate_line_tilt(self, centers, dot_spacing):
        """
        ประมาณค่ามุมเอียงของบรรทัดอักษรเบรลล์ (ในหน่วย radian) จากคู่จุดข้างเคียงในกริด
        จุดที่อยู่ติดกันในระยะ [0.7 * d .. 1.45 * d] จะอยู่ในแนวเดียวกัน (แนวนอนหรือแนวตั้ง)
        """
        n = len(centers)
        if n < 4 or dot_spacing <= 0:
            return 0.0

        angles = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                d = np.hypot(dx, dy)
                if 0.7 * dot_spacing <= d <= 1.45 * dot_spacing:
                    if abs(dx) >= abs(dy):
                        # Horizontal neighbor pair
                        a = np.arctan2(dy, dx)
                        if a > np.pi / 2:
                            a -= np.pi
                        elif a < -np.pi / 2:
                            a += np.pi
                        angles.append(a)
                    else:
                        # Vertical neighbor pair
                        a = np.arctan2(-dx, dy)
                        if a > np.pi / 2:
                            a -= np.pi
                        elif a < -np.pi / 2:
                            a += np.pi
                        angles.append(a)

        if not angles:
            return 0.0
        return float(np.median(angles))

    def _cluster_into_cells(self, dots):
        """
        จัดจุดที่ตรวจพบเข้าเป็น Braille cells ด้วย Line-Adaptive Robust Clustering
        รองรับประโยคยาว อักษรเบรลล์ขนาดเล็ก และชดเชยการเอียงของกล้อง (Deskew)

        Algorithm:
        1. หา within-cell spacing (d) จาก nearest-neighbor distances
        2. ประมาณมุมเอียงของบรรทัด (theta) จากคู่จุดข้างเคียง แล้วหมุนพิกัดเข้าสู่แนวระนาบ
        3. แบ่งกลุ่มจุดตามแกน Y ออกเป็นบรรทัดการอ่าน (Reading Lines)
        4. สำหรับแต่ละบรรทัด:
           - กำหนด baseline rows (row 0, 1, 2) ประจำบรรทัด (ป้องกันเซลล์ 1-2 จุดถูกเลื่อนแถวผิด)
           - เรียงจุดตามแกน X และแยกเซลล์ด้วย Gap threshold > 1.25 * d
           - จัดจุดเข้า Slot 1-6 ในกริด 2x3
        5. แปลงพิกัดกรอบ (Bounding Box) และ Slots กลับสู่มุมกล้องเดิมสำหรับแสดงผล
        """
        if not dots:
            return []

        centers = np.array([d['center'] for d in dots], dtype=np.float64)
        n_dots = len(centers)
        if n_dots == 0:
            return []

        # 1. ประมาณ within-cell dot spacing
        dot_spacing = self._estimate_dot_spacing(centers)
        if dot_spacing <= 0:
            return []

        # 2. ประมาณมุมเอียงและหมุนพิกัดเข้าสู่แนวระนาบ (Deskew)
        tilt_angle = self._estimate_line_tilt(centers, dot_spacing)
        cx_mean = float(np.mean(centers[:, 0]))
        cy_mean = float(np.mean(centers[:, 1]))
        cos_a = np.cos(-tilt_angle)
        sin_a = np.sin(-tilt_angle)

        dx = centers[:, 0] - cx_mean
        dy = centers[:, 1] - cy_mean
        rot_x = dx * cos_a - dy * sin_a + cx_mean
        rot_y = dx * sin_a + dy * cos_a + cy_mean

        # 3. จัดกลุ่มบรรทัดการอ่าน (Reading Lines) ตามแกน Y
        # ความสูงเซลล์ ~ 2 * dot_spacing, ระยะห่างระหว่างบรรทัด >= 2.2 * dot_spacing
        order_y = np.argsort(rot_y)
        line_gap_thresh = dot_spacing * 2.2

        reading_lines = []
        curr_line = [order_y[0]]
        for idx in order_y[1:]:
            if rot_y[idx] - rot_y[curr_line[-1]] > line_gap_thresh:
                reading_lines.append(curr_line)
                curr_line = [idx]
            else:
                curr_line.append(idx)
        reading_lines.append(curr_line)

        # เรียงบรรทัดจากบนลงล่าง
        reading_lines.sort(key=lambda line: np.median(rot_y[line]))

        cells = []
        cos_b = np.cos(tilt_angle)
        sin_b = np.sin(tilt_angle)

        # 4. ประมวลผลแต่ละบรรทัดการอ่าน
        for line_indices in reading_lines:
            line_indices = np.array(line_indices)
            l_rot_x = rot_x[line_indices]
            l_rot_y = rot_y[line_indices]

            # 4a. กำหนดตำแหน่งแถว 0, 1, 2 อ้างอิงของบรรทัดนี้
            y_min = float(np.min(l_rot_y))
            y_max = float(np.max(l_rot_y))
            y_span = y_max - y_min

            if y_span >= dot_spacing * 1.4:
                # มีจุดครอบคลุมทั้งแถวบนและแถวล่าง
                top_mask = l_rot_y < y_min + dot_spacing * 0.55
                bot_mask = l_rot_y > y_max - dot_spacing * 0.55
                mid_mask = (~top_mask) & (~bot_mask)

                r_top = float(np.median(l_rot_y[top_mask])) if np.any(top_mask) else (y_max - 2 * dot_spacing)
                r_bot = float(np.median(l_rot_y[bot_mask])) if np.any(bot_mask) else (y_min + 2 * dot_spacing)
                r_mid = float(np.median(l_rot_y[mid_mask])) if np.any(mid_mask) else ((r_top + r_bot) / 2.0)
                expected_rows = [r_top, r_mid, r_bot]
            elif y_span >= dot_spacing * 0.6:
                # มี 2 แถว
                r0 = y_min
                r1 = y_max
                if (r1 - r0) > dot_spacing * 1.5:
                    expected_rows = [r0, (r0 + r1) / 2.0, r1]
                else:
                    expected_rows = [r0, r1, r1 + dot_spacing]
            else:
                # มีเพียง 1 แถว
                r_mid = float(np.median(l_rot_y))
                expected_rows = [r_mid - dot_spacing, r_mid, r_mid + dot_spacing]

            # 4b. แบ่งกลุ่มจุดตามแกน X เป็นเซลล์ (Split เมื่อ gap > 1.25 * dot_spacing)
            order_x = np.argsort(l_rot_x)
            cell_split_thresh = dot_spacing * 1.25

            cell_clusters = []
            curr_cluster = [order_x[0]]
            for i in range(1, len(order_x)):
                idx_curr = order_x[i]
                idx_prev = curr_cluster[-1]
                if l_rot_x[idx_curr] - l_rot_x[idx_prev] > cell_split_thresh:
                    cell_clusters.append(curr_cluster)
                    curr_cluster = [idx_curr]
                else:
                    curr_cluster.append(idx_curr)
            cell_clusters.append(curr_cluster)

            # 4c. กำหนดคอลัมน์ซ้าย-ขวา และจัดจุดเข้า 2x3 Grid
            prev_cell_right = None
            for cluster in cell_clusters:
                cluster_dot_indices = line_indices[cluster]
                c_xs = rot_x[cluster_dot_indices]
                c_ys = rot_y[cluster_dot_indices]

                # กำหนดคอลัมน์ (มี 2 คอลัมน์ หรือ 1 คอลัมน์)
                if np.ptp(c_xs) > dot_spacing * 0.55:
                    c_mid = (np.min(c_xs) + np.max(c_xs)) / 2.0
                    col_0 = float(np.mean(c_xs[c_xs < c_mid]))
                    col_1 = float(np.mean(c_xs[c_xs >= c_mid]))
                else:
                    c_val = float(np.mean(c_xs))
                    cell_gap = dot_spacing * 1.375
                    pitch = dot_spacing + cell_gap
                    is_right = False
                    if prev_cell_right is not None:
                        dist = c_val - prev_cell_right
                        offset = dist - cell_gap
                        rem = (offset + pitch / 2.0) % pitch - (pitch / 2.0)
                        if abs(rem - dot_spacing) < abs(rem):
                            is_right = True
                    if is_right:
                        col_0 = c_val - dot_spacing
                        col_1 = c_val
                    else:
                        col_0 = c_val
                        col_1 = c_val + dot_spacing

                prev_cell_right = col_1
                expected_cols = [col_0, col_1]

                # กำหนดจุด 1-6
                cell_dots = set()
                for d_idx in cluster_dot_indices:
                    px = rot_x[d_idx]
                    py = rot_y[d_idx]

                    # Row index (0=top, 1=mid, 2=bot)
                    r_dists = [abs(py - r) for r in expected_rows]
                    r_idx = int(np.argmin(r_dists))

                    # Col index (0=left, 1=right)
                    c_idx = 0 if abs(px - col_0) < abs(px - col_1) else 1

                    dot_num = c_idx * 3 + (r_idx + 1)
                    cell_dots.add(dot_num)

                # คำนวณพิกัด Slots ในภาพต้นฉบับ (Back-rotate)
                rot_slots = {
                    1: (col_0, expected_rows[0]),
                    2: (col_0, expected_rows[1]),
                    3: (col_0, expected_rows[2]),
                    4: (col_1, expected_rows[0]),
                    5: (col_1, expected_rows[1]),
                    6: (col_1, expected_rows[2]),
                }

                slots = {}
                for k, (sx, sy) in rot_slots.items():
                    sdx = sx - cx_mean
                    sdy = sy - cy_mean
                    orig_sx = sdx * cos_b - sdy * sin_b + cx_mean
                    orig_sy = sdx * sin_b + sdy * cos_b + cy_mean
                    slots[k] = (orig_sx, orig_sy)

                # Center ในภาพต้นฉบับ
                cen_rx = (col_0 + col_1) / 2.0
                cen_ry = (expected_rows[0] + expected_rows[2]) / 2.0
                cdx = cen_rx - cx_mean
                cdy = cen_ry - cy_mean
                orig_cx = cdx * cos_b - cdy * sin_b + cx_mean
                orig_cy = cdx * sin_b + cdy * cos_b + cy_mean

                margin = dot_spacing * 0.45
                x_min = min(s[0] for s in slots.values()) - margin
                x_max = max(s[0] for s in slots.values()) + margin
                y_min_b = min(s[1] for s in slots.values()) - margin
                y_max_b = max(s[1] for s in slots.values()) + margin

                grid_info = {
                    'expected_cols': expected_cols,
                    'expected_rows': expected_rows,
                    'bbox': (int(x_min), int(y_min_b), int(x_max), int(y_max_b)),
                    'slots': slots,
                }

                cells.append({
                    'dots': frozenset(cell_dots),
                    'center': (int(orig_cx), int(orig_cy)),
                    'x': float(orig_cx),
                    'y': float(orig_cy),
                    'grid': grid_info,
                })

        return cells

    # -----------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------

    def _estimate_dot_spacing(self, centers):
        """
        หาระยะห่างภายใน cell (within-cell spacing) จาก
        nearest-neighbor distances — Perf 3: O(n log n) via KDTree
        """
        n = len(centers)
        if n < 2:
            return 0

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(centers)
            # query k=2: ตัวเอง (dist=0) + nearest neighbor
            dists, _ = tree.query(centers, k=2)
            nn_dists = dists[:, 1]  # column 1 = nearest neighbor distance
        except ImportError:
            # Fallback: sort-based approximation (still faster than O(n²) brute force)
            # Project onto X and Y separately, sort, then find nearest gap
            nn_dists = np.full(n, float('inf'))
            for axis in range(2):
                order = np.argsort(centers[:, axis])
                sorted_vals = centers[order]
                for k in range(n - 1):
                    d = np.linalg.norm(sorted_vals[k + 1] - sorted_vals[k])
                    orig_i = order[k]
                    orig_j = order[k + 1]
                    if d < nn_dists[orig_i]:
                        nn_dists[orig_i] = d
                    if d < nn_dists[orig_j]:
                        nn_dists[orig_j] = d

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
        # within-cell gap ≈ 1.0 * dot_spacing (40px)
        # between-cell gap ≥ 1.35 * dot_spacing (55px+)
        # ใช้ 1.18 * dot_spacing เป็นเส้นแบ่ง
        split_threshold = dot_spacing * 1.18

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

    def _assign_dots_to_cell(self, dots, row_group, col_group, dot_spacing, prev_col=None, next_col=None, prev_grid=None):
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
        - ตัดสินใจว่า single column เป็นคอลัมน์ซ้ายหรือขวาจากระยะห่างรอบข้าง
        - จับ dot เข้า slot ที่ใกล้ที่สุดใน grid
        """
        margin = dot_spacing * 0.45
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
        if len(cg) >= 2:
            # มี 2 คอลัมน์ → ใช้ค่าจริง
            expected_cols = [cg[0], cg[-1]]
        else:
            # มีแค่ 1 คอลัมน์ → ตรวจสอบว่าเป็น Left column หรือ Right column
            c = cg[0]
            is_right_col = False
            cell_gap = dot_spacing * 1.375
            pitch = dot_spacing + cell_gap  # ระยะห่างระหว่างเซลล์ ≈ 2.375 * dot_spacing

            if prev_grid is not None:
                prev_right = prev_grid['expected_cols'][1]
                dist = c - prev_right
                offset = dist - cell_gap
                # คำนวณ signed remainder ในช่วง [-pitch/2, pitch/2]
                rem = (offset + pitch / 2.0) % pitch - (pitch / 2.0)
                if abs(rem - dot_spacing) < abs(rem):
                    is_right_col = True
                else:
                    is_right_col = False
            elif prev_col is not None:
                dist = c - prev_col
                offset = dist - cell_gap
                rem = (offset + pitch / 2.0) % pitch - (pitch / 2.0)
                if abs(rem - dot_spacing) < abs(rem):
                    is_right_col = True
                else:
                    is_right_col = False
            elif next_col is not None:
                dist = next_col - c
                rem = (dist + pitch / 2.0) % pitch - (pitch / 2.0)
                if abs(rem - cell_gap) < abs(rem):
                    is_right_col = True
                else:
                    is_right_col = False

            if is_right_col:
                expected_cols = [c - dot_spacing, c]
            else:
                expected_cols = [c, c + dot_spacing]

        # --- bounding box ครอบ expected grid ทั้งหมด ---
        y_min = expected_rows[0] - margin
        y_max = expected_rows[-1] + margin
        x_min = expected_cols[0] - margin
        x_max = expected_cols[-1] + margin

        # พิกัดของทั้ง 6 ช่องใน Grid 2x3
        slots = {
            1: (expected_cols[0], expected_rows[0]),
            2: (expected_cols[0], expected_rows[1]),
            3: (expected_cols[0], expected_rows[2]),
            4: (expected_cols[1], expected_rows[0]),
            5: (expected_cols[1], expected_rows[1]),
            6: (expected_cols[1], expected_rows[2]),
        }

        grid_info = {
            'expected_cols': expected_cols,
            'expected_rows': expected_rows,
            'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
            'slots': slots,
        }

        cell_dots = set()

        for dot in dots:
            cx, cy = dot['center']

            if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
                continue

            # --- column index (left=0, right=1) ---
            col_mid = (expected_cols[0] + expected_cols[1]) / 2.0
            col_idx = 0 if cx < col_mid else 1

            # --- row index (0=top, 1=mid, 2=bottom) ---
            row_dists = [abs(cy - er) for er in expected_rows]
            row_idx = int(np.argmin(row_dists))

            if row_dists[row_idx] > dot_spacing * 0.7:
                continue

            # dot number: col 0 → 1,2,3  |  col 1 → 4,5,6
            dot_num = (row_idx + 1) + (col_idx * 3)
            if 1 <= dot_num <= 6:
                cell_dots.add(dot_num)

        return cell_dots, grid_info

    # =================================================================
    # Visualization (2x3 Grid Overlay & Thai/English Text Banner)
    # =================================================================

    # Perf 4: Font cache — avoid repeated disk I/O per frame
    _font_cache = {}

    def _get_font(self, size=20, bold=False):
        """โหลด TrueType font ที่รองรับภาษาไทย (cached)"""
        import os
        from PIL import ImageFont

        cache_key = (size, bold)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font_candidates = [
            'C:/Windows/Fonts/leelawdb.ttf' if bold else 'C:/Windows/Fonts/leelawad.ttf',
            'C:/Windows/Fonts/tahomabd.ttf' if bold else 'C:/Windows/Fonts/tahoma.ttf',
            'C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',
            '/usr/share/fonts/truetype/thai/Loma-Bold.ttf' if bold else '/usr/share/fonts/truetype/thai/Loma.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
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

    def _annotate(self, image, dots, cells, decoded_text="", verbose_results=None, lang="english"):
        """วาด 2x3 Grid Overlay พร้อมแบนเนอร์แสดงคำที่อ่านได้ลงบนภาพ"""
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw

        h, w = image.shape[:2]
        banner_h = 80  # ความสูงของแถบข้อความด้านล่าง

        # สร้าง canvas เพิ่มพื้นที่ด้านล่างสำหรับแถบผลลัพธ์
        canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
        canvas[:h, :w] = image.copy()
        canvas[h:, :] = (30, 23, 15)  # Dark slate navy background (BGR)

        # 1. วาดเส้น 2x3 Grid รอบแต่ละ Cell
        for idx, cell in enumerate(cells, 1):
            grid = cell.get('grid')
            if not grid:
                continue

            x_min, y_min, x_max, y_max = grid['bbox']
            slots = grid.get('slots', {})

            if slots and 1 in slots and 4 in slots and 2 in slots and 3 in slots:
                col_mid = int((slots[1][0] + slots[4][0]) / 2.0)
                row_mid1 = int((slots[1][1] + slots[2][1]) / 2.0)
                row_mid2 = int((slots[2][1] + slots[3][1]) / 2.0)
            else:
                cols = grid.get('expected_cols', [x_min, x_max])
                rows = grid.get('expected_rows', [y_min, (y_min + y_max) / 2.0, y_max])
                col_mid = int((cols[0] + cols[1]) / 2.0)
                row_mid1 = int((rows[0] + rows[1]) / 2.0)
                row_mid2 = int((rows[1] + rows[2]) / 2.0)

            # กรอบนอกของ Cell (สีฟ้า Cyan)
            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), (255, 200, 0), 2)

            # เส้นแบ่งคอลัมน์แนวตั้ง
            cv2.line(canvas, (col_mid, y_min), (col_mid, y_max), (200, 160, 0), 1)

            # เส้นแบ่งแถวแนวนอน
            cv2.line(canvas, (x_min, row_mid1), (x_max, row_mid1), (200, 160, 0), 1)
            cv2.line(canvas, (x_min, row_mid2), (x_max, row_mid2), (200, 160, 0), 1)

            # วาดสัญลักษณ์ช่องว่าง (Empty Slot Circles)
            for dot_id, (sx, sy) in grid['slots'].items():
                if dot_id not in cell['dots']:
                    cv2.circle(canvas, (int(sx), int(sy)), 5, (180, 180, 180), 1)

            # วาดเส้นใต้ขอบ Cell
            dots_str = ','.join(map(str, sorted(cell['dots'])))
            y_pos = y_max + 18
            label = f"[{dots_str}]"
            cv2.putText(
                canvas, label,
                (int(cell['center'][0]) - 16, int(y_pos)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 0, 220), 1, cv2.LINE_AA
            )

        # 2. วาดไฮไลท์รอบจุดสีที่ตรวจจับได้ (สีเขียว Bright Green)
        for dot in dots:
            cx, cy = dot['center']
            radius = int(np.sqrt(dot['area'] / np.pi))
            cv2.circle(canvas, (int(cx), int(cy)), radius + 3, (0, 255, 0), 2)
            cv2.circle(canvas, (int(cx), int(cy)), 2, (0, 0, 255), -1)

        # 3. ใช้ PIL เพื่อวาดข้อความภาษาไทย/อังกฤษที่คมชัด
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        font_large = self._get_font(size=24, bold=True)
        font_mid = self._get_font(size=18, bold=True)
        font_small = self._get_font(size=14, bold=False)

        # วาดตัวอักษรของแต่ละเซลล์เหนือกล่อง Grid
        if verbose_results:
            for idx, item in enumerate(verbose_results, 1):
                if idx - 1 < len(cells):
                    cell = cells[idx - 1]
                    grid = cell.get('grid')
                    if grid:
                        x_min, y_min = grid['bbox'][0], grid['bbox'][1]
                        char_text = f"C{idx}: {item['char']}"
                        draw.text((x_min + 2, y_min - 26), char_text, fill=(255, 180, 0), font=font_mid)
        else:
            for idx, cell in enumerate(cells, 1):
                grid = cell.get('grid')
                if grid:
                    x_min, y_min = grid['bbox'][0], grid['bbox'][1]
                    draw.text((x_min + 2, y_min - 22), f"C{idx}", fill=(255, 180, 0), font=font_small)

        # 4. วาดแถบข้อความสรุปผลลัพธ์ด้านล่าง (Bottom Banner)
        # เส้นแบ่งแถบด้านล่าง
        draw.line([(0, h), (w, h)], fill=(70, 85, 105), width=2)

        if decoded_text:
            text_disp = f"ข้อความ: \"{decoded_text}\"" if lang == 'thai' else f"Text: \"{decoded_text}\""
            braille_chars = [item.get('unicode', '·') for item in (verbose_results or [])]
            braille_disp = ' '.join(braille_chars)

            # หัวข้อผลลัพธ์ตัวใหญ่สีเหลืองทอง/ขาว
            draw.text((20, h + 12), text_disp, fill=(255, 255, 100), font=font_large)

            # คำบรรยายรายละเอียดภาษาและจำนวนเซลล์
            lang_label = "ภาษาไทย (Thai)" if lang == 'thai' else "ภาษาอังกฤษ (English)"
            sub_text = f"Braille: {braille_disp}   |   {len(cells)} Cells ({len(dots)} Dots)   |   {lang_label}"
            draw.text((20, h + 48), sub_text, fill=(180, 200, 220), font=font_small)
        else:
            draw.text((20, h + 25), "Braille Detection Running...", fill=(180, 200, 220), font=font_mid)

        # แปลงกลับเป็น OpenCV BGR
        annotated_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return annotated_result

    def annotate_with_text(self, image, dots, cells, decoded_text="", verbose_results=None, lang="english"):
        """Public API สำหรับวาด annotation พร้อมข้อความผลลัพธ์"""
        return self._annotate(image, dots, cells, decoded_text=decoded_text, verbose_results=verbose_results, lang=lang)
