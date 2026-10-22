# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.abspath('.'))
import cv2
import numpy as np

def estimate_line_tilt(centers, dot_spacing):
    """
    Estimate text line tilt angle (in radians) from adjacent dot pairs in grid.
    Dots within distance [0.7 * d .. 1.4 * d] are either horizontal or vertical neighbors.
    """
    n = len(centers)
    if n < 4:
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
                    if a > np.pi / 2: a -= np.pi
                    elif a < -np.pi / 2: a += np.pi
                    angles.append(a)
                else:
                    # Vertical neighbor pair
                    a = np.arctan2(-dx, dy)
                    if a > np.pi / 2: a -= np.pi
                    elif a < -np.pi / 2: a += np.pi
                    angles.append(a)

    if not angles:
        return 0.0
    return float(np.median(angles))

def cluster_into_cells_robust(dots, dot_spacing, cluster_tolerance=0.5):
    """
    Robust Braille cell clustering with:
    1. Dot-pair line tilt estimation and rotation (deskew).
    2. Multi-line Y clustering into reading lines.
    3. Within-line X-gap splitting into cells.
    4. Line-anchored row positioning (preventing single-dot or row-1/2 cells from misaligning).
    5. Correct original-coordinate back-projection for UI overlays.
    """
    if not dots:
        return []

    centers = np.array([d['center'] for d in dots], dtype=np.float64)
    n_dots = len(centers)
    if n_dots == 0 or dot_spacing <= 0:
        return []

    # 1. Estimate line tilt angle
    tilt_angle = estimate_line_tilt(centers, dot_spacing)

    # 2. Rotate coordinates to horizontal frame
    cx_mean = float(np.mean(centers[:, 0]))
    cy_mean = float(np.mean(centers[:, 1]))
    cos_a = np.cos(-tilt_angle)
    sin_a = np.sin(-tilt_angle)

    dx = centers[:, 0] - cx_mean
    dy = centers[:, 1] - cy_mean
    rot_x = dx * cos_a - dy * sin_a + cx_mean
    rot_y = dx * sin_a + dy * cos_a + cy_mean

    # 3. Partition into reading lines along Y
    # Line height is ~ 2 * dot_spacing. Gap between lines is >= 2.5 * dot_spacing
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

    # Sort reading lines top to bottom by median Y
    reading_lines.sort(key=lambda line: np.median(rot_y[line]))

    cells = []

    # 4. Process each reading line
    for line_indices in reading_lines:
        line_indices = np.array(line_indices)
        l_rot_x = rot_x[line_indices]
        l_rot_y = rot_y[line_indices]

        # 4a. Determine expected rows (row 0, 1, 2) for this reading line
        y_min = float(np.min(l_rot_y))
        y_max = float(np.max(l_rot_y))
        y_span = y_max - y_min

        if y_span >= dot_spacing * 1.4:
            # Full 3-row span
            top_mask = l_rot_y < y_min + dot_spacing * 0.55
            bot_mask = l_rot_y > y_max - dot_spacing * 0.55
            mid_mask = (~top_mask) & (~bot_mask)

            r_top = float(np.median(l_rot_y[top_mask])) if np.any(top_mask) else (y_max - 2 * dot_spacing)
            r_bot = float(np.median(l_rot_y[bot_mask])) if np.any(bot_mask) else (y_min + 2 * dot_spacing)
            r_mid = float(np.median(l_rot_y[mid_mask])) if np.any(mid_mask) else ((r_top + r_bot) / 2.0)
            expected_rows = [r_top, r_mid, r_bot]
        elif y_span >= dot_spacing * 0.6:
            # 2 rows present
            r0 = y_min
            r1 = y_max
            # Check gap
            if (r1 - r0) > dot_spacing * 1.5:
                # Row 0 and row 2 (missing mid)
                expected_rows = [r0, (r0 + r1) / 2.0, r1]
            else:
                # 2 adjacent rows. Default to top & mid, unless dots are closer to bottom
                expected_rows = [r0, r1, r1 + dot_spacing]
        else:
            # Only 1 row of dots in entire line
            r_mid = float(np.median(l_rot_y))
            expected_rows = [r_mid - dot_spacing, r_mid, r_mid + dot_spacing]

        # 4b. Group dots into cells along X (split by gap > 1.25 * dot_spacing)
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

        # 4c. Assign dots and construct 2x3 grid for each cell
        prev_cell_right = None
        for cluster in cell_clusters:
            cluster_dot_indices = line_indices[cluster]
            c_xs = rot_x[cluster_dot_indices]
            c_ys = rot_y[cluster_dot_indices]

            # Determine columns (2 cols vs 1 col)
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

            # Assign dots to 2x3 grid
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

            # Compute bounding box and slots in original (unrotated) space
            # Slot coordinates in rotated space
            rot_slots = {
                1: (col_0, expected_rows[0]),
                2: (col_0, expected_rows[1]),
                3: (col_0, expected_rows[2]),
                4: (col_1, expected_rows[0]),
                5: (col_1, expected_rows[1]),
                6: (col_1, expected_rows[2]),
            }

            # Back-rotate slots to original image space
            cos_b = np.cos(tilt_angle)
            sin_b = np.sin(tilt_angle)
            slots = {}
            for k, (sx, sy) in rot_slots.items():
                sdx = sx - cx_mean
                sdy = sy - cy_mean
                orig_sx = sdx * cos_b - sdy * sin_b + cx_mean
                orig_sy = sdx * sin_b + sdy * cos_b + cy_mean
                slots[k] = (orig_sx, orig_sy)

            # Center in original space
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

if __name__ == '__main__':
    from test_accuracy import TEST_DATASET
    from decoder import decode_cells
    from detector import BrailleDetector

    print('=' * 70)
    print('Testing new clustering algorithm on accuracy benchmark:')
    print('=' * 70)

    total = len(TEST_DATASET)
    passed = 0
    for idx, (img_path, color, lang, expected) in enumerate(TEST_DATASET, 1):
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        det = BrailleDetector(dot_color=color)
        _, debug_info = det.detect(img)
        dots = debug_info['dots']
        spacing = det._estimate_dot_spacing(np.array([d['center'] for d in dots]))
        cells = cluster_into_cells_robust(dots, spacing)
        actual = decode_cells(cells, lang=lang)
        is_pass = (actual == expected)
        if is_pass:
            passed += 1
            print(f'[{idx:2d}/{total}] {expected:<12} -> {actual:<12} OK')
        else:
            print(f'[{idx:2d}/{total}] {expected:<12} -> {actual:<12} FAIL')

    print(f'\nResult: {passed}/{total} PASSED ({passed/total*100:.1f}%)')
