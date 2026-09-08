import sys, os
sys.path.insert(0, os.path.abspath('.'))
import cv2
import numpy as np
from yolo_detector import YOLOBrailleDetector
from decoder import decode_cells_thai, decode_cells_verbose

img = cv2.imread('C:/Users/bavon/.gemini/antigravity-ide/brain/9d3e5e78-a361-42e7-90ce-ee53cf6cd319/.user_uploaded/media_1788542807961.png')
h, w = img.shape[:2]
cam_img = img[50:h-90, :]
detector = YOLOBrailleDetector(mode='yolo')
cells, debug = detector.detect(cam_img)
dots = debug['dots']
centers = np.array([d['center'] for d in dots])
xs = centers[:, 0]
ys = centers[:, 1]

# 1. Estimate spacing
dot_spacing = detector._opencv_detector._estimate_dot_spacing(centers)
print(f'Estimated dot_spacing: {dot_spacing:.2f}')

# 2. Estimate line orientation (tilt angle)
p = np.polyfit(xs, ys, 1)
angle_rad = np.arctan(p[0])
print(f'Line angle: {np.degrees(angle_rad):.3f} deg')

# 3. Rotate coordinates to horizontal
cos_a = np.cos(-angle_rad)
sin_a = np.sin(-angle_rad)
cx_mean, cy_mean = np.mean(xs), np.mean(ys)
dx = xs - cx_mean
dy = ys - cy_mean
rot_x = dx * cos_a - dy * sin_a + cx_mean
rot_y = dx * sin_a + dy * cos_a + cy_mean

# 4. Find line baseline rows in pure NumPy
y_min = float(np.min(rot_y))
y_max = float(np.max(rot_y))
span_y = y_max - y_min

if span_y >= dot_spacing * 1.5:
    # Both top and bottom rows present
    top_dots = rot_y[rot_y < y_min + dot_spacing * 0.5]
    bot_dots = rot_y[rot_y > y_max - dot_spacing * 0.5]
    mid_dots = rot_y[(rot_y >= y_min + dot_spacing * 0.5) & (rot_y <= y_max - dot_spacing * 0.5)]
    
    r_top = float(np.median(top_dots)) if len(top_dots) > 0 else (y_max - 2 * dot_spacing)
    r_bot = float(np.median(bot_dots)) if len(bot_dots) > 0 else (y_min + 2 * dot_spacing)
    r_mid = float(np.median(mid_dots)) if len(mid_dots) > 0 else ((r_top + r_bot) / 2.0)
    expected_rows = [r_top, r_mid, r_bot]
else:
    r_mid = float(np.median(rot_y))
    expected_rows = [r_mid - dot_spacing, r_mid, r_mid + dot_spacing]

print('Line expected rows (pure NumPy):', [round(r, 1) for r in expected_rows])

# 5. Sort dots by rot_x and group into cells
order = np.argsort(rot_x)
cell_groups = []
curr_group = [order[0]]
split_thresh = dot_spacing * 1.35
for idx in order[1:]:
    if rot_x[idx] - rot_x[curr_group[-1]] > split_thresh:
        cell_groups.append(curr_group)
        curr_group = [idx]
    else:
        curr_group.append(idx)
cell_groups.append(curr_group)

print(f'Grouped into {len(cell_groups)} cells along X.')

# 6. Assign dots to 2x3 grid in each cell
detected_cells = []
prev_cell_right = None

for c_i, group in enumerate(cell_groups):
    g_xs = rot_x[group]
    g_ys = rot_y[group]
    
    # Determine columns
    if np.ptp(g_xs) > dot_spacing * 0.55:
        # Two columns present
        c_mid = (g_xs.min() + g_xs.max()) / 2.0
        col_0 = np.mean(g_xs[g_xs < c_mid])
        col_1 = np.mean(g_xs[g_xs >= c_mid])
    else:
        # Single column present: check pitch from prev cell or lattice
        c_val = np.mean(g_xs)
        cell_gap = dot_spacing * 1.4
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
    
    cell_dots = set()
    for d_idx in group:
        dx_val = rot_x[d_idx]
        dy_val = rot_y[d_idx]
        
        # Row assignment
        r_dists = [abs(dy_val - r) for r in expected_rows]
        r_idx = int(np.argmin(r_dists))
        
        # Col assignment
        c_idx = 0 if abs(dx_val - col_0) < abs(dx_val - col_1) else 1
        
        dot_num = c_idx * 3 + (r_idx + 1)
        cell_dots.add(dot_num)
        
    detected_cells.append({
        'dots': frozenset(cell_dots),
        'center': (int(np.mean(centers[group, 0])), int(np.mean(centers[group, 1]))),
        'x': float(np.mean(centers[group, 0])),
        'y': float(np.mean(centers[group, 1])),
    })

print('\nDecoded cells summary:')
for i, c in enumerate(detected_cells):
    print(f'  Cell {i:2d}: dots={sorted(c["dots"])}')

text = decode_cells_thai(detected_cells)
print(f'\n===> Decoded Text: "{text}"')
