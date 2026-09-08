# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import cv2
import numpy as np
from detector import BrailleDetector
from yolo_detector import YOLOBrailleDetector

img_path = 'C:/Users/bavon/.gemini/antigravity-ide/brain/9d3e5e78-a361-42e7-90ce-ee53cf6cd319/.user_uploaded/media_1788542807961.png'
img = cv2.imread(img_path)
print('Image shape:', img.shape)

# Crop out the HUD and bottom banner so we only see the camera content
# HUD is top 42px, banner is bottom 80px
h, w = img.shape[:2]
cam_img = img[50:h-90, :]

detector = YOLOBrailleDetector(mode='yolo')
cells, debug = detector.detect(cam_img)
dots = debug['dots']
print('Detected dots count:', len(dots))
print('Detected cells count:', len(cells))

centers = np.array([d['center'] for d in dots])
print(f"X range: {centers[:, 0].min():.1f} to {centers[:, 0].max():.1f}")
print(f"Y range: {centers[:, 1].min():.1f} to {centers[:, 1].max():.1f}")

dot_spacing = detector._opencv_detector._estimate_dot_spacing(centers)
print(f"Estimated dot_spacing: {dot_spacing:.2f}")

tol = dot_spacing * detector._opencv_detector.config.CLUSTER_TOLERANCE
print(f"Tolerance: {tol:.2f}")

_row_labels, row_centers = detector._opencv_detector._cluster_1d(centers[:, 1], tol)
print(f"Row centers ({len(row_centers)}):", [round(r, 1) for r in sorted(row_centers)])

_col_labels, col_centers = detector._opencv_detector._cluster_1d(centers[:, 0], tol)
print(f"Col centers count: {len(col_centers)}")

row_groups = detector._opencv_detector._group_rows_into_cells(sorted(row_centers), dot_spacing)
print(f"Row groups ({len(row_groups)}):", [[round(r, 1) for r in g] for g in row_groups])

col_groups = detector._opencv_detector._split_columns_into_cells(sorted(col_centers), dot_spacing)
print(f"Col groups count: {len(col_groups)}")
