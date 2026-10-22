import sys, os
sys.path.insert(0, os.path.abspath('.'))
import cv2, numpy as np
from yolo_detector import YOLOBrailleDetector
from scratch.test_new_clustering import cluster_into_cells_robust
from decoder import decode_cells

img = cv2.imread('C:/Users/bavon/.gemini/antigravity-ide/brain/9d3e5e78-a361-42e7-90ce-ee53cf6cd319/.user_uploaded/media_1788542807961.png')
h, w = img.shape[:2]
cam_img = img[50:h-90, :]
detector = YOLOBrailleDetector(mode='yolo')
_, debug = detector.detect(cam_img)
dots = debug['dots']
spacing = detector._opencv_detector._estimate_dot_spacing(np.array([d['center'] for d in dots]))
cells = cluster_into_cells_robust(dots, spacing)

print(f'Detected {len(cells)} cells from {len(dots)} dots:')
for i, c in enumerate(cells):
    print(f'  Cell {i:2d}: dots={sorted(c["dots"])} center={c["center"]}')

thai_text = decode_cells(cells, lang='thai')
print(f'\nDecoded Thai text: "{thai_text}"')
