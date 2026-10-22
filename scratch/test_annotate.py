import sys, os
sys.path.insert(0, os.path.abspath('.'))
import cv2
import numpy as np
from yolo_detector import YOLOBrailleDetector
from scratch.test_new_clustering import cluster_into_cells_robust
from decoder import decode_cells, decode_cells_verbose

img = cv2.imread('C:/Users/bavon/.gemini/antigravity-ide/brain/9d3e5e78-a361-42e7-90ce-ee53cf6cd319/.user_uploaded/media_1788542807961.png')
h, w = img.shape[:2]
cam_img = img[50:h-90, :]

detector = YOLOBrailleDetector(mode='yolo')
_, debug = detector.detect(cam_img)
dots = debug['dots']
spacing = detector._opencv_detector._estimate_dot_spacing(np.array([d['center'] for d in dots]))
cells = cluster_into_cells_robust(dots, spacing)

decoded_text = decode_cells(cells, lang='thai')
verbose = decode_cells_verbose(cells, lang='thai')

annotated = detector._opencv_detector._annotate(
    cam_img, dots, cells,
    decoded_text=decoded_text,
    verbose_results=verbose,
    lang='thai'
)

out_path = 'scratch/annotated_result.png'
cv2.imwrite(out_path, annotated)
print(f'Saved annotated image to {out_path}')
print(f'Decoded: "{decoded_text}"')
