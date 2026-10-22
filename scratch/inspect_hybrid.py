import sys, os
sys.path.insert(0, os.path.abspath('.'))
import cv2
from yolo_detector import YOLOBrailleDetector

img = cv2.imread('C:/Users/bavon/.gemini/antigravity-ide/brain/9d3e5e78-a361-42e7-90ce-ee53cf6cd319/.user_uploaded/media_1788542807961.png')
h, w = img.shape[:2]
cam_img = img[50:h-90, :]

det = YOLOBrailleDetector(mode='hybrid')
results = det.model(cam_img, verbose=False, conf=det.confidence)
raw_dots = det._extract_yolo_boxes(results, cam_img.shape)
print('Raw YOLO dots count:', len(raw_dots))
refined_dots = det._refine_dots_with_opencv(cam_img, raw_dots)
print('Refined dots count:', len(refined_dots))

for i in range(min(15, len(raw_dots))):
    rc = raw_dots[i]['center']
    fc = refined_dots[i]['center']
    dist = ((rc[0]-fc[0])**2 + (rc[1]-fc[1])**2)**0.5
    print(f'Dot {i:2d}: raw=({rc[0]:.1f}, {rc[1]:.1f}) -> refined=({fc[0]:.1f}, {fc[1]:.1f}), shift={dist:.1f}px, area={refined_dots[i]["area"]:.1f}, circ={refined_dots[i]["circularity"]:.2f}')
