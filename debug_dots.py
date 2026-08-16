"""Debug script to inspect dot detection"""
import cv2
import numpy as np
from detector import BrailleDetector

img = cv2.imread('sample_images/test_hello_blue.png')
det = BrailleDetector('blue')

processed = det._preprocess(img)
mask = det._color_segment(processed)
mask = det._morph_clean(mask)

# Raw contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours found: {len(contours)}")
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    print(f"  Contour {i}: center=({cx},{cy}), area={area:.0f}, circ={circ:.2f}")

print()

# After filtering
dots = det._find_dots(mask)
print(f"Dots after filtering: {len(dots)}")
for i, d in enumerate(dots):
    cx, cy = d["center"]
    print(f"  Dot {i}: center=({cx},{cy}), area={d['area']:.0f}, circ={d['circularity']:.2f}")
