# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.abspath('.'))
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
from yolo_detector import YOLOBrailleDetector
from decoder import decode_cells, decode_cells_verbose

img_path = 'C:/Users/bavon/.gemini/antigravity-ide/brain/9d3e5e78-a361-42e7-90ce-ee53cf6cd319/.user_uploaded/media_1788542807961.png'
img = cv2.imread(img_path)
h, w = img.shape[:2]
cam_img = img[50:h-90, :]

print("=" * 60)
print("1. ทดสอบโหมด YOLO บนภาพจริงของผู้ใช้")
print("=" * 60)
detector_yolo = YOLOBrailleDetector(mode='yolo')
cells_yolo, debug_yolo = detector_yolo.detect(cam_img)
text_yolo = decode_cells(cells_yolo, lang='thai')
print(f"YOLO: ตรวจพบ {len(debug_yolo['dots'])} จุด, {len(cells_yolo)} เซลล์")
print(f"ข้อความที่ถอดรหัสได้: '{text_yolo}'")

print()
print("=" * 60)
print("2. ทดสอบโหมด HYBRID บนภาพจริงของผู้ใช้")
print("=" * 60)
detector_hybrid = YOLOBrailleDetector(mode='hybrid')
cells_hybrid, debug_hybrid = detector_hybrid.detect(cam_img)
text_hybrid = decode_cells(cells_hybrid, lang='thai')
print(f"HYBRID: ตรวจพบ {len(debug_hybrid['dots'])} จุด, {len(cells_hybrid)} เซลล์")
print(f"ข้อความที่ถอดรหัสได้: '{text_hybrid}'")

# Generate and save final annotated image for verification
verbose = decode_cells_verbose(cells_yolo, lang='thai')
annotated = detector_yolo._opencv_detector._annotate(
    cam_img, debug_yolo['dots'], cells_yolo,
    decoded_text=text_yolo,
    verbose_results=verbose,
    lang='thai'
)
cv2.imwrite('output/user_sample_verified.png', annotated)
print()
print("บันทึกภาพผลลัพธ์พร้อม Grid ที่สมบูรณ์แล้วที่: output/user_sample_verified.png")
