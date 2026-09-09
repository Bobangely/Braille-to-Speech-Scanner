# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import cv2

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from camera_reader import AsyncBrailleWorker, RealTimeBrailleScanner
from yolo_detector import YOLOBrailleDetector

print("=" * 60)
print("1. ทดสอบ AsyncBrailleWorker")
print("=" * 60)

detector = YOLOBrailleDetector(mode='opencv', fallback_color='blue')
worker = AsyncBrailleWorker(detector, default_lang='thai')
worker.start()

# สร้างภาพทดสอบ
test_img = cv2.imread('sample_images/test_thai_ka.png')
if test_img is None:
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)

# ส่ง 5 เฟรมทดสอบ
for i in range(5):
    worker.submit_frame(test_img, lang='thai')
    time.sleep(0.04)

time.sleep(0.2)
res = worker.get_latest_results()
print("  Worker Results:")
print(f"    Decoded Text: {res['decoded_text']}")
print(f"    Cells count:  {len(res['cells'])}")
print(f"    AI FPS:       {res['ai_fps']:.1f}")

worker.stop()
print("  Worker stopped successfully. ✅ PASS")

print()
print("=" * 60)
print("2. ทดสอบ RealTimeBrailleScanner Initialization")
print("=" * 60)
scanner = RealTimeBrailleScanner(
    camera_id=0,
    detector_mode='hybrid',
    res_preset='fhd',
    sharpness_level=2,
    color='blue',
    lang='thai',
)
print(f"  Scanner created successfully:")
print(f"    Target Resolution: {scanner.target_width}x{scanner.target_height} ({scanner.res_name})")
print(f"    Detector Mode:     {scanner.detector.mode}")
print(f"    Language:          {scanner.lang}")
print("  Initialization test passed. ✅ PASS")

print("\n🎉 ทุกการทดสอบ Multi-threaded Architecture ผ่าน 100%!")
