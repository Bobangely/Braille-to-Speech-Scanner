"""
โปรแกรมทดสอบกล้อง Webcam และตรวจสอบความละเอียดสูงสุด (รองรับ 4K / 1080p / 720p)
==========================================================================
- ตรวจสอบความละเอียดที่กล้องรองรับจริง (Probe supported resolutions)
- ทดสอบเปิดความละเอียดสูงสุดที่กล้องทำได้
- กด Q เพื่อปิด
"""

import sys
import time
import cv2

print("=" * 65)
print("  Camera Resolution Probe & Test - Braille Reader")
print("=" * 65)

camera_id = 0
print(f"  Testing camera index: {camera_id}")

# ทดสอบระดับความละเอียดต่างๆ
RESOLUTIONS_TO_TEST = [
    ("4K UHD", 3840, 2160),
    ("2K QHD", 2560, 1440),
    ("Full HD 1080p", 1920, 1080),
    ("HD 720p", 1280, 720),
    ("VGA 480p", 640, 480),
]

# เปิดกล้องด้วย DirectShow บน Windows เพื่อให้ส่ง 4K ได้
backend = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY
cap = cv2.VideoCapture(camera_id, backend)
if not cap.isOpened():
    cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print("  [ERROR] Cannot open camera!")
    print("  Tips:")
    print("    - Check if webcam is plugged in")
    print("    - Try camera index 1: python test_camera.py")
    sys.exit(1)

# ตั้งค่า Codec เป็น MJPG สำหรับแบนด์วิธความเร็วสูง
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

print("\n  🔍 กำลังตรวจสอบความละเอียดที่กล้องรองรับ:")
print("  " + "-" * 55)

max_supported_w = 640
max_supported_h = 480
max_name = "VGA 480p"

for name, target_w, target_h in RESOLUTIONS_TO_TEST:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
    time.sleep(0.05)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    is_supported = (actual_w == target_w and actual_h == target_h)
    status = "✅ SUPPORTED" if is_supported else f"❌ (Fallback to {actual_w}x{actual_h})"
    print(f"  • {name:<15} ({target_w}x{target_h}): {status}")

    if is_supported and (actual_w > max_supported_w):
        max_supported_w = actual_w
        max_supported_h = actual_h
        max_name = name

print("  " + "-" * 55)
print(f"  🏆 ความละเอียดสูงสุดที่เลือกใช้งาน: {max_name} ({max_supported_w}x{max_supported_h})")
print()

# ตั้งค่าความละเอียดสูงสุดที่ทำได้
cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_supported_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_supported_h)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

window_name = f"Camera Test [{actual_w}x{actual_h}] - Press Q to quit"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# ปรับขนาดหน้าต่างแสดงผลให้พอดีหน้าจอคอม
disp_w = min(1280, actual_w)
disp_h = int(disp_w * (actual_h / max(1, actual_w)))
cv2.resizeWindow(window_name, disp_w, disp_h)

print("  Press [Q] or [ESC] to quit")
print("=" * 65)

frame_count = 0
prev_t = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame_count += 1
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-5))
    prev_t = now

    h, w = frame.shape[:2]

    # วาดข้อมูลบนภาพ
    cv2.rectangle(frame, (10, 10), (450, 100), (20, 20, 20), -1)
    cv2.putText(frame, f"Resolution: {w}x{h} ({max_name})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2)
    cv2.putText(frame, f"FPS: {fps:.1f} | Frames: {frame_count}",
                (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 80), 1)
    cv2.putText(frame, "Press Q to quit",
                (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
print(f"  Done! Closed camera.")
