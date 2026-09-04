"""
Auto-Annotation & Dataset Augmentation for Braille Images
=========================================================
ใช้ OpenCV Detector ดึง Ground-Truth Dot Bounding Boxes จาก sample_images/ ทั้งหมด
พร้อมสร้าง Augmented Images (จำลองสภาพแสง เงา มุมกล้อง การสั่นไหว)
เพื่อเทรน YOLOv8 ให้จำจุดเบรลล์ในรูปภาพจริงได้อย่างแม่นยำ 100%
"""

import os
import sys
import glob
import random
import cv2
import numpy as np
import yaml

# ปรับ encoding สำหรับ Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from detector import BrailleDetector
from decoder import decode_cells

# Color detector candidates
SUPPORTED_COLORS = ['blue', 'red', 'green', 'black']


def find_best_dots(image_path):
    """
    ค้นหาจุดที่ถูกต้องที่สุดจากภาพ โดยลองทุกสีใน OpenCV detector
    เลือกสีที่ได้ dots และ cells สมบูรณ์ที่สุด
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    best_dots = []
    best_color = None
    max_cells = -1

    for color in SUPPORTED_COLORS:
        detector = BrailleDetector(dot_color=color)
        try:
            cells, debug_info = detector.detect(img)
            dots = debug_info.get('dots', [])
            if len(cells) > max_cells and len(dots) >= 1:
                max_cells = len(cells)
                best_dots = dots
                best_color = color
        except Exception:
            continue

    # หากไม่มี cell เลย ให้ลองเอาจุดที่มากที่สุดและ circularity สูง
    if not best_dots:
        for color in SUPPORTED_COLORS:
            detector = BrailleDetector(dot_color=color)
            dots = detector._find_dots(img)
            if len(dots) > len(best_dots):
                best_dots = dots
                best_color = color

    return img, best_dots, best_color


def dots_to_yolo_bboxes(img_shape, dots):
    """
    แปลงพิกัด dots เป็น YOLO format:
    class_id(0) x_center y_center width height (normalized 0.0 - 1.0)
    """
    h, w = img_shape[:2]
    bboxes = []

    for dot in dots:
        cx, cy = dot['center']
        # คำนวณ radius จาก area หรือ contour
        if 'contour' in dot and dot['contour'] is not None:
            bx, by, bw, bh = cv2.boundingRect(dot['contour'])
            # เพิ่ม padding เล็กน้อยเพื่อให้ครอบคลุมจุดทั้งหมด
            pad = max(2, int(bw * 0.15))
            bx1 = max(0, bx - pad)
            by1 = max(0, by - pad)
            bx2 = min(w, bx + bw + pad)
            by2 = min(h, by + bh + pad)
            bw = bx2 - bx1
            bh = by2 - by1
            cx = bx1 + bw / 2.0
            cy = by1 + bh / 2.0
        else:
            r = np.sqrt(dot['area'] / np.pi) * 1.2
            bw = bh = 2 * r

        # Normalize
        norm_cx = cx / w
        norm_cy = cy / h
        norm_w = bw / w
        norm_h = bh / h

        # Clamp 0.0 - 1.0
        norm_cx = max(0.001, min(0.999, norm_cx))
        norm_cy = max(0.001, min(0.999, norm_cy))
        norm_w = max(0.001, min(0.999, norm_w))
        norm_h = max(0.001, min(0.999, norm_h))

        bboxes.append((0, norm_cx, norm_cy, norm_w, norm_h))

    return bboxes


def augment_image_and_boxes(img, bboxes, aug_idx=0):
    """
    สร้างภาพ Augmented จำลองสถานการณ์กล้องจริง:
    - ปรับแสง (Brightness/Contrast)
    - เพิ่มเงาตกกระทบ (Shadow gradient)
    - เบลอเล็กน้อย (Gaussian/Motion blur)
    - หมุนเล็กน้อย (-5 ถึง +5 องศา) พร้อมปรับ bounding boxes ตาม
    - เพิ่ม Noise (Sensor grain)
    """
    h, w = img.shape[:2]
    aug_img = img.copy()
    aug_bboxes = list(bboxes)

    # 1. ปรับแสงสว่างและ Contrast
    alpha = random.uniform(0.75, 1.25)  # Contrast
    beta = random.randint(-25, 25)      # Brightness
    aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

    # 2. จำลองเงาตกกระทบ (Lighting Gradient) ในบางรูป
    if random.random() < 0.5:
        gradient = np.linspace(random.uniform(0.7, 1.0), random.uniform(0.9, 1.2), w)
        gradient = np.tile(gradient, (h, 1))
        aug_img = np.clip(aug_img * gradient[:, :, np.newaxis], 0, 255).astype(np.uint8)

    # 3. หมุนเอียงเล็กน้อย (Rotation)
    angle = random.uniform(-4.0, 4.0)
    if abs(angle) > 0.5:
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # หมุนพิกัด Bounding Boxes ตาม
        new_bboxes = []
        rad = np.radians(-angle)

        for cls_id, ncx, ncy, nw, nh in bboxes:
            px = ncx * w
            py = ncy * h
            # Transform point
            tx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
            ty = M[1, 0] * px + M[1, 1] * py + M[1, 2]

            # คำนวณขนาด bounding box ใหม่หลังหมุน
            cur_w = nw * w
            cur_h = nh * h
            new_w = abs(cur_w * np.cos(np.radians(angle))) + abs(cur_h * np.sin(np.radians(angle)))
            new_h = abs(cur_w * np.sin(np.radians(angle))) + abs(cur_h * np.cos(np.radians(angle)))

            new_ncx = max(0.001, min(0.999, tx / w))
            new_ncy = max(0.001, min(0.999, ty / h))
            new_nw = max(0.001, min(0.999, new_w / w))
            new_nh = max(0.001, min(0.999, new_h / h))

            new_bboxes.append((cls_id, new_ncx, new_ncy, new_nw, new_nh))
        aug_bboxes = new_bboxes

    # 4. เพิ่ม Gaussian Blur หรือ Motion Blur จำลองมือสั่น
    blur_choice = random.random()
    if blur_choice < 0.35:
        ksize = random.choice([3, 5])
        aug_img = cv2.GaussianBlur(aug_img, (ksize, ksize), 0)
    elif blur_choice < 0.55:
        size = random.choice([3, 5])
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size)
        kernel = kernel / size
        aug_img = cv2.filter2D(aug_img, -1, kernel)

    # 5. เพิ่ม Gaussian Noise (Webcam sensor noise)
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(3, 10), aug_img.shape).astype(np.float32)
        aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return aug_img, aug_bboxes


def process_dataset(sample_dir='sample_images', output_dir='datasets/braille_dots', num_augs_per_img=6):
    """
    ประมวลผลรูปภาพใน sample_images/ ทั้งหมด
    และสร้าง YOLO dataset พร้อม train/val split
    """
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)

    img_patterns = [
        os.path.join(sample_dir, '*.png'),
        os.path.join(sample_dir, '*.jpg'),
    ]

    all_images = []
    for pattern in img_patterns:
        for p in glob.glob(pattern):
            # ข้ามรูป output เก่าที่ลงท้ายด้วย _yolo หรือ annotated
            base = os.path.basename(p)
            if '_yolo' in base or 'annotated' in base:
                continue
            all_images.append(p)

    all_images = sorted(list(set(all_images)))
    print(f"📦 พบรูปภาพต้นแบบใน {sample_dir}: {len(all_images)} ภาพ")

    success_count = 0
    total_samples_generated = 0

    random.seed(42)
    random.shuffle(all_images)

    for idx, img_path in enumerate(all_images):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        img, dots, color = find_best_dots(img_path)

        if img is None or not dots or len(dots) == 0:
            print(f"  ⚠️ ไม่สามารถสกัดจุดจาก: {base_name}")
            continue

        success_count += 1
        bboxes = dots_to_yolo_bboxes(img.shape, dots)

        # สร้างรูปต้นฉบับ + Augmented versions
        variants = [(f"{base_name}_orig", img, bboxes)]

        for aug_i in range(num_augs_per_img):
            aug_img, aug_boxes = augment_image_and_boxes(img, bboxes, aug_i)
            variants.append((f"{base_name}_aug_{aug_i+1}", aug_img, aug_boxes))

        # สุ่มจัดสรรเข้า train (85%) หรือ val (15%)
        for var_name, var_img, var_boxes in variants:
            split = 'val' if random.random() < 0.15 else 'train'

            img_out = os.path.join(output_dir, 'images', split, f"{var_name}.png")
            lbl_out = os.path.join(output_dir, 'labels', split, f"{var_name}.txt")

            cv2.imwrite(img_out, var_img)
            with open(lbl_out, 'w', encoding='utf-8') as f:
                for cls_id, cx, cy, w, h in var_boxes:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            total_samples_generated += 1

    # สร้าง data.yaml สำหรับ YOLO
    abs_output_dir = os.path.abspath(output_dir).replace('\\', '/')
    yaml_data = {
        'path': abs_output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'braille_dot',
        },
        'nc': 1,
    }

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"✅ Auto-Annotation เสร็จสิ้น!")
    print(f"  • ภาพต้นแบบที่ดึงจุดสำเร็จ: {success_count}/{len(all_images)} ภาพ")
    print(f"  • รวมชุดข้อมูลทั้งหมด (รวม Augmentations): {total_samples_generated} ภาพ")
    print(f"  • บันทึก Config ไปที่: {yaml_path}")
    return total_samples_generated


if __name__ == '__main__':
    process_dataset()
