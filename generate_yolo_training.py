"""
YOLO Training Data Generator for Braille Dot Detection
=========================================================
สร้างภาพเบรลล์สังเคราะห์ (synthetic) พร้อม annotation ในรูปแบบ YOLO
เพื่อใช้เป็น training data สำหรับ train YOLOv8 ตรวจจับจุดเบรลล์

โครงสร้าง Output:
    datasets/braille_dots/
    ├── images/
    │   ├── train/       ← ภาพ training (80%)
    │   └── val/         ← ภาพ validation (20%)
    ├── labels/
    │   ├── train/       ← YOLO annotation (.txt) สำหรับ training
    │   └── val/         ← YOLO annotation (.txt) สำหรับ validation
    └── data.yaml        ← config ไฟล์สำหรับ Ultralytics YOLO training

รูปแบบ YOLO Annotation (ต่อบรรทัด):
    <class_id> <x_center> <y_center> <width> <height>
    ค่าทั้งหมดเป็น normalized (0.0 - 1.0) เทียบกับขนาดภาพ

    class_id = 0 (braille_dot) — เรามี class เดียวคือ "จุดเบรลล์"
    x_center, y_center = จุดศูนย์กลางของ bounding box
    width, height = ขนาดของ bounding box

ตัวอย่าง:
    0 0.125 0.333 0.050 0.050   ← จุดที่ตำแหน่ง (12.5%, 33.3%) ขนาด 5%x5%
"""

import os
import cv2
import numpy as np
import random
import yaml

from config_thai import THAI_CHAR_TO_BRAILLE, THAI_CONSONANTS, THAI_VOWELS


# =============================================================================
# ค่า Config สำหรับสร้างภาพสังเคราะห์
# =============================================================================

# ขนาดภาพ output
IMG_WIDTH = 640
IMG_HEIGHT = 320

# ขนาดจุดเบรลล์ (เป็น pixel)
DOT_RADIUS_MIN = 8
DOT_RADIUS_MAX = 14

# ระยะห่างระหว่างจุดใน cell (pixel)
DOT_SPACING_MIN = 22
DOT_SPACING_MAX = 35

# ระยะห่างระหว่าง cell (pixel)
CELL_GAP_MIN = 15
CELL_GAP_MAX = 30

# Margin ขอบภาพ (pixel)
MARGIN = 40

# สีจุดที่รองรับ (BGR format)
DOT_COLORS = {
    'blue':  (180, 100, 40),     # น้ำเงินเข้ม
    'red':   (30, 30, 180),      # แดง
    'green': (40, 130, 40),      # เขียว
    'black': (30, 30, 30),       # ดำ
}

# สีพื้นหลัง (BGR format) — สุ่มให้หลากหลาย
BG_COLORS = [
    (255, 255, 255),   # ขาว
    (245, 245, 240),   # ครีม
    (250, 250, 250),   # เทาอ่อน
    (240, 235, 230),   # เบจ
    (245, 240, 250),   # ม่วงอ่อน (lavender)
    (235, 245, 245),   # ฟ้าอ่อน
]


def generate_braille_cell_dots():
    """
    สุ่มสร้าง Braille cell แบบสมจริง
    
    Returns:
        set of int: ตำแหน่งจุดที่มี (1-6)
    
    หลักการ:
        - 70% สุ่มจาก dot pattern ที่ใช้จริงในภาษาไทย/อังกฤษ
        - 30% สุ่ม random เพื่อให้ model เรียนรู้ pattern ที่ไม่คุ้นเคย
    """
    if random.random() < 0.7:
        # สุ่มจาก real Thai Braille patterns
        all_patterns = list(THAI_CONSONANTS.keys()) + list(THAI_VOWELS.keys())
        pattern = random.choice(all_patterns)
        return set(pattern)
    else:
        # สุ่ม random 1-6 จุด
        num_dots = random.randint(1, 6)
        return set(random.sample([1, 2, 3, 4, 5, 6], num_dots))


def draw_braille_cell(image, x, y, dots, dot_spacing, dot_radius, dot_color_bgr, annotations, img_w, img_h):
    """
    วาด Braille cell 1 เซลล์ลงบนภาพ พร้อมบันทึก annotation

    Parameters:
    -----------
    image : np.ndarray
        ภาพ (BGR) ที่จะวาดลง
    x, y : int
        ตำแหน่งมุมซ้ายบนของ cell
    dots : set of int
        ตำแหน่งจุดที่ต้องวาด (1-6)
        Layout:  (1) (4)
                 (2) (5)
                 (3) (6)
    dot_spacing : int
        ระยะห่างระหว่างจุดแนวตั้ง/นอน (pixel)
    dot_radius : int
        รัศมีของจุดกลม (pixel)
    dot_color_bgr : tuple
        สี BGR ของจุด
    annotations : list
        รายการ annotation ที่จะเพิ่ม (YOLO format)
    img_w, img_h : int
        ขนาดภาพ (สำหรับ normalize ค่า annotation)
    """
    # ตำแหน่ง dot 1-6 ภายใน cell:
    #   dot 1 = (col=0, row=0), dot 4 = (col=1, row=0)
    #   dot 2 = (col=0, row=1), dot 5 = (col=1, row=1)
    #   dot 3 = (col=0, row=2), dot 6 = (col=1, row=2)
    dot_positions = {
        1: (0, 0), 2: (0, 1), 3: (0, 2),
        4: (1, 0), 5: (1, 1), 6: (1, 2),
    }

    for dot_id in dots:
        col, row = dot_positions[dot_id]
        # คำนวณ pixel position ของจุด
        cx = x + col * dot_spacing
        cy = y + row * dot_spacing

        # เพิ่มความสมจริง: สุ่ม jitter ตำแหน่งเล็กน้อย (±2px)
        cx += random.randint(-2, 2)
        cy += random.randint(-2, 2)

        # สุ่มขนาดจุดเล็กน้อย (±1px) เพื่อความหลากหลาย
        r = dot_radius + random.randint(-1, 1)
        r = max(3, r)  # ไม่ให้เล็กเกินไป

        # วาดจุดกลม
        cv2.circle(image, (cx, cy), r, dot_color_bgr, -1, cv2.LINE_AA)

        # --- สร้าง YOLO annotation ---
        # Bounding box = สี่เหลี่ยมรอบวงกลม
        # YOLO format: class_id x_center y_center width height (normalized 0-1)
        bbox_size = r * 2.5  # bounding box ใหญ่กว่าจุดเล็กน้อย (padding)
        x_center_norm = cx / img_w
        y_center_norm = cy / img_h
        w_norm = bbox_size / img_w
        h_norm = bbox_size / img_h

        # Clamp ค่าให้อยู่ในช่วง 0-1
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        w_norm = max(0.001, min(1.0, w_norm))
        h_norm = max(0.001, min(1.0, h_norm))

        # class_id = 0 (braille_dot)
        annotations.append(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}")


def add_noise_and_augmentation(image):
    """
    เพิ่ม noise และ augmentation ให้ภาพดูสมจริงมากขึ้น
    เพื่อให้ YOLO model เรียนรู้ได้ดีขึ้นในสภาพแวดล้อมจริง

    Augmentation ที่ใช้:
    1. Gaussian Noise — จำลองกล้องคุณภาพต่ำ
    2. Brightness/Contrast variation — จำลองแสงไม่คงที่
    3. Slight blur — จำลองกล้อง out of focus เล็กน้อย
    4. Salt & Pepper noise — จำลอง scan artifacts
    """
    # 1. Gaussian Noise (50% โอกาส)
    if random.random() < 0.5:
        noise_level = random.uniform(3, 12)
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 2. Brightness/Contrast variation (60% โอกาส)
    if random.random() < 0.6:
        alpha = random.uniform(0.85, 1.15)  # contrast
        beta = random.randint(-15, 15)       # brightness
        image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 3. Slight blur (30% โอกาส)
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    return image


def generate_single_image(index, output_dir, split='train'):
    """
    สร้างภาพเบรลล์สังเคราะห์ 1 ภาพ พร้อม YOLO annotation

    Parameters:
    -----------
    index : int
        ลำดับภาพ (สำหรับตั้งชื่อไฟล์)
    output_dir : str
        โฟลเดอร์ output (datasets/braille_dots/)
    split : str
        'train' หรือ 'val'

    Returns:
    --------
    int : จำนวนจุดที่สร้างในภาพนี้
    """
    # --- สุ่มพารามิเตอร์ภาพ ---
    bg_color = random.choice(BG_COLORS)
    dot_color_name = random.choice(list(DOT_COLORS.keys()))
    dot_color_bgr = DOT_COLORS[dot_color_name]

    # สุ่มขนาดจุดและระยะห่าง
    dot_radius = random.randint(DOT_RADIUS_MIN, DOT_RADIUS_MAX)
    dot_spacing = random.randint(DOT_SPACING_MIN, DOT_SPACING_MAX)
    cell_gap = random.randint(CELL_GAP_MIN, CELL_GAP_MAX)

    # สร้างภาพพื้นหลัง
    image = np.full((IMG_HEIGHT, IMG_WIDTH, 3), bg_color, dtype=np.uint8)

    # สุ่มจำนวน cell ที่จะวาด (3-12 cells)
    num_cells = random.randint(3, 12)

    # คำนวณ cell width (2 columns × dot_spacing)
    cell_w = dot_spacing + cell_gap  # ความกว้างรวม gap ถัดไป

    annotations = []
    total_dots = 0

    # วาง cells ไล่จากซ้ายไปขวา
    x_cursor = MARGIN
    for cell_idx in range(num_cells):
        # เช็คว่ายังพอใส่ cell ได้
        if x_cursor + dot_spacing > IMG_WIDTH - MARGIN:
            break

        # สุ่ม dot pattern สำหรับ cell นี้
        dots = generate_braille_cell_dots()
        total_dots += len(dots)

        # ตำแหน่ง Y ตรงกลางภาพ
        y_start = (IMG_HEIGHT - dot_spacing * 2) // 2

        # วาด cell
        draw_braille_cell(
            image, x_cursor, y_start, dots,
            dot_spacing, dot_radius, dot_color_bgr,
            annotations, IMG_WIDTH, IMG_HEIGHT,
        )

        # เลื่อน cursor ไปขวา
        x_cursor += dot_spacing + cell_gap

    # เพิ่ม noise/augmentation
    image = add_noise_and_augmentation(image)

    # --- บันทึกไฟล์ ---
    img_filename = f"braille_{index:05d}.png"
    label_filename = f"braille_{index:05d}.txt"

    img_path = os.path.join(output_dir, 'images', split, img_filename)
    label_path = os.path.join(output_dir, 'labels', split, label_filename)

    cv2.imwrite(img_path, image)
    with open(label_path, 'w') as f:
        f.write('\n'.join(annotations))

    return total_dots


def generate_dataset(output_dir='datasets/braille_dots', num_train=800, num_val=200):
    """
    สร้าง dataset ทั้งหมดสำหรับ YOLO training

    Parameters:
    -----------
    output_dir : str
        โฟลเดอร์ output
    num_train : int
        จำนวนภาพ training (default: 800)
    num_val : int
        จำนวนภาพ validation (default: 200)
    """
    print("=" * 60)
    print("  YOLO Braille Dot Training Data Generator")
    print("=" * 60)
    print()

    # สร้างโฟลเดอร์
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # สร้าง data.yaml (config สำหรับ YOLO training)
    # ไฟล์นี้บอก YOLO ว่า:
    #   - ภาพ training อยู่ที่ไหน
    #   - ภาพ validation อยู่ที่ไหน
    #   - มีกี่ class, ชื่ออะไรบ้าง
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,                      # number of classes = 1
        'names': ['braille_dot'],     # ชื่อ class = "braille_dot"
    }
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"  📄 data.yaml → {yaml_path}")

    # สร้างภาพ training
    print(f"\n  🔨 กำลังสร้างภาพ Training ({num_train} ภาพ)...")
    total_train_dots = 0
    for i in range(num_train):
        total_train_dots += generate_single_image(i, output_dir, 'train')
        if (i + 1) % 100 == 0:
            print(f"     [{i+1}/{num_train}] สร้างแล้ว...")

    # สร้างภาพ validation
    print(f"\n  🔨 กำลังสร้างภาพ Validation ({num_val} ภาพ)...")
    total_val_dots = 0
    for i in range(num_val):
        total_val_dots += generate_single_image(num_train + i, output_dir, 'val')
        if (i + 1) % 50 == 0:
            print(f"     [{i+1}/{num_val}] สร้างแล้ว...")

    # สรุปผล
    print(f"\n{'=' * 60}")
    print(f"  ✅ สร้าง Dataset เสร็จสมบูรณ์!")
    print(f"{'=' * 60}")
    print(f"  📁 Output: {os.path.abspath(output_dir)}")
    print(f"  📸 Training:    {num_train} ภาพ ({total_train_dots} จุด)")
    print(f"  📸 Validation:  {num_val} ภาพ ({total_val_dots} จุด)")
    print(f"  📄 Config:      {yaml_path}")
    print()
    print(f"  ขั้นตอนถัดไป:")
    print(f"    .venv\\Scripts\\python.exe train_yolo.py")
    print()


if __name__ == '__main__':
    generate_dataset()
