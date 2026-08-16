"""
Braille Reader - Test Image Generator
=======================================
สร้างภาพทดสอบอักษรเบรลล์ที่มีจุดสี
เพื่อใช้ทดสอบ detector โดยไม่ต้องมีภาพจริง
"""

import cv2
import numpy as np
import os

from config import CHAR_TO_BRAILLE


def generate_braille_image(
    text,
    dot_color_bgr=(255, 120, 0),   # สีน้ำเงิน (BGR)
    bg_color_bgr=(245, 245, 245),  # สีพื้นหลัง
    dot_radius=10,
    dot_spacing=40,                 # ระยะห่างระหว่างจุดใน cell
    cell_gap=55,                    # ระยะห่างระหว่าง cell
    margin=60,
    add_noise=False,
):
    """
    สร้างภาพ Braille จากข้อความ

    Parameters
    ----------
    text : str
        ข้อความที่ต้องการแปลง (a-z เท่านั้น)
    dot_color_bgr : tuple
        สีของจุด ในรูปแบบ BGR
    bg_color_bgr : tuple
        สีพื้นหลัง ในรูปแบบ BGR
    dot_radius : int
        รัศมีของจุด (pixels)
    dot_spacing : int
        ระยะห่างระหว่างจุดภายใน cell (pixels)
    cell_gap : int
        ระยะห่างระหว่าง cell (pixels)
    margin : int
        ขอบรอบภาพ (pixels)
    add_noise : bool
        เพิ่ม noise เพื่อจำลองภาพจริง

    Returns
    -------
    image : np.ndarray
        ภาพ BGR
    """
    text = text.lower()

    # กรองเอาเฉพาะอักษรที่มี mapping
    valid_chars = [c for c in text if c in CHAR_TO_BRAILLE or c == ' ']

    if not valid_chars:
        raise ValueError(f"ไม่พบตัวอักษรที่รองรับใน '{text}'")

    # คำนวณขนาดภาพ
    n_chars = len(valid_chars)
    cell_width = dot_spacing   # ความกว้างของ 1 cell (2 columns = dot_spacing)
    cell_height = dot_spacing * 2  # ความสูงของ 1 cell (3 rows = 2*dot_spacing)

    img_width = margin * 2 + n_chars * cell_width + (n_chars - 1) * cell_gap
    img_height = margin * 2 + cell_height

    # สร้างภาพพื้นหลัง
    image = np.full((img_height, img_width, 3), bg_color_bgr, dtype=np.uint8)

    # เพิ่ม texture เบาๆ เพื่อจำลองกระดาษ
    if add_noise:
        noise = np.random.normal(0, 3, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # วาดจุดสำหรับแต่ละตัวอักษร
    for i, char in enumerate(valid_chars):
        if char == ' ':
            continue

        dots = CHAR_TO_BRAILLE.get(char, frozenset())

        # จุดเริ่มต้นของ cell (มุมซ้ายบน)
        cell_x = margin + i * (cell_width + cell_gap)
        cell_y = margin

        for dot_num in dots:
            # แปลง dot number เป็น (row, col)
            # dots 1,2,3 = col 0 | dots 4,5,6 = col 1
            if dot_num <= 3:
                col = 0
                row = dot_num - 1
            else:
                col = 1
                row = dot_num - 4

            # คำนวณตำแหน่ง pixel
            px = cell_x + col * dot_spacing
            py = cell_y + row * dot_spacing

            # วาดจุดกลม
            cv2.circle(image, (px, py), dot_radius, dot_color_bgr, -1)

            # เพิ่มขอบเบาๆ เพื่อให้ดูเหมือนจุดนูน
            cv2.circle(
                image, (px, py), dot_radius,
                tuple(max(0, c - 40) for c in dot_color_bgr), 2
            )

    # วาดข้อความ ground truth ด้านล่าง
    text_str = ''.join(valid_chars)
    cv2.putText(
        image, f'Ground Truth: "{text_str}"',
        (margin, img_height - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (100, 100, 100), 1,
    )

    return image


def generate_test_suite(output_dir='sample_images'):
    """
    สร้างชุดภาพทดสอบหลายรูปแบบ
    """
    os.makedirs(output_dir, exist_ok=True)

    test_cases = [
        # (ชื่อไฟล์, ข้อความ, สี BGR, noise)
        ('test_hello_blue.png', 'hello', (255, 120, 0), False),
        ('test_abc_blue.png', 'abc', (255, 120, 0), False),
        ('test_world_red.png', 'world', (0, 0, 220), False),
        ('test_braille_green.png', 'braille', (0, 180, 0), False),
        ('test_hello_noisy.png', 'hello', (255, 120, 0), True),
        ('test_alphabet.png', 'abcdefghij', (255, 120, 0), False),
        ('test_python.png', 'python', (255, 120, 0), False),
    ]

    generated = []

    for filename, text, color, noise in test_cases:
        filepath = os.path.join(output_dir, filename)
        try:
            img = generate_braille_image(
                text,
                dot_color_bgr=color,
                add_noise=noise,
            )
            cv2.imwrite(filepath, img)
            print(f"  [OK] {filepath:<40} -> '{text}'")
            generated.append(filepath)
        except Exception as e:
            print(f"  [ERR] {filepath:<40} -> Error: {e}")

    return generated


if __name__ == '__main__':
    print("=" * 60)
    print("  Braille Test Image Generator")
    print("=" * 60)
    print()

    files = generate_test_suite()

    print()
    print(f"สร้างภาพทดสอบสำเร็จ {len(files)} ไฟล์")
    print("ใช้คำสั่ง: python main.py sample_images/test_hello_blue.png")
