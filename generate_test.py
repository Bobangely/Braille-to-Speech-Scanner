"""
Braille Reader - Test Image Generator
=======================================
สร้างภาพทดสอบอักษรเบรลล์ที่มีจุดสี
เพื่อใช้ทดสอบ detector โดยไม่ต้องมีภาพจริง
"""

import cv2
import numpy as np
import os
import sys

# แก้ปัญหา encoding บน Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from config import CHAR_TO_BRAILLE
from config_thai import THAI_CHAR_TO_BRAILLE


def text_to_braille_cells(text, lang='english'):
    """
    แปลงข้อความเป็น list ของ cell dot sets
    """
    cells = []
    if lang.lower() in ('thai', 'th'):
        for char in text:
            if char == ' ':
                cells.append(None)
            elif char in THAI_CHAR_TO_BRAILLE:
                pattern = THAI_CHAR_TO_BRAILLE[char]
                if isinstance(pattern, tuple):
                    # Multi-cell (e.g. prefix 6 + base)
                    cells.extend(pattern)
                else:
                    cells.append(pattern)
    else:
        for char in text.lower():
            if char == ' ':
                cells.append(None)
            elif char in CHAR_TO_BRAILLE:
                cells.append(CHAR_TO_BRAILLE[char])
    return cells


def generate_braille_image(
    text,
    dot_color_bgr=(255, 120, 0),   # สีน้ำเงิน (BGR)
    bg_color_bgr=(245, 245, 245),  # สีพื้นหลัง
    dot_radius=10,
    dot_spacing=40,                 # ระยะห่างระหว่างจุดใน cell
    cell_gap=55,                    # ระยะห่างระหว่าง cell
    margin=60,
    add_noise=False,
    lang='english',
):
    """
    สร้างภาพ Braille จากข้อความ (รองรับทั้ง English และ Thai)
    """
    cells = text_to_braille_cells(text, lang=lang)
    if not cells:
        raise ValueError(f"ไม่พบตัวอักษรที่รองรับใน '{text}' (lang={lang})")

    n_cells = len(cells)
    cell_width = dot_spacing
    cell_height = dot_spacing * 2

    img_width = margin * 2 + n_cells * cell_width + (n_cells - 1) * cell_gap
    img_height = margin * 2 + cell_height

    # สร้างภาพพื้นหลัง
    image = np.full((img_height, img_width, 3), bg_color_bgr, dtype=np.uint8)

    if add_noise:
        noise = np.random.normal(0, 3, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # วาดจุดสำหรับแต่ละ cell
    for i, dots in enumerate(cells):
        if dots is None:
            continue

        cell_x = margin + i * (cell_width + cell_gap)
        cell_y = margin

        for dot_num in dots:
            if dot_num <= 3:
                col = 0
                row = dot_num - 1
            else:
                col = 1
                row = dot_num - 4

            px = cell_x + col * dot_spacing
            py = cell_y + row * dot_spacing

            # วาดจุดกลม
            cv2.circle(image, (px, py), dot_radius, dot_color_bgr, -1)
            cv2.circle(
                image, (px, py), dot_radius,
                tuple(max(0, c - 40) for c in dot_color_bgr), 2
            )

    return image


def generate_test_suite(output_dir='sample_images'):
    """
    สร้างชุดภาพทดสอบภาษาอังกฤษและภาษาไทย
    """
    os.makedirs(output_dir, exist_ok=True)

    # ภาษาอังกฤษ
    english_cases = [
        ('test_hello_blue.png', 'hello', (255, 120, 0), False),
        ('test_abc_blue.png', 'abc', (255, 120, 0), False),
        ('test_world_red.png', 'world', (0, 0, 220), False),
        ('test_braille_green.png', 'braille', (0, 180, 0), False),
        ('test_hello_noisy.png', 'hello', (255, 120, 0), True),
        ('test_alphabet.png', 'abcdefghij', (255, 120, 0), False),
        ('test_python.png', 'python', (255, 120, 0), False),
    ]

    # ภาษาไทย
    thai_cases = [
        ('test_thai_ka.png', 'กา', (255, 120, 0), False),
        ('test_thai_thai.png', 'ไทย', (255, 120, 0), False),
        ('test_thai_khon.png', 'คน', (0, 0, 220), False),
        ('test_thai_cat.png', 'แมว', (0, 180, 0), False),
        ('test_thai_home.png', 'บ้าน', (255, 120, 0), False),
        ('test_thai_consonants.png', 'กขคงจ', (255, 120, 0), False),
    ]

    generated = []

    print("--- สร้างภาพทดสอบภาษาอังกฤษ ---")
    for filename, text, color, noise in english_cases:
        filepath = os.path.join(output_dir, filename)
        try:
            img = generate_braille_image(
                text, dot_color_bgr=color, add_noise=noise, lang='english'
            )
            cv2.imwrite(filepath, img)
            print(f"  [OK] {filepath:<40} -> '{text}'")
            generated.append(filepath)
        except Exception as e:
            print(f"  [ERR] {filepath:<40} -> Error: {e}")

    print("\n--- สร้างภาพทดสอบภาษาไทย ---")
    for filename, text, color, noise in thai_cases:
        filepath = os.path.join(output_dir, filename)
        try:
            img = generate_braille_image(
                text, dot_color_bgr=color, add_noise=noise, lang='thai'
            )
            cv2.imwrite(filepath, img)
            print(f"  [OK] {filepath:<40} -> '{text}'")
            generated.append(filepath)
        except Exception as e:
            print(f"  [ERR] {filepath:<40} -> Error: {e}")

    return generated


if __name__ == '__main__':
    print("=" * 60)
    print("  Braille Test Image Generator (English & Thai)")
    print("=" * 60)
    print()

    files = generate_test_suite()

    print()
    print(f"สร้างภาพทดสอบสำเร็จ {len(files)} ไฟล์")
    print("ทดสอบอังกฤษ:  python main.py sample_images/test_hello_blue.png")
    print("ทดสอบไทย:     python main.py sample_images/test_thai_ka.png --lang thai")

