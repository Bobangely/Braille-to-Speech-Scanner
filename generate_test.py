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

    # Word gap ต้องใหญ่กว่า detector space_threshold (dot_spacing * 6.5)
    word_gap = int(dot_spacing * 7.0)
    cell_pitch = cell_gap + dot_spacing  # within-word cell pitch

    # คำนวณ X offset ของแต่ละ cell (None = word space → เพิ่ม word_gap)
    cell_x_positions = []
    x_cursor = margin
    for i, dots in enumerate(cells):
        if dots is None:
            # space: เพิ่ม gap พิเศษ (ไม่วาดจุด)
            x_cursor += word_gap
        else:
            cell_x_positions.append((x_cursor, dots))
            x_cursor += cell_pitch

    cell_height = dot_spacing * 2
    img_width = x_cursor + margin
    img_height = margin * 2 + cell_height

    # สร้างภาพพื้นหลัง
    image = np.full((img_height, img_width, 3), bg_color_bgr, dtype=np.uint8)

    if add_noise:
        noise = np.random.normal(0, 3, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # วาดจุดสำหรับแต่ละ cell
    for cell_x, dots in cell_x_positions:
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


from detector import BrailleDetector
from decoder import decode_cells, decode_cells_verbose


def bgr_to_color_name(bgr):
    """แปลง BGR tuple เป็นชื่อสีสำหรับ detector"""
    if bgr == (255, 120, 0):
        return 'blue'
    elif bgr == (0, 0, 220):
        return 'red'
    elif bgr == (0, 180, 0):
        return 'green'
    return 'blue'


def generate_test_suite(output_dir='sample_images', annotated_dir='output', save_annotated=True):
    """
    สร้างชุดภาพทดสอบภาษาอังกฤษและภาษาไทย
    พร้อมสร้างภาพ Annotated (มี Grid 2x3 และแถบข้อความคำแปล) บันทึกลง output/
    """
    os.makedirs(output_dir, exist_ok=True)
    if save_annotated:
        os.makedirs(annotated_dir, exist_ok=True)

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

    # ภาษาไทย — ชุดทดสอบครอบคลุม
    thai_cases = [
        # --- 1. คำเดี่ยว พื้นฐาน ---
        ('test_thai_ka.png', 'กา', (255, 120, 0), False),
        ('test_thai_thai.png', 'ไทย', (255, 120, 0), False),
        ('test_thai_khon.png', 'คน', (0, 0, 220), False),
        ('test_thai_cat.png', 'แมว', (0, 180, 0), False),
        ('test_thai_home.png', 'บ้าน', (255, 120, 0), False),
        ('test_thai_consonants.png', 'กขคงจ', (255, 120, 0), False),

        # --- 2. สระเดี่ยว (Short & Long Vowels) ---
        ('test_thai_v_a.png', 'กะ', (255, 120, 0), False),           # สระอะ
        ('test_thai_v_aa.png', 'กา', (255, 120, 0), False),          # สระอา
        ('test_thai_v_i.png', 'กิ', (255, 120, 0), False),           # สระอิ
        ('test_thai_v_ii.png', 'กี', (255, 120, 0), False),          # สระอี
        ('test_thai_v_ue.png', 'กึ', (255, 120, 0), False),          # สระอึ
        ('test_thai_v_uee.png', 'กื', (255, 120, 0), False),         # สระอือ
        ('test_thai_v_u.png', 'กุ', (255, 120, 0), False),           # สระอุ
        ('test_thai_v_uu.png', 'กู', (255, 120, 0), False),          # สระอู

        # --- 3. สระนำ (Leading Vowels) ---
        ('test_thai_v_e.png', 'เก', (255, 120, 0), False),           # สระเอ
        ('test_thai_v_ae.png', 'แก', (255, 120, 0), False),          # สระแอ
        ('test_thai_v_o.png', 'โก', (0, 0, 220), False),             # สระโอ
        ('test_thai_v_ai.png', 'ไก', (255, 120, 0), False),          # สระไอ

        # --- 4. วรรณยุกต์ (Tone Marks) ---
        ('test_thai_t_ek.png', 'ก่า', (255, 120, 0), False),         # ไม้เอก
        ('test_thai_t_tho.png', 'ก้า', (255, 120, 0), False),        # ไม้โท
        ('test_thai_t_tri.png', 'ก๊า', (0, 180, 0), False),          # ไม้ตรี
        ('test_thai_t_chat.png', 'ก๋า', (255, 120, 0), False),       # ไม้จัตวา

        # --- 5. สระผสม (Compound Vowels) ---
        ('test_thai_cv_ao.png', 'เกา', (255, 120, 0), False),        # เ◌า
        ('test_thai_cv_uea.png', 'เกือ', (255, 120, 0), False),      # เ◌ือ
        ('test_thai_cv_am.png', 'กำ', (0, 0, 220), False),           # สระอำ

        # --- 6. ประโยค (Sentences) ---
        ('test_thai_sent_khon_rak_hma.png', 'คน รัก หมา', (255, 120, 0), False),
        ('test_thai_sent_pai_kin_khao.png', 'ไป กิน ข้าว', (0, 0, 220), False),
        ('test_thai_sent_wan_nee_dee.png', 'วัน นี้ ดี', (0, 180, 0), False),
        ('test_thai_sent_chan_rak_ther.png', 'ฉัน รัก เธอ', (255, 120, 0), False),
        ('test_thai_sent_kin_khao_kan.png', 'กิน ข้าว กัน', (255, 120, 0), False),
        ('test_thai_sent_nam_jai_dee.png', 'น้ำ ใจ ดี', (255, 120, 0), False),
        ('test_thai_sent_maa_kin_khao.png', 'มา กิน ข้าว', (0, 180, 0), False),
    ]

    generated = []

    def _process_cases(cases, lang):
        for filename, text, color, noise in cases:
            filepath = os.path.join(output_dir, filename)
            base_name = os.path.splitext(filename)[0]
            try:
                # 1. สร้างภาพอักษรเบรลล์ดิบ (Raw dot image)
                img = generate_braille_image(
                    text, dot_color_bgr=color, add_noise=noise, lang=lang
                )
                cv2.imwrite(filepath, img)

                # 2. สร้างภาพ Annotated พร้อม Grid และแบนเนอร์แสดงข้อความ
                anno_info = ""
                if save_annotated:
                    color_name = bgr_to_color_name(color)
                    detector = BrailleDetector(dot_color=color_name)
                    cells, debug_info = detector.detect(img)
                    decoded_text = decode_cells(cells, lang=lang)
                    verbose_results = decode_cells_verbose(cells, lang=lang)

                    annotated_img = detector.annotate_with_text(
                        img, debug_info['dots'], cells,
                        decoded_text=decoded_text,
                        verbose_results=verbose_results,
                        lang=lang
                    )
                    anno_path = os.path.join(annotated_dir, f"{base_name}_annotated.png")
                    cv2.imwrite(anno_path, annotated_img)
                    anno_info = f" -> Grid: {anno_path}"

                print(f"  [OK] {filepath:<40} -> '{text}'{anno_info}")
                generated.append(filepath)
            except Exception as e:
                print(f"  [ERR] {filepath:<40} -> Error: {e}")

    print("--- สร้างภาพทดสอบภาษาอังกฤษ ---")
    _process_cases(english_cases, lang='english')

    print("\n--- สร้างภาพทดสอบภาษาไทย ---")
    _process_cases(thai_cases, lang='thai')

    return generated


if __name__ == '__main__':
    print("=" * 70)
    print("  Braille Test Image Generator (with 2x3 Grid & Word Banner)")
    print("=" * 70)
    print()

    files = generate_test_suite()

    print()
    print(f"สร้างภาพทดสอบสำเร็จ {len(files)} ไฟล์")
    print("ภาพดิบ (Input):     sample_images/")
    print("ภาพมี Grid + คำ:   output/*_annotated.png")
    print()
    print("ทดสอบอังกฤษ:  python main.py sample_images/test_hello_blue.png")
    print("ทดสอบไทย:     python main.py sample_images/test_thai_ka.png --lang thai")

