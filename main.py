"""
Braille Reader - Main Entry Point
====================================
อ่านภาพอักษรเบรลล์ที่มีจุดสี แล้วแปลงเป็นข้อความ

Usage:
    python main.py <image_path> [--color blue|red|green] [--save]

Examples:
    python main.py sample_images/test_hello_blue.png
    python main.py sample_images/test_world_red.png --color red
    python main.py photo.jpg --color blue --save
"""

import argparse
import sys
import os

import cv2
import numpy as np

# แก้ปัญหา encoding บน Windows (cp874 ไม่รองรับ Braille Unicode)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from detector import BrailleDetector
from decoder import decode_cells, decode_cells_verbose, dots_to_braille_unicode
from config import DetectionConfig


def print_banner():
    """แสดง banner"""
    print()
    print("=" * 50)
    print("   Braille Reader - Colored Dot OBR")
    print("   OpenCV + Python Prototype")
    print("=" * 50)
    print()


def print_results(cells, verbose_results, decoded_text):
    """แสดงผลลัพธ์การถอดรหัสใน Terminal อย่างชัดเจน"""
    print("-" * 55)
    print(f"   จุดที่พบ (Dots):     {sum(len(c['dots']) for c in cells)} จุด")
    print(f"   เซลล์อักษร (Cells): {len(cells)} เซลล์")
    print("-" * 55)
    print()

    if verbose_results:
        print("  🔍 รายละเอียดแต่ละเซลล์:")
        print("  " + "─" * 45)
        for i, r in enumerate(verbose_results, 1):
            dots_str = ','.join(map(str, r['dots']))
            print(
                f"    Cell {i:>2}:  "
                f"dots({dots_str:>11})  "
                f"->  {r['braille_unicode']}  "
                f"->  '{r['char']}'"
            )
        print("  " + "─" * 45)
        print()

    # แสดง Braille Unicode
    braille_line = ' '.join(r['braille_unicode'] for r in verbose_results)

    # กรอบแสดงข้อความเด่นชัด
    box_width = max(len(decoded_text) + 16, len(braille_line) + 16, 50)
    
    print("╔" + "═" * (box_width - 2) + "╗")
    print(f"║   Braille:  {braille_line:<{box_width - 15}}║")
    print(f"║   Text:     {decoded_text:<{box_width - 15}}║")
    print("╚" + "═" * (box_width - 2) + "╝")
    print()
    print(f"   ข้อความที่อ่านได้:  >>>  {decoded_text}  <<<")
    print()


def save_debug_images(debug_info, output_dir, base_name):
    """บันทึกภาพ debug"""
    os.makedirs(output_dir, exist_ok=True)

    # Save mask
    mask_path = os.path.join(output_dir, f'{base_name}_mask.png')
    cv2.imwrite(mask_path, debug_info['mask'])
    print(f"  -> Mask:      {mask_path}")

    # Save annotated image
    anno_path = os.path.join(output_dir, f'{base_name}_annotated.png')
    cv2.imwrite(anno_path, debug_info['annotated'])
    print(f"  -> Annotated: {anno_path}")


def display_images(image, debug_info):
    """แสดงภาพใน window (ถ้ามี GUI)"""
    try:
        # Resize ถ้าภาพใหญ่เกินไป
        max_width = 800
        h, w = image.shape[:2]
        if w > max_width:
            scale = max_width / w
            image = cv2.resize(image, None, fx=scale, fy=scale)
            debug_info['mask'] = cv2.resize(
                debug_info['mask'], None, fx=scale, fy=scale
            )
            debug_info['annotated'] = cv2.resize(
                debug_info['annotated'], None, fx=scale, fy=scale
            )

        cv2.imshow('Original', image)
        cv2.imshow('Color Mask', debug_info['mask'])
        cv2.imshow('Detected Braille', debug_info['annotated'])

        print("  กด key ใดก็ได้เพื่อปิดหน้าต่าง...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("  (ไม่สามารถแสดงหน้าต่างได้ — ใช้ --save เพื่อบันทึกภาพ)")


def main():
    print_banner()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Braille Reader — ตรวจจับจุดสีบนอักษรเบรลล์',
    )
    parser.add_argument(
        'image', type=str,
        help='Path ของภาพที่ต้องการอ่าน',
    )
    parser.add_argument(
        '--color', type=str, default='blue',
        choices=['blue', 'red', 'green', 'black'],
        help='สีของจุดที่แต้ม (default: blue, รองรับ: blue, red, green, black)',
    )
    parser.add_argument(
        '--save', action='store_true',
        help='บันทึกภาพ debug (mask, annotated) ลง output/',
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='ไม่แสดงหน้าต่างภาพ',
    )

    args = parser.parse_args()

    # ตรวจสอบไฟล์
    if not os.path.exists(args.image):
        print(f"  [ERR] File not found: {args.image}")
        sys.exit(1)

    # อ่านภาพ
    print(f"  อ่านภาพ:  {args.image}")
    print(f"  สีจุด:     {args.color}")
    print()

    image = cv2.imread(args.image)
    if image is None:
        print(f"  [ERR] Cannot read image: {args.image}")
        sys.exit(1)

    print(f"  ขนาดภาพ:  {image.shape[1]} x {image.shape[0]} px")
    print()

    # ตรวจจับ
    detector = BrailleDetector(dot_color=args.color)
    cells, debug_info = detector.detect(image)

    if not cells:
        print("  [WARN] No Braille characters found in image")
        print("  ลอง:")
        print("    - เปลี่ยน --color ให้ตรงกับสีจุดในภาพ")
        print("    - ปรับค่า HSV range ใน config.py")
        print("    - ตรวจสอบว่าภาพมีจุดสีชัดเจน")
        print()

        if args.save:
            base = os.path.splitext(os.path.basename(args.image))[0]
            save_debug_images(debug_info, 'output', base)

        sys.exit(0)

    # ถอดรหัส
    decoded_text = decode_cells(cells)
    verbose_results = decode_cells_verbose(cells)

    # แสดงผล
    print_results(cells, verbose_results, decoded_text)

    # บันทึก / แสดงภาพ
    if args.save:
        base = os.path.splitext(os.path.basename(args.image))[0]
        save_debug_images(debug_info, 'output', base)
        print()

    if not args.no_display:
        display_images(image, debug_info)


if __name__ == '__main__':
    main()
