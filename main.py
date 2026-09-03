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
    """แสดงภาพใน window ให้คมชัด ขนาดใหญ่ สัดส่วนคงเดิมไม่บิดเบี้ยว สามารถย่อขยายหน้าต่างได้อิสระ"""
    try:
        orig = image.copy()
        mask = debug_info['mask'].copy()
        anno = debug_info['annotated'].copy()

        # --- ฟังก์ชันช่วย: resize ภาพให้พอดีกับ target width โดยรักษาสัดส่วนเดิม ---
        def fit_width(img, target_w, interp=cv2.INTER_LANCZOS4):
            h, w = img.shape[:2]
            if w == 0:
                return img
            ratio = target_w / float(w)
            new_h = int(h * ratio)
            return cv2.resize(img, (target_w, new_h), interpolation=interp)

        # --- ขนาดหน้าต่างที่ต้องการ (px) ---
        anno_w = 1200   # หน้าต่าง annotated ขนาดกว้างเต็มจอ
        small_w = 580   # หน้าต่าง original / mask ขนาดครึ่งจอ

        # Scale ภาพให้พอดีกับหน้าต่าง (รักษาอัตราส่วนเดิมอย่างเคร่งครัด)
        anno_scaled = fit_width(anno, anno_w)
        orig_scaled = fit_width(orig, small_w, cv2.INTER_NEAREST)
        mask_scaled = fit_width(mask, small_w, cv2.INTER_NEAREST)

        # --- กำหนดชื่อหน้าต่าง ---
        win_anno = 'Detected Braille'
        win_orig = 'Original'
        win_mask = 'Color Mask'

        # สร้างหน้าต่างแบบ WINDOW_AUTOSIZE เพื่อให้ขนาด = ขนาดภาพจริงพอดี (ไม่ยืด/บีบ)
        # ผู้ใช้ยังสามารถคลิก maximize ได้ แต่ default จะเป็นขนาดภาพจริง
        cv2.namedWindow(win_anno, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow(win_orig, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow(win_mask, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        # ตั้งขนาดหน้าต่างให้ตรงกับขนาดภาพที่ scale แล้ว (ไม่บิดเบี้ยว)
        anno_h = anno_scaled.shape[0]
        orig_h = orig_scaled.shape[0]
        mask_h = mask_scaled.shape[0]

        cv2.resizeWindow(win_anno, anno_w, anno_h)
        cv2.resizeWindow(win_orig, small_w, orig_h)
        cv2.resizeWindow(win_mask, small_w, mask_h)

        # จัดตำแหน่งหน้าต่างให้อยู่เป็นระเบียบบนหน้าจอ
        cv2.moveWindow(win_anno, 50, 30)
        cv2.moveWindow(win_orig, 50, anno_h + 70)
        cv2.moveWindow(win_mask, 50 + small_w + 20, anno_h + 70)

        cv2.imshow(win_anno, anno_scaled)
        cv2.imshow(win_orig, orig_scaled)
        cv2.imshow(win_mask, mask_scaled)

        print("  💡 สามารถคลิกลากขอบหน้าต่างเพื่อขยาย/ย่อได้อิสระ (สัดส่วนจะคงเดิม)")
        print("  ⌨️  กดปุ่มใดก็ได้บนหน้าต่างภาพเพื่อปิด...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"  (ไม่สามารถแสดงหน้าต่างได้: {e} — ใช้ --save เพื่อบันทึกภาพ)")


def main():
    print_banner()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Braille Reader — ตรวจจับจุดสีบนอักษรเบรลล์',
    )
    parser.add_argument(
        'image', type=str, nargs='?', default=None,
        help='Path ของภาพที่ต้องการอ่าน (เว้นว่างไว้หากต้องการเปิดกล้อง Webcam)',
    )
    parser.add_argument(
        '--camera', type=int, nargs='?', const=0, default=None,
        help='เปิดโหมดสแกน Real-Time จากกล้อง Webcam ID (default: 0)',
    )
    parser.add_argument(
        '--color', type=str, default='blue',
        choices=['blue', 'red', 'green', 'black'],
        help='สีของจุดที่แต้ม (default: blue, รองรับ: blue, red, green, black)',
    )
    # Shortcut flags สำหรับสี
    parser.add_argument('--blue', dest='color_blue', action='store_true', help='ทางลัดเลือกจุดสีน้ำเงิน')
    parser.add_argument('--red', dest='color_red', action='store_true', help='ทางลัดเลือกจุดสีแดง')
    parser.add_argument('--green', dest='color_green', action='store_true', help='ทางลัดเลือกจุดสีเขียว')
    parser.add_argument('--black', dest='color_black', action='store_true', help='ทางลัดเลือกจุดสีดำ')

    parser.add_argument(
        '--lang', type=str, default='thai',
        choices=['english', 'thai'],
        help='ภาษาที่ต้องการถอดรหัส (default: thai, รองรับ: english, thai)',
    )
    # Shortcut flags สำหรับภาษา
    parser.add_argument('--thai', dest='lang_thai', action='store_true', help='ทางลัดเลือกภาษาไทย')
    parser.add_argument('--english', dest='lang_eng', action='store_true', help='ทางลัดเลือกภาษาอังกฤษ')

    parser.add_argument(
        '--speak', action='store_true',
        help='ออกเสียงข้อความที่อ่านได้ (Text-to-Speech)',
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

    # ตรวจสอบ Shortcut flags
    if args.color_black:
        args.color = 'black'
    elif args.color_red:
        args.color = 'red'
    elif args.color_green:
        args.color = 'green'
    elif args.color_blue:
        args.color = 'blue'

    if args.lang_thai:
        args.lang = 'thai'
    elif args.lang_eng:
        args.lang = 'english'

    # ถ้าเลือก --camera หรือไม่ได้ระบุ path รูปภาพ ให้เปิดโหมดกล้อง Webcam
    if args.camera is not None or args.image is None:
        from camera_reader import RealTimeBrailleScanner
        cam_id = args.camera if args.camera is not None else 0
        scanner = RealTimeBrailleScanner(
            camera_id=cam_id,
            color=args.color,
            lang=args.lang,
            auto_speak=True,  # camera mode always enables auto-speak
        )
        scanner.run()
        return

    # ตรวจสอบไฟล์
    if not os.path.exists(args.image):
        print(f"  [ERR] File not found: {args.image}")
        sys.exit(1)

    # อ่านภาพ
    print(f"  อ่านภาพ:  {args.image}")
    print(f"  สีจุด:     {args.color}")
    print(f"  ภาษา:     {args.lang}")
    if args.speak:
        print(f"  เสียงพูด:  เปิดใช้งาน (TTS)")
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
    decoded_text = decode_cells(cells, lang=args.lang)
    verbose_results = decode_cells_verbose(cells, lang=args.lang)

    # อัปเดตภาพ Annotated ให้มีแบนเนอร์แสดงคำที่อ่านได้
    debug_info['annotated'] = detector.annotate_with_text(
        image, debug_info['dots'], cells,
        decoded_text=decoded_text, verbose_results=verbose_results, lang=args.lang
    )

    # แสดงผล
    print_results(cells, verbose_results, decoded_text)

    # ออกเสียงพูด (TTS)
    if args.speak and decoded_text:
        print("  🔊 กำลังออกเสียง...")
        from tts import speak
        base = os.path.splitext(os.path.basename(args.image))[0]
        audio_save_path = os.path.join('output', f'{base}_speech.mp3') if args.save else None
        speak(decoded_text, lang=args.lang, save_file=audio_save_path)
        if audio_save_path:
            print(f"  [SAVED] บันทึกไฟล์เสียง: {audio_save_path}")

    # บันทึก / แสดงภาพ
    if args.save:
        base = os.path.splitext(os.path.basename(args.image))[0]
        save_debug_images(debug_info, 'output', base)
        print()

    if not args.no_display:
        display_images(image, debug_info)


if __name__ == '__main__':
    main()
