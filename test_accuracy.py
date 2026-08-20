"""
Braille Reader - Automated Accuracy & Regression Test Suite
============================================================
ทดสอบความแม่นยำของระบบอ่านอักษรเบรลล์ทั้งภาษาอังกฤษและภาษาไทย
คำนวณ Character Accuracy และ Sentence Accuracy อัตโนมัติ
"""

import os
import sys
import cv2

# ปรับ encoding สำหรับ Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from detector import BrailleDetector
from decoder import decode_cells


# รายการภาพทดสอบและ Ground Truth
TEST_DATASET = [
    # (Image file, dot color, language, Ground Truth)
    # --- English Test Set ---
    ('sample_images/test_hello_blue.png', 'blue', 'english', 'hello'),
    ('sample_images/test_abc_blue.png', 'blue', 'english', 'abc'),
    ('sample_images/test_world_red.png', 'red', 'english', 'world'),
    ('sample_images/test_braille_green.png', 'green', 'english', 'braille'),
    ('sample_images/test_hello_noisy.png', 'blue', 'english', 'hello'),
    ('sample_images/test_alphabet.png', 'blue', 'english', 'abcdefghij'),
    ('sample_images/test_python.png', 'blue', 'english', 'python'),

    # --- Thai Test Set ---
    ('sample_images/test_thai_ka.png', 'blue', 'thai', 'กา'),
    ('sample_images/test_thai_thai.png', 'blue', 'thai', 'ไทย'),
    ('sample_images/test_thai_khon.png', 'red', 'thai', 'คน'),
    ('sample_images/test_thai_cat.png', 'green', 'thai', 'แมว'),
    ('sample_images/test_thai_home.png', 'blue', 'thai', 'บ้าน'),
    ('sample_images/test_thai_consonants.png', 'blue', 'thai', 'กขคงจ'),
]


def run_benchmark():
    print("=" * 70)
    print("   Braille Reader - Accuracy Benchmark & Test Suite")
    print("=" * 70)
    print()

    total_tests = len(TEST_DATASET)
    passed_tests = 0
    total_chars = 0
    correct_chars = 0

    print(f"{'#':<3} {'Test Image':<32} {'Lang':<8} {'Expected':<12} {'Actual':<12} {'Status'}")
    print("-" * 75)

    for idx, (img_path, color, lang, expected) in enumerate(TEST_DATASET, 1):
        if not os.path.exists(img_path):
            print(f"{idx:<3} {os.path.basename(img_path):<32} {lang:<8} {expected:<12} {'[MISSING]':<12} ❌ FILE NOT FOUND")
            continue

        img = cv2.imread(img_path)
        detector = BrailleDetector(dot_color=color)
        cells, _ = detector.detect(img)
        actual = decode_cells(cells, lang=lang)

        # คำนวณความถูกต้อง
        is_exact_match = (actual == expected)
        if is_exact_match:
            passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        # นับจำนวนตัวอักษรที่ตรงกัน
        total_chars += len(expected)
        matched_c = sum(1 for a, b in zip(actual, expected) if a == b)
        correct_chars += matched_c

        exp_disp = expected if len(expected) <= 10 else expected[:8] + ".."
        act_disp = actual if len(actual) <= 10 else actual[:8] + ".."
        print(f"{idx:<3} {os.path.basename(img_path):<32} {lang:<8} {exp_disp:<12} {act_disp:<12} {status}")

    print("-" * 75)
    sentence_acc = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    char_acc = (correct_chars / total_chars) * 100 if total_chars > 0 else 0

    print()
    print("📊 สรุปผลการทดสอบ:")
    print(f"  • ภาพที่ทดสอบทั้งหมด:    {total_tests} ภาพ")
    print(f"  • ผ่านแบบ 100% (Exact):  {passed_tests}/{total_tests} ภาพ ({sentence_acc:.1f}%)")
    print(f"  • ความถูกต้องระดับตัวอักษร: {correct_chars}/{total_chars} ตัว ({char_acc:.1f}%)")
    print()

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! ระบบพร้อมใช้งาน 100%")
    else:
        print(f"⚠️ มีการทดสอบที่ไม่ผ่าน {total_tests - passed_tests} รายการ")

    return passed_tests == total_tests


if __name__ == '__main__':
    success = run_benchmark()
    sys.exit(0 if success else 1)
