"""
Diagnostic script: ดู raw cell detection + decode ทีละ cell
"""
import cv2
import sys
sys.path.insert(0, '.')

from detector import BrailleDetector
from decoder import decode_cells, decode_cells_thai
from config_thai import (
    THAI_CONSONANTS, THAI_VOWELS, THAI_TONE_MARKS, THAI_SPECIAL_MARKS,
    THAI_CONSONANTS_PREFIX6, THAI_CONSONANTS_PREFIX36, THAI_CONSONANTS_PREFIX356,
    CONSONANT_KEYS, VOWEL_KEYS, TONE_KEYS, SPECIAL_KEYS,
    PREFIX_6, PREFIX_36, PREFIX_356, NUMBER_INDICATOR,
)

def diagnose(image_path, color='blue'):
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC: {image_path}")
    print(f"  Color: {color}")
    print(f"{'='*70}\n")

    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERR: Cannot read image")
        return

    detector = BrailleDetector(dot_color=color)
    cells, debug_info = detector.detect(image)

    if not cells:
        print("  No cells detected!")
        return

    print(f"  Total cells detected: {len(cells)}")
    print()

    for idx, cell in enumerate(cells):
        dots = cell['dots']
        dots_sorted = sorted(dots)
        dots_fs = frozenset(dots)

        # Determine type
        type_info = []
        if dots_fs in CONSONANT_KEYS:
            type_info.append(f"CONS: {THAI_CONSONANTS[dots_fs]}")
        if dots_fs in VOWEL_KEYS:
            type_info.append(f"VOWEL: {THAI_VOWELS[dots_fs]}")
        if dots_fs in TONE_KEYS:
            type_info.append(f"TONE: {THAI_TONE_MARKS[dots_fs]}")
        if dots_fs in SPECIAL_KEYS:
            type_info.append(f"SPECIAL: {THAI_SPECIAL_MARKS[dots_fs]}")
        if dots_fs == PREFIX_6:
            type_info.append("PREFIX_6")
        if dots_fs == PREFIX_36:
            type_info.append("PREFIX_36 (or ไม้โท ้)")
        if dots_fs == PREFIX_356:
            type_info.append("PREFIX_356 (or การันต์ ์)")
        if dots_fs == NUMBER_INDICATOR:
            type_info.append("NUMBER_INDICATOR")

        # Check if ambiguous
        is_ambiguous = len(type_info) > 1

        x = cell.get('x', 0)
        y = cell.get('y', 0)

        dots_str = str(dots_sorted)
        print(f"  C{idx+1:2d}  dots={dots_str:<16s}  x={x:5.0f} y={y:5.0f}  "
              f"{'⚠️ AMBIGUOUS' if is_ambiguous else ''}")
        for t in type_info:
            print(f"        -> {t}")
        print()

    # Decode result
    decoded = decode_cells(cells, lang='thai')
    print(f"  {'='*50}")
    print(f"  DECODED RESULT: \"{decoded}\"")
    print(f"  {'='*50}")

    # Show char-by-char
    print(f"\n  Character breakdown:")
    for i, ch in enumerate(decoded):
        code = ord(ch)
        import unicodedata
        name = unicodedata.name(ch, '?')
        print(f"    [{i:2d}] '{ch}' (U+{code:04X}) {name}")

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'sample_images/Test_thai_01.png'
    color = sys.argv[2] if len(sys.argv) > 2 else 'blue'
    diagnose(path, color)
