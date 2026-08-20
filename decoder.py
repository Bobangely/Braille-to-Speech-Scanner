"""
Braille Reader - Decoder
=========================
แปลง detected Braille cells (dot positions) เป็นตัวอักษร (English & Thai)
"""

from config import BRAILLE_TO_CHAR
from config_thai import (
    THAI_CONSONANTS,
    THAI_BRAILLE_TO_CHAR,
    THAI_CONSONANTS_PREFIX6,
    THAI_TONE_MARKS,
    THAI_DIGIT_MAP,
)


def decode_cells(cells, lang='english'):
    """
    แปลง list ของ Braille cells เป็นข้อความตามภาษาที่เลือก

    Parameters
    ----------
    cells : list of dict
        แต่ละ cell มี key 'dots' (frozenset ของ dot numbers 1-6)
    lang : str, optional
        ภาษา ('english' หรือ 'thai')

    Returns
    -------
    str
        ข้อความที่ถอดรหัสได้
    """
    if not cells:
        return ""

    lang_lower = lang.lower()
    if lang_lower in ('thai', 'th'):
        return decode_cells_thai(cells)
    else:
        return decode_cells_english(cells)


def decode_cells_english(cells):
    """
    แปลง Braille cells เป็นภาษาอังกฤษ (Grade 1)
    """
    letter_to_digit = {
        'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5',
        'f': '6', 'g': '7', 'h': '8', 'i': '9', 'j': '0'
    }

    result = []
    capitalize_next = False
    number_mode = False

    for i, cell in enumerate(cells):
        dots = cell['dots']

        # เช็คระยะห่างเพื่อแทรก space ระหว่างคำ
        if i > 0:
            prev_cell = cells[i - 1]
            prev_y = prev_cell.get('y', 0)
            curr_y = cell.get('y', 0)

            # ถ้าขึ้นบรรทัดใหม่
            if abs(curr_y - prev_y) > 30:
                if result and not result[-1].endswith(' '):
                    result.append(' ')
                number_mode = False
            else:
                # คำนวณระยะห่างแนวนอน
                dx = cell.get('x', 0) - prev_cell.get('x', 0)
                if dx > 180 and result and not result[-1].endswith(' '):
                    result.append(' ')
                    number_mode = False

        # 1. Capital indicator (จุด 6 เดี่ยวๆ)
        if dots == frozenset({6}):
            capitalize_next = True
            continue

        # 2. Number indicator (จุด 3,4,5,6)
        if dots == frozenset({3, 4, 5, 6}):
            number_mode = True
            continue

        # 3. ถอดรหัสตัวอักษร
        if dots in BRAILLE_TO_CHAR:
            ch = BRAILLE_TO_CHAR[dots]

            # แปลงเป็นตัวเลขถ้าอยู่ใน number mode
            if number_mode and ch in letter_to_digit:
                ch = letter_to_digit[ch]
            elif capitalize_next:
                ch = ch.upper()
                capitalize_next = False
            else:
                capitalize_next = False

            result.append(ch)
        else:
            # ไม่พบใน mapping -> แสดงเป็น dot pattern
            dot_list = sorted(dots)
            result.append(f'[{",".join(map(str, dot_list))}]')
            capitalize_next = False

    return ''.join(result).strip()


def decode_cells_thai(cells):
    """
    แปลง Braille cells เป็นภาษาไทย (Thai Braille Grade 1)
    รองรับ:
    - พยัญชนะ 1 เซลล์ (ก, ข, ค, ...)
    - พยัญชนะ 2 เซลล์ ที่มีจุด 6 นำหน้า (ฃ, ฆ, ฌ, ฎ, ฏ, ฐ, ฑ, ฒ, ณ, ถ, ธ, ผ, ฝ, ภ, ศ, ษ, ฬ)
    - สระหน้า, สระหลัง, สระบน, สระล่าง (เ, แ, โ, ไ, ใ, ะ, า, ิ, ี, ึ, ื, ุ, ู, ำ, ั, ็, ์, ๆ, ฯ)
    - วรรณยุกต์ (่, ้, ๊, ๋)
    - ตัวเลขไทย/อารบิก (เมื่อมีเครื่องหมาย # จุด 3,4,5,6 นำหน้า)
    """
    i = 0
    n = len(cells)
    result = []
    number_mode = False

    while i < n:
        cell = cells[i]
        dots = cell['dots']

        # เช็คระยะห่างเพื่อแทรก space
        if i > 0:
            prev_cell = cells[i - 1]
            prev_y = prev_cell.get('y', 0)
            curr_y = cell.get('y', 0)

            if abs(curr_y - prev_y) > 30:
                if result and not result[-1].endswith(' '):
                    result.append(' ')
                number_mode = False
            else:
                dx = cell.get('x', 0) - prev_cell.get('x', 0)
                if dx > 180 and result and not result[-1].endswith(' '):
                    result.append(' ')
                    number_mode = False

        # 1. Number indicator (จุด 3,4,5,6)
        if dots == frozenset({3, 4, 5, 6}):
            number_mode = True
            i += 1
            continue

        if number_mode:
            if dots in THAI_DIGIT_MAP:
                result.append(THAI_DIGIT_MAP[dots])
                i += 1
                continue
            else:
                number_mode = False

        # 2. จุด 6 (อาจเป็นจุดนำพยัญชนะ 2 เซลล์ หรือ ไม้โท)
        if dots == frozenset({6}):
            # ตรวจสอบ cell ถัดไปว่าคู่กับพยัญชนะ prefix 6 หรือไม่
            if i + 1 < n and cells[i + 1]['dots'] in THAI_CONSONANTS_PREFIX6:
                next_dots = cells[i + 1]['dots']
                ch = THAI_CONSONANTS_PREFIX6[next_dots]
                result.append(ch)
                i += 2  # ข้ามทั้ง 2 เซลล์
                continue
            else:
                # ไม่ใช่จุดนำพยัญชนะ -> เป็นไม้โท '้'
                result.append('้')
                i += 1
                continue

        # 3. วรรณยุกต์อื่นๆ (ไม้เอก, ไม้ตรี, ไม้จัตวา)
        if dots in THAI_TONE_MARKS and dots != frozenset({6}):
            result.append(THAI_TONE_MARKS[dots])
            i += 1
            continue

        # 4. พยัญชนะ / สระ (1 เซลล์)
        if dots in THAI_BRAILLE_TO_CHAR:
            result.append(THAI_BRAILLE_TO_CHAR[dots])
            i += 1
            continue

        # 6. Unknown dot pattern
        dot_list = sorted(dots)
        result.append(f'[{",".join(map(str, dot_list))}]')
        i += 1

    return ''.join(result).strip()


def dots_to_braille_unicode(dots):
    """
    แปลง dot positions เป็น Unicode Braille character
    Unicode Braille: U+2800 + (dot1*1 + dot2*2 + dot3*4 + dot4*8 + dot5*16 + dot6*32)

    Parameters
    ----------
    dots : frozenset of int
        เลข dot ที่มี (1-6)

    Returns
    -------
    str
        Braille Unicode character
    """
    offset = 0
    dot_to_bit = {1: 0x01, 2: 0x02, 3: 0x04, 4: 0x08, 5: 0x10, 6: 0x20}

    for d in dots:
        if d in dot_to_bit:
            offset |= dot_to_bit[d]

    return chr(0x2800 + offset)


def decode_cells_verbose(cells, lang='english'):
    """
    แปลง cells เป็นข้อความ พร้อมรายละเอียดของแต่ละ cell ตามภาษา

    Returns
    -------
    list of dict
        แต่ละ dict มี: 'dots', 'char', 'braille_unicode', 'center'
    """
    results = []
    is_thai = lang.lower() in ('thai', 'th')

    if is_thai:
        mapping = dict(THAI_BRAILLE_TO_CHAR)
        mapping.update(THAI_TONE_MARKS)
    else:
        mapping = BRAILLE_TO_CHAR

    for cell in cells:
        dots = cell['dots']
        char = mapping.get(dots, '?')
        braille_uni = dots_to_braille_unicode(dots)

        results.append({
            'dots': sorted(dots),
            'char': char,
            'braille_unicode': braille_uni,
            'center': cell.get('center', (0, 0)),
        })

    return results
