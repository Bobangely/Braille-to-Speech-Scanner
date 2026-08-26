"""
Braille Reader - Decoder
=========================
แปลง detected Braille cells (dot positions) เป็นตัวอักษร (English & Thai)
ใช้ Context-Aware State Machine สำหรับภาษาไทย เพื่อแยกพยัญชนะ/สระ/วรรณยุกต์อย่างแม่นยำ
"""

from config import BRAILLE_TO_CHAR
from config_thai import (
    THAI_CONSONANTS,
    THAI_BRAILLE_TO_CHAR,
    THAI_CONSONANTS_PREFIX6,
    THAI_CONSONANTS_PREFIX36,
    THAI_CONSONANTS_PREFIX356,
    THAI_VOWELS,
    THAI_TONE_MARKS,
    THAI_SPECIAL_MARKS,
    THAI_DIGIT_MAP,
    CONSONANT_KEYS,
    VOWEL_KEYS,
    TONE_KEYS,
    SPECIAL_KEYS,
    LEADING_VOWELS,
    COMPOUND_VOWELS,
    COMBINING_VOWELS,
    PREFIX_6,
    PREFIX_36,
    PREFIX_356,
    NUMBER_INDICATOR,
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
                dx = cell.get('x', 0) - prev_cell.get('x', 0)
                if dx > 135 and result and not result[-1].endswith(' '):
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


# =============================================================================
# Thai Braille Decoder — Context-Aware State Machine
# =============================================================================

def _check_space(cells, i, result, number_mode):
    """
    ตรวจสอบว่าต้องแทรก space ระหว่าง cell ก่อนหน้ากับ cell ปัจจุบันไหม
    Returns: number_mode (อาจ reset เป็น False เมื่อเจอ space)
    """
    if i > 0:
        prev_cell = cells[i - 1]
        prev_y = prev_cell.get('y', 0)
        curr_y = cells[i].get('y', 0)

        if abs(curr_y - prev_y) > 30:
            if result and not result[-1].endswith(' '):
                result.append(' ')
            return False  # reset number mode on new line
        else:
            dx = cells[i].get('x', 0) - prev_cell.get('x', 0)
            if dx > 135 and result and not result[-1].endswith(' '):
                result.append(' ')
                return False  # reset number mode on space
    return number_mode


def _get_last_consonant(result):
    """
    ดึงพยัญชนะตัวสุดท้ายใน result (ถ้ามี) สำหรับการจัดเรียงสระนำ/สระผสม
    Returns: (index, consonant_char) หรือ (None, None)
    """
    if not result:
        return None, None

    # ค้นหาย้อนกลับจากท้าย result หา consonant ตัวล่าสุด
    all_consonants = set(THAI_CONSONANTS.values())
    all_consonants.update(THAI_CONSONANTS_PREFIX6.values())
    all_consonants.update(THAI_CONSONANTS_PREFIX36.values())
    all_consonants.update(THAI_CONSONANTS_PREFIX356.values())

    for idx in range(len(result) - 1, -1, -1):
        ch = result[idx]
        if ch in all_consonants:
            return idx, ch
        # ถ้าเจอ space หรือตัวอื่นที่ไม่ใช่ vowel/tone → หยุดค้นหา
        if ch == ' ':
            break
    return None, None


def decode_cells_thai(cells):
    """
    แปลง Braille cells เป็นภาษาไทย (Thai Braille Grade 1)
    ตามมาตรฐานอักษรเบรลล์ไทยสากล (Genevieve Caulfield / มูลนิธิช่วยคนตาบอดแห่งประเทศไทย)

    ใช้ Context-Aware State Machine:
    - State 'CONSONANT': คาดว่า cell ถัดไปเป็นพยัญชนะต้น
    - State 'VOWEL_TONE': เพิ่งเจอพยัญชนะ คาดว่าจะเป็นสระ/วรรณยุกต์
    """
    i = 0
    n = len(cells)
    result = []
    number_mode = False

    # State: 'CONSONANT' = กำลังรอพยัญชนะต้น, 'VOWEL_TONE' = รอสระ/วรรณยุกต์
    state = 'CONSONANT'

    while i < n:
        cell = cells[i]
        dots = cell['dots']

        # --- ตรวจ space ---
        number_mode = _check_space(cells, i, result, number_mode)
        # ถ้า result ลงท้ายด้วย space → reset state เป็น CONSONANT (ต้นคำใหม่)
        if result and result[-1] == ' ':
            state = 'CONSONANT'

        # ==============================================================
        # 1. Number indicator (จุด 3,4,5,6)
        # ==============================================================
        if dots == NUMBER_INDICATOR:
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

        # ==============================================================
        # 2. สระ 'ใ' (2 cells: dots 1,5,6 + dot 2)
        # ==============================================================
        if dots == frozenset({1, 5, 6}) and i + 1 < n and cells[i + 1]['dots'] == frozenset({2}):
            # สระ ใ เป็นสระนำ — ต้อง reorder ไปหน้าพยัญชนะ
            cons_idx, cons_ch = _get_last_consonant(result)
            if cons_idx is not None:
                result.insert(cons_idx, 'ใ')
            else:
                result.append('ใ')
            state = 'VOWEL_TONE'  # อาจตามด้วยวรรณยุกต์
            i += 2
            continue

        # ==============================================================
        # 3. Prefix consonants (2-cell consonants)
        # ==============================================================
        # Prefix 6
        if dots == PREFIX_6:
            if i + 1 < n and cells[i + 1]['dots'] in THAI_CONSONANTS_PREFIX6:
                next_dots = cells[i + 1]['dots']
                result.append(THAI_CONSONANTS_PREFIX6[next_dots])
                state = 'VOWEL_TONE'
                i += 2
                continue
            # ถ้าไม่มี match → ถือเป็น stand-alone (ในภาษาอังกฤษ = capital indicator, ไม่ใช้ในไทย)
            # fallthrough

        # Prefix 36
        if dots == PREFIX_36:
            if i + 1 < n and cells[i + 1]['dots'] in THAI_CONSONANTS_PREFIX36:
                next_dots = cells[i + 1]['dots']
                result.append(THAI_CONSONANTS_PREFIX36[next_dots])
                state = 'VOWEL_TONE'
                i += 2
                continue

        # Prefix 356
        if dots == PREFIX_356:
            if i + 1 < n and cells[i + 1]['dots'] in THAI_CONSONANTS_PREFIX356:
                next_dots = cells[i + 1]['dots']
                result.append(THAI_CONSONANTS_PREFIX356[next_dots])
                state = 'VOWEL_TONE'
                i += 2
                continue
            else:
                # Standalone 356 = การันต์ (์)
                result.append('์')
                i += 1
                continue

        # ==============================================================
        # 4. Context-Aware Resolution
        # ==============================================================

        # 4a. วรรณยุกต์ (Tone marks) — เสมอมาหลังพยัญชนะ/สระ
        if dots in TONE_KEYS:
            result.append(THAI_TONE_MARKS[dots])
            # state ยังเป็น VOWEL_TONE (อาจมี tone+special ต่อ)
            i += 1
            continue

        # 4b. เครื่องหมายพิเศษ (การันต์, ไม้ยมก, ไม้ไต่คู้)
        if dots in SPECIAL_KEYS:
            result.append(THAI_SPECIAL_MARKS[dots])
            i += 1
            continue

        # 4c. สระ — ถ้า state เป็น VOWEL_TONE หรือ dots อยู่ใน VOWEL_KEYS
        if dots in VOWEL_KEYS and state == 'VOWEL_TONE':
            raw_char = THAI_VOWELS[dots]
            _apply_vowel(result, raw_char)
            # state ยังเป็น VOWEL_TONE (อาจมีวรรณยุกต์ตาม)
            i += 1
            continue

        # 4d. พยัญชนะ — ถ้า state เป็น CONSONANT หรือ dots อยู่ใน CONSONANT_KEYS
        if dots in CONSONANT_KEYS:
            result.append(THAI_CONSONANTS[dots])
            state = 'VOWEL_TONE'
            i += 1
            continue

        # 4e. ถ้า state เป็น CONSONANT แต่ dots อยู่ใน VOWEL_KEYS (สระลอย เช่น ต้นคำ)
        if dots in VOWEL_KEYS:
            raw_char = THAI_VOWELS[dots]
            _apply_vowel(result, raw_char)
            state = 'VOWEL_TONE'
            i += 1
            continue

        # ==============================================================
        # 5. Fallback — ลองหาใน combined dict
        # ==============================================================
        if dots in THAI_BRAILLE_TO_CHAR:
            raw_char = THAI_BRAILLE_TO_CHAR[dots]
            result.append(raw_char)
            i += 1
            continue

        # 6. Unknown dot pattern
        dot_list = sorted(dots)
        result.append(f'[{",".join(map(str, dot_list))}]')
        state = 'CONSONANT'
        i += 1

    return ''.join(result).strip()


def _apply_vowel(result, raw_char):
    """
    จัดการสระ: สระนำ (reorder), สระผสม (decompose), สระ combining (append)

    Parameters
    ----------
    result : list of str
        ผลลัพธ์สะสม (mutable, แก้ไขโดยตรง)
    raw_char : str
        สระที่ได้จาก lookup (อาจเป็นสระนำ, สระผสม, หรือสระ combining)
    """
    # --- สระนำ (Leading Vowels): เ แ โ ไ ---
    if raw_char in LEADING_VOWELS:
        cons_idx, cons_ch = _get_last_consonant(result)
        if cons_idx is not None:
            result.insert(cons_idx, raw_char)
        else:
            # ไม่มีพยัญชนะก่อนหน้า → ใส่ 'อ' เป็น placeholder
            result.append(raw_char)
        return

    # --- สระผสม (Compound Vowels) ---
    if raw_char == 'เ◌า':
        cons_idx, cons_ch = _get_last_consonant(result)
        if cons_idx is not None:
            result.insert(cons_idx, 'เ')
            result.append('า')
        else:
            result.append('เอา')
        return

    if raw_char == 'เ◌ีย':
        cons_idx, cons_ch = _get_last_consonant(result)
        if cons_idx is not None:
            result.insert(cons_idx, 'เ')
            result.append('ีย')
        else:
            result.append('เอีย')
        return

    if raw_char == 'เ◌ือ':
        cons_idx, cons_ch = _get_last_consonant(result)
        if cons_idx is not None:
            result.insert(cons_idx, 'เ')
            result.append('ือ')
        else:
            result.append('เอือ')
        return

    if raw_char == 'เ◌อ':
        cons_idx, cons_ch = _get_last_consonant(result)
        if cons_idx is not None:
            result.insert(cons_idx, 'เ')
            result.append('อ')
        else:
            result.append('เออ')
        return

    if raw_char == 'เ◌ิ◌':
        cons_idx, cons_ch = _get_last_consonant(result)
        if cons_idx is not None:
            result.insert(cons_idx, 'เ')
            result.append('ิ')
        else:
            result.append('เอิ')
        return

    if raw_char == '◌ัว':
        cons_idx, cons_ch = _get_last_consonant(result)
        if cons_idx is not None:
            result.append('ัว')
        else:
            result.append('อัว')
        return

    # --- สระ combining ปกติ (ะ ั า ิ ี ึ ื ุ ู ำ) ---
    result.append(raw_char)


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
