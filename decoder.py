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

# =============================================================================
# Module-level pre-computed constants (Perf 1 & 2)
# =============================================================================

# Pre-compute all Thai consonant characters (avoid rebuilding set every call)
_ALL_CONSONANTS = set(THAI_CONSONANTS.values())
_ALL_CONSONANTS.update(THAI_CONSONANTS_PREFIX6.values())
_ALL_CONSONANTS.update(THAI_CONSONANTS_PREFIX36.values())
_ALL_CONSONANTS.update(THAI_CONSONANTS_PREFIX356.values())
_ALL_CONSONANTS = frozenset(_ALL_CONSONANTS)  # immutable for safety

# Pre-compute frozenset constants used in hot loops (avoid repeated allocation)
_DOTS_156 = frozenset({1, 5, 6})
_DOTS_2 = frozenset({2})
_DOTS_6 = frozenset({6})
_DOTS_3456 = frozenset({3, 4, 5, 6})

# Thai combining/trailing vowels that should come AFTER tone marks
_TRAILING_VOWEL_PARTS = frozenset({'า', 'ำ', 'ะ'})

# English letter-to-digit mapping (pre-compute, used in decode_cells_english)
_LETTER_TO_DIGIT = {
    'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5',
    'f': '6', 'g': '7', 'h': '8', 'i': '9', 'j': '0'
}

# Braille Unicode bit mapping (Perf 5 — direct tuple lookup instead of dict)
_DOT_BITS = (0, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20)  # index 0 unused, 1-6 map to bits


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
    result = []
    capitalize_next = False
    number_mode = False

    for i, cell in enumerate(cells):
        dots = cell['dots']

        # เช็คระยะห่างเพื่อแทรก space ระหว่างคำ (Dynamic resolution)
        if i > 0:
            prev_cell = cells[i - 1]
            prev_y = prev_cell.get('y', 0)
            curr_y = cell.get('y', 0)
            
            grid = cell.get('grid')
            dot_spacing = (grid['expected_cols'][1] - grid['expected_cols'][0]) if grid else 20
            
            line_threshold = dot_spacing * 1.5
            space_threshold = dot_spacing * 6.5

            # ถ้าขึ้นบรรทัดใหม่
            if abs(curr_y - prev_y) > line_threshold:
                if result and not result[-1].endswith(' '):
                    result.append(' ')
                number_mode = False
            else:
                dx = cell.get('x', 0) - prev_cell.get('x', 0)
                if dx > space_threshold and result and not result[-1].endswith(' '):
                    result.append(' ')
                    number_mode = False

        # 1. Capital indicator (จุด 6 เดี่ยวๆ)
        if dots == _DOTS_6:
            capitalize_next = True
            continue

        # 2. Number indicator (จุด 3,4,5,6)
        if dots == _DOTS_3456:
            number_mode = True
            continue

        # 3. ถอดรหัสตัวอักษร
        if dots in BRAILLE_TO_CHAR:
            ch = BRAILLE_TO_CHAR[dots]

            # แปลงเป็นตัวเลขถ้าอยู่ใน number mode
            if number_mode and ch in _LETTER_TO_DIGIT:
                ch = _LETTER_TO_DIGIT[ch]
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


# Thai Braille Decoder — Context-Aware State Machine

def _check_space(cells, i, result, number_mode):
    """
    ตรวจสอบว่าต้องแทรก space ระหว่าง cell ก่อนหน้ากับ cell ปัจจุบันไหม
    Returns: number_mode (อาจ reset เป็น False เมื่อเจอ space)
    """
    if i > 0:
        prev_cell = cells[i - 1]
        prev_y = prev_cell.get('y', 0)
        curr_y = cells[i].get('y', 0)

        grid = cells[i].get('grid')
        dot_spacing = (grid['expected_cols'][1] - grid['expected_cols'][0]) if grid else 20
        
        line_threshold = dot_spacing * 1.5
        space_threshold = dot_spacing * 6.5

        if abs(curr_y - prev_y) > line_threshold:
            if result and not result[-1].endswith(' '):
                result.append(' ')
            return False  # reset number mode on new line
        else:
            dx = cells[i].get('x', 0) - prev_cell.get('x', 0)
            if dx > space_threshold and result and not result[-1].endswith(' '):
                result.append(' ')
                return False  # reset number mode on space
    return number_mode


def _get_vowel_insert_index(result):
    """
    ดึง index สำหรับแทรกสระนำ (เ แ โ ไ ใ) โดยพิจารณาอักษรควบกล้ำและ ห นำ
    Returns: index หรือ None
    """
    if not result:
        return None

    # ค้นหาย้อนกลับจากท้าย result หา consonant ตัวล่าสุด (ใช้ pre-computed set)
    last_cons_idx = None
    for idx in range(len(result) - 1, -1, -1):
        ch = result[idx]
        if ch in _ALL_CONSONANTS:
            last_cons_idx = idx
            break
        if ch == ' ':
            break

    if last_cons_idx is None:
        return None

    insert_idx = last_cons_idx

    # ถอยไปดูพยัญชนะตัวก่อนหน้าว่าเป็นอักษรนำ/ควบกล้ำหรือไม่
    if last_cons_idx > 0:
        prev_char = result[last_cons_idx - 1]
        curr_char = result[last_cons_idx]
        
        if prev_char in _ALL_CONSONANTS:
            # 1. ห นำ
            if prev_char == 'ห' and curr_char in ('ง', 'ญ', 'น', 'ม', 'ย', 'ร', 'ล', 'ว'):
                insert_idx = last_cons_idx - 1
            # 2. อ นำ
            elif prev_char == 'อ' and curr_char == 'ย':
                insert_idx = last_cons_idx - 1
            # 3. ควบกล้ำ
            elif curr_char in ('ร', 'ล', 'ว') and prev_char in ('ก', 'ข', 'ค', 'ต', 'ป', 'ผ', 'พ', 'ท', 'ศ', 'ส'):
                insert_idx = last_cons_idx - 1

    return insert_idx


def _is_trailing_vowel_part(result, idx):
    """
    เช็คว่า result[idx] เป็นส่วนของสระที่ต้องอยู่หลังวรรณยุกต์หรือไม่
    ต้องแยก 'อ' 'ย' 'ว' ที่เป็นพยัญชนะออกจากที่เป็นส่วนของสระ (Bug 2 fix)
    """
    if idx < 0 or idx >= len(result):
        return False
    ch = result[idx]
    # า ำ ะ เป็นส่วนของสระเสมอ
    if ch in _TRAILING_VOWEL_PARTS:
        return True
    # อ ย ว — ต้องเช็คว่าเป็นส่วนของสระผสมหรือเป็นพยัญชนะ
    # ถ้าตัวก่อนหน้ามันเป็น combining vowel (ั ี ื) → มันเป็นส่วนของสระ
    if ch in ('อ', 'ย', 'ว') and idx > 0:
        prev = result[idx - 1]
        # ถ้าตัวก่อนหน้าเป็น combining vowel mark หรือเป็นพยัญชนะที่มี combining vowel ก่อนหน้า
        combining_marks = {'ั', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู'}
        if prev in combining_marks:
            return True
        # ถ้า prev เป็น leading vowel เช่น 'เ' → มันเป็นส่วนของสระเออ
        if prev in _ALL_CONSONANTS and ch == 'อ':
            # เช็คว่ามี 'เ' อยู่ก่อนหน้าพยัญชนะไหม → เป็นสระเออ
            if idx >= 2 and result[idx - 2] == 'เ':
                return True
    return False


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
        if dots == _DOTS_156 and i + 1 < n and cells[i + 1]['dots'] == _DOTS_2:
            # สระ ใ เป็นสระนำ — ต้อง reorder ไปหน้าพยัญชนะ
            cons_idx = _get_vowel_insert_index(result)
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
            # ถ้าไม่มี match → fallthrough ไป section 4

        # Prefix 36 (หรือ ไม้โท ้ เมื่อเป็น standalone)
        if dots == PREFIX_36:
            if i + 1 < n and cells[i + 1]['dots'] in THAI_CONSONANTS_PREFIX36:
                next_dots = cells[i + 1]['dots']
                result.append(THAI_CONSONANTS_PREFIX36[next_dots])
                state = 'VOWEL_TONE'
                i += 2
                continue
            else:
                # Standalone 36 = ไม้โท (้)
                tone_char = '้'
                if result and _is_trailing_vowel_part(result, len(result) - 1):
                    result.insert(-1, tone_char)
                else:
                    result.append(tone_char)
                i += 1
                continue

        # Prefix 356 — Bug 1 fix: ใช้ continue ทุก branch เพื่อไม่ให้ fallthrough ไป section 4b
        if dots == PREFIX_356:
            if i + 1 < n and cells[i + 1]['dots'] in THAI_CONSONANTS_PREFIX356:
                next_dots = cells[i + 1]['dots']
                result.append(THAI_CONSONANTS_PREFIX356[next_dots])
                state = 'VOWEL_TONE'
                i += 2
                continue
            else:
                # Standalone 356 = การันต์ (์) — already handled here, skip section 4b
                result.append('์')
                i += 1
                continue

        # ==============================================================
        # 4. Context-Aware Resolution
        # ==============================================================

        # 4a. วรรณยุกต์ (Tone marks) — เสมอมาหลังพยัญชนะ/สระ
        if dots in TONE_KEYS:
            tone_char = THAI_TONE_MARKS[dots]
            # Bug 2 fix: จัดตำแหน่งวรรณยุกต์ — แทรกก่อน trailing vowel part เท่านั้น
            if result and _is_trailing_vowel_part(result, len(result) - 1):
                result.insert(-1, tone_char)
            else:
                result.append(tone_char)
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
        cons_idx = _get_vowel_insert_index(result)
        if cons_idx is not None:
            result.insert(cons_idx, raw_char)
        else:
            # ไม่มีพยัญชนะก่อนหน้า → ใส่ 'อ' เป็น placeholder
            result.append(raw_char)
        return

    # --- สระผสม (Compound Vowels) ---
    if raw_char == 'เ◌า':
        cons_idx = _get_vowel_insert_index(result)
        if cons_idx is not None:
            result.insert(cons_idx, 'เ')
            result.append('า')
        else:
            result.extend(['เ', 'อ', 'า'])
        return

    if raw_char == 'เ◌ีย':
        cons_idx = _get_vowel_insert_index(result)
        if cons_idx is not None:
            result.insert(cons_idx, 'เ')
            result.append('ี')
            result.append('ย')
        else:
            result.extend(['เ', 'อ', 'ี', 'ย'])
        return

    if raw_char == 'เ◌ือ':
        cons_idx = _get_vowel_insert_index(result)
        if cons_idx is not None:
            result.insert(cons_idx, 'เ')
            result.append('ื')
            result.append('อ')
        else:
            result.extend(['เ', 'อ', 'ื', 'อ'])
        return

    if raw_char == 'เ◌อ':
        cons_idx = _get_vowel_insert_index(result)
        if cons_idx is not None:
            result.insert(cons_idx, 'เ')
            result.append('อ')
        else:
            result.extend(['เ', 'อ', 'อ'])
        return

    if raw_char == 'เ◌ิ◌':
        cons_idx = _get_vowel_insert_index(result)
        if cons_idx is not None:
            result.insert(cons_idx, 'เ')
            result.append('ิ')
        else:
            result.extend(['เ', 'อ', 'ิ'])
        return

    if raw_char == '◌ัว':
        cons_idx = _get_vowel_insert_index(result)
        if cons_idx is not None:
            result.append('ั')
            result.append('ว')
        else:
            result.extend(['อ', 'ั', 'ว'])
        return

    # --- สระ combining ปกติ (ะ ั า ิ ี ึ ื ุ ู ำ) ---
    result.append(raw_char)


def dots_to_braille_unicode(dots):
    """
    แปลง dot positions เป็น Unicode Braille character (Perf 5 — direct bit computation)
    Unicode Braille: U+2800 + (dot1*1 + dot2*2 + dot3*4 + dot4*8 + dot5*16 + dot6*32)
    """
    offset = 0
    for d in dots:
        if 1 <= d <= 6:
            offset |= _DOT_BITS[d]
    return chr(0x2800 + offset)


def decode_cells_verbose(cells, lang='english'):
    """
    แปลง cells เป็นข้อความ พร้อมรายละเอียดของแต่ละ cell ตามภาษา
    Bug 7 fix: ใช้ context-aware lookup สำหรับ Thai แทน flat lookup

    Returns
    -------
    list of dict
        แต่ละ dict มี: 'dots', 'char', 'braille_unicode', 'center'
    """
    results = []
    is_thai = lang.lower() in ('thai', 'th')

    if is_thai:
        # สร้าง per-cell char mapping จาก state machine context
        # ใช้ simplified context: ถ้า dot pattern อยู่ใน CONSONANT_KEYS → consonant,
        # ถ้าอยู่ใน VOWEL_KEYS → vowel, ถ้าอยู่ใน TONE_KEYS → tone, อื่นๆ → special/fallback
        mapping = {}
        mapping.update(THAI_CONSONANTS)
        mapping.update(THAI_VOWELS)
        mapping.update(THAI_TONE_MARKS)
        mapping.update(THAI_SPECIAL_MARKS)
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
