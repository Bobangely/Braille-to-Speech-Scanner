"""
Braille Reader - Decoder
=========================
แปลง detected Braille cells (dot positions) เป็นตัวอักษร
"""

from config import BRAILLE_TO_CHAR


def decode_cells(cells):
    """
    แปลง list ของ Braille cells เป็นข้อความ
    รองรับ:
    - Capital indicator (จุด 6 นำหน้าตัวอักษร) -> ตัวพิมพ์ใหญ่
    - Number indicator (จุด 3,4,5,6 นำหน้า a-j) -> ตัวเลข 1-0
    - เว้นวรรคตามระยะห่างระหว่างกลุ่มคำ

    Parameters
    ----------
    cells : list of dict
        แต่ละ cell มี key 'dots' (frozenset ของ dot numbers 1-6)

    Returns
    -------
    str
        ข้อความที่ถอดรหัสได้
    """
    if not cells:
        return ""

    # ตารางแปลงตัวเลข (เมื่อมี number indicator # นำหน้า)
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
                # ในภาพเบรลล์ทั่วไป ช่องว่างระหว่างคำจะกว้างกว่าช่องว่างระหว่าง cell ปกติ
                # ถ้า dx มากกว่าระยะปกติมากๆ (เช่น มี cell ว่างคั่น)
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


def decode_cells_verbose(cells):
    """
    แปลง cells เป็นข้อความ พร้อมรายละเอียดของแต่ละ cell

    Returns
    -------
    list of dict
        แต่ละ dict มี: 'dots', 'char', 'braille_unicode', 'center'
    """
    results = []

    for cell in cells:
        dots = cell['dots']
        char = BRAILLE_TO_CHAR.get(dots, '?')
        braille_uni = dots_to_braille_unicode(dots)

        results.append({
            'dots': sorted(dots),
            'char': char,
            'braille_unicode': braille_uni,
            'center': cell.get('center', (0, 0)),
        })

    return results
