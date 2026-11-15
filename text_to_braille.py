"""แปลงข้อความเป็นเซลล์เบรลล์ ใช้สร้างหนังสือ"""

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
                    # หลายเซลล์ (ตัวนำ + ตัวหลัก)
                    cells.extend(pattern)
                else:
                    cells.append(pattern)
    else: # ภาษาอังกฤษ
        for char in text.lower():
            if char == ' ':
                cells.append(None)
            elif char in CHAR_TO_BRAILLE:
                cells.append(CHAR_TO_BRAILLE[char])
    return cells
