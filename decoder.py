"""ถอดเซลล์เบรลล์ไทยและอังกฤษตามลำดับ โดยเก็บข้อมูลเซลล์ที่อ่านไม่ได้ไว้"""

from config import BRAILLE_TO_CHAR
from thai_decoder import _break_before, cell_read_warning, decode_thai, thai_verbose


_CAPITAL = frozenset({6}) #สัญลักษณ์ตัวพิมพ์ใหญ่ ตัวอักษรอังกฤษ 
_NUMBER = frozenset({3, 4, 5, 6}) #สัญลักษณ์ตัวเลข 
_LETTER_TO_DIGIT = dict(zip('abcdefghij', '1234567890')) 


def dots_to_braille_unicode(dots):
    return chr(0x2800 + sum(1 << (dot - 1) for dot in set(dots) if 1 <= dot <= 6)) #แสดงเซลล์ให้คนดู


def _english_tokens(cells):
    capitalize_next = number_mode = False # สถานะตัวนำหน้า 
    tokens = []
    for index, cell in enumerate(cells):
        dots = frozenset(cell['dots'])
        boundary = _break_before(cells, index) # ดูว่ามีช่องว่างคำก่อนเซลล์นี้ไหม
        previous = cells[index - 1] if index else {}
        line_break = (index > 0 and 'line_id' in previous and 'line_id' in cell
                      and previous['line_id'] != cell['line_id']) # ขึ้นบรรทัดใหม่
        if boundary:
            capitalize_next = number_mode = False # ขึ้นคำใหม่ 
        char, warning = '', None
        if cell_read_warning(cell): #กล้องอ่าน
            char, warning = '�', cell_read_warning(cell)
            capitalize_next = number_mode = False
        elif not dots:
            char = ' '
            capitalize_next = number_mode = False
        elif dots == _CAPITAL:
            capitalize_next = True
        elif dots == _NUMBER:
            number_mode = True
        else:
            char = BRAILLE_TO_CHAR.get(dots)
            if char is None:
                char = '[' + ','.join(map(str, sorted(dots))) + ']'
                warning = 'unknown_pattern'
                number_mode = False
            elif number_mode and char in _LETTER_TO_DIGIT:
                char = _LETTER_TO_DIGIT[char]
            elif number_mode and char in (',', '.'):
                pass
            else:
                number_mode = False
                if capitalize_next:
                    char = char.upper()
            capitalize_next = False
        tokens.append(dict(dots=sorted(dots), char=char,
                           braille_unicode=dots_to_braille_unicode(dots),
                           center=cell.get('center', (0, 0)), warning=warning,
                           break_before=boundary, line_break_before=line_break))
    return tokens


def decode_cells_english(cells):
    output = []
    for token in _english_tokens(cells):
        if token['line_break_before'] and output: # ขึ้นบรรทัดใหม่
            while output and output[-1] == ' ':
                output.pop()
            output.append('\n') #ลบช่องว่างท้าย 
        elif token['break_before'] and output and output[-1] not in (' ', '\n'):
            output.append(' ')  # ขึ้นคำใหม่
        char = token['char']
        if char and not (char == ' ' and (not output or output[-1] in (' ', '\n'))): # ไม่ใส่ช่องว่างซ้ำ
            output.append(char)
    return ''.join(output).strip()


def decode_cells(cells, lang='english', *, compatibility=False):
    language = lang.lower()
    if language in ('thai', 'th'):
        return decode_thai(cells, compatibility=compatibility)
    if language in ('english', 'en'):
        return decode_cells_english(cells)
    raise ValueError(f'Unsupported Braille language: {lang}')


def decode_cells_verbose(cells, lang='english', *, compatibility=False): # 'Thai' กับ 'thai'
    language = lang.lower()
    if language in ('thai', 'th'): 
        return thai_verbose(cells, compatibility=compatibility)
    if language in ('english', 'en'):
        return _english_tokens(cells)
    raise ValueError(f'Unsupported Braille language: {lang}')
