"""เปลี่ยนเซลล์เบรลล์ไทยเป็นตัวอักษรทีละเซลล์ (lexer) ก่อนจัดพยางค์"""
from numbers import Integral

from config_thai import THAI_BRAILLE_TO_CHAR, THAI_READING_MULTI_CELL, THAI_DIGIT_MAP, NUMBER_INDICATOR
from thai_renderer import CONSONANTS, TONES, LEADING, normalize_vowel

# สระที่เขียนหลังพยัญชนะต้นเป็น 2 เซลล์
POST_ONSET_VOWELS = {
    ('124', '1'): 'เอะ', ('126', '1'): 'แอะ', ('24', '1'): 'โอะ',
    ('135', '1'): 'เอาะ', ('146', '1'): 'เออะ',
    ('12356', '1'): 'เอียะ', ('12345', '1'): 'เอือะ', ('15', '1'): 'อัวะ',
    ('26', '135'): 'อือ',
}


def _pattern(dots):#แปรงcellให้เรียงเลข {2, 1} -> '12'
    return ''.join(map(str, sorted(dots)))


def _descriptor(value):
    """บอกชนิดตัวอักษร พยัญชนะ สระนำหน้า วรรณยุกต์ สระ หรือข้อความอื่น"""
    canonical = normalize_vowel(value)
    if value in CONSONANTS: 
        kind = 'consonant'#พยัญชนะ
    elif value in LEADING: 
        kind = 'leading' #สระนำหน้า
    elif value in TONES:
        kind = 'tone' #วรรณยุกต์
    elif canonical is not None:
        kind = 'vowel'#สระ
    else:
        kind = 'literal'
    return dict(kind=kind, value=value, vowel=canonical,
                semantic=kind == 'vowel' and len(value) > 1)


# สร้างครั้งเดียวตอน import: รูปเซลล์ -> ชนิดตัวอักษร
SINGLE_CELL = {_pattern(dots): _descriptor(value) for dots, value in THAI_BRAILLE_TO_CHAR.items()}
MULTI_CELL = {tuple(_pattern(dots) for dots in sequence): _descriptor(value)
              for sequence, value in THAI_READING_MULTI_CELL.items()}
_LONGEST_FIRST = sorted(MULTI_CELL, key=len, reverse=True) #เรียงตารางหลายเซลล์


def match_post_onset_vowel(atoms, index):
    """สระ 2 เซลล์หลังพยัญชนะต้น คืน (สระ, จำนวนเซลล์) หรือ None """
    for sequence in sorted(POST_ONSET_VOWELS, key=len, reverse=True):
        following = atoms[index:index+len(sequence)]
        if (len(following) == len(sequence)
                and all(a.get('patterns') == [p] for a, p in zip(following, sequence)) #จุดตรง
                and all(a.get('adjacent_before', True) for a in following[1:])): #ติดกัน
            return POST_ONSET_VOWELS[sequence], len(sequence) # (สระ, จำนวนเซลล์)
    return None


def is_unread_cell(cell): #กล้องเห็นเซลล์ แต่อ่านจุดไม่ได้
    return not cell.get('dots') and cell.get('crop_status') in ('empty', 'invalid_grid') #แยกเซลล์ที่อ่านไม่ได้ออกจากช่องว่าง


def cell_read_warning(cell):
    """คำเตือนอ่านภาพ  None ถ้าเซลล์อ่านได้ปกติ"""
    if cell.get('geometry_status') == 'invalid' or cell.get('crop_status') == 'invalid_grid':
        return 'invalid_cell_grid' #ตารางไม่ถูก
    if is_unread_cell(cell):
        return 'empty_crop' #ว่าง
    if cell.get('row_ambiguous'):
        return 'ambiguous_row_grid' #ตารางแถวไม่ชัดเจน    
    return None


def _break_before(cells, index, adjacent=False):
    """มีเว้นคำก่อนเซลล์นี้ไหม: เซลล์ว่าง ขึ้นบรรทัดใหม่ หรืออยู่ห่างกันเกิน"""
    if index == 0:
        return False
    left, right = cells[index - 1:index + 1]
    if right.get('break_before') or right.get('line_break_before'):
        return True
    if any(not cell.get('dots') and not is_unread_cell(cell) for cell in (left, right)):
        return True
    if 'line_id' in left and 'line_id' in right and left['line_id'] != right['line_id']:
        return True
    spacing = 20.0  # ระยะอ้างอิงระหว่างเซลล์ ( x, y)
    if abs(right.get('y', 0) - left.get('y', 0)) > spacing * 1.5: #30
        return True
    if 'x' not in left or 'x' not in right:
        return False  # เซลล์ที่เรียงลำดับแล้วไม่ต้องใช้พิกัดภาพ
    dx = right['x'] - left['x'] #ถอยหลัง
    return dx < 0 or dx > spacing * (3.4 if adjacent else 6.5) #ห่างเกิน


def snapshot_cells(cells):
    """เก็บรูปเซลล์และขอบคำของแต่ละเซลล์ไว้ก่อนตีความ"""
    raw = []
    for index, cell in enumerate(cells):
        values = list(cell.get('dots', ()))
        # จุดต้องเป็นเลข 1-6 ไม่ซ้ำ
        valid = ('dots' in cell and all(isinstance(d, Integral) and 1 <= d <= 6 for d in values)
                 and len(values) == len(set(values))) #ไม่มีจุดซ้ำ
        dots = sorted(values) if valid else values
        previous_line = cells[index-1].get('line_id') if index else None
        raw.append(dict(dots=dots, pattern=''.join(map(str, dots)), line_id=cell.get('line_id'),
                        break_before=_break_before(cells, index), #ขึ้นคำใหม่
                        adjacent_before=not _break_before(cells, index, adjacent=True), #ติดกับตัวหน้า
                        line_break_before=(bool(cell.get('line_break_before')) or #ขึ้นบรรทัดใหม่
                                           (previous_line is not None and cell.get('line_id') is not None
                                            and previous_line != cell['line_id'])),
                        warning=cell_read_warning(cell) if valid else 'invalid_raw_pattern'))
    return raw


def lex_braille(cells):
    """เซลล์ -> (raw, tokens, atoms)

    raw = ข้อมูลเซลล์, tokens = ตัวอักษรรายเซลล์ -> page_map,
    atoms = หน่วยพร้อมชนิดตัวอักษรสำหรับจัดพยางค์
    """
    raw = snapshot_cells(cells)
    tokens, atoms = [], []
    index, numeric = 0, False  #โหมดตัวเลข
    while index < len(raw):
        cell = raw[index]
        pattern = cell['pattern']
        if cell['break_before'] or not pattern: #คำใหม่หรือช่องว่าง
            numeric = False
        span, char, warning, descriptor = 1, '', cell['warning'], None
        if warning: #เซลเสีย
            char, numeric = '�', False
        elif not pattern:
            char = ' '
        elif frozenset(cell['dots']) == NUMBER_INDICATOR: #เซลน์นำตัวเลข
            numeric = True
        elif numeric and frozenset(cell['dots']) in THAI_DIGIT_MAP:
            char = THAI_DIGIT_MAP[frozenset(cell['dots'])]
        else:
            numeric = False
            # ลองตัวอักษรหลายเซลล์ก่อน 
            for sequence in _LONGEST_FIRST:
                following = raw[index:index+len(sequence)]
                if (tuple(c['pattern'] for c in following) == sequence
                        and not any(c['warning'] for c in following)
                        and all(c['adjacent_before'] for c in following[1:])):
                    descriptor = dict(MULTI_CELL[sequence])
                    char, span = descriptor['value'], len(sequence)
                    break
            if descriptor is None and pattern in SINGLE_CELL:
                descriptor = dict(SINGLE_CELL[pattern])
                char = descriptor['value']
                if pattern == '135': #อ
                    descriptor['kind'] = 'ambiguous_o'  #อ เป็นได้ทั้งพยัญชนะและสระ ออ
                elif pattern == '3': # ( ็ ) 
                    descriptor['kind'] = 'ambiguous_mark'  # ไม้ไต่คู้ ต้องดูบริบท
            elif descriptor is None:
                char, warning = '�', 'unknown_or_incomplete_symbol'
        if warning:
            descriptor = dict(kind='error', value='�', vowel=None, semantic=False)
        elif descriptor is None:
            descriptor = dict(kind='literal', value=char, vowel=None, semantic=False)
        source_indices = list(range(index, index+span))
        atoms.append(dict(descriptor, index=index, source_indices=source_indices, #ดูว่าใช้เซลล์ไหน
                          patterns=[raw[i]['pattern'] for i in source_indices],
                          adjacent_before=cell['adjacent_before'])) #ดูว่าติดกับตัวก่อนหน้า
        atoms.extend([None] * (span-1))
        for offset in range(span):
            first = offset == 0
            # เซลล์ที่ 2 ขึ้นไปของตัวอักษรเดียวกัน consumed=True และไม่มีตัวอักษรของตัวเอง
            tokens.append(dict(char=char if first else '', cell_start=index, cell_end=index+span-1,
                               consumed=not first, warning=warning, line_id=raw[index+offset]['line_id'],
                               break_before=cell['break_before'] if first else False,
                               line_break_before=cell['line_break_before'] if first else False))
        index += span #ข้ามเซลล์ที่ใช้ไป
    return raw, tokens, atoms
