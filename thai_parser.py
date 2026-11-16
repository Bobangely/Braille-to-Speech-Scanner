"""แยกลำดับ token เป็นพยางค์ไทยก่อนแปลงเป็นตัวสะกด"""

from collections import Counter
from copy import deepcopy

from thai_renderer import (CONSONANTS, TONES, LEADING, SHORT, INHERENT_FINAL,
                           CODA_CHANGES_VOWEL, RULES, normalize_vowel)
from thai_braille import lex_braille, match_post_onset_vowel
from thai_raw_rules import raw_encoding_error


def _is_consonant(atom): #นับ อ ที่กำกวมเป็นพยัญชนะด้วย
    return atom['kind'] in ('consonant', 'ambiguous_o')


def _source_indices(atom):
    """ตำแหน่งเซลล์ต้นทางของ atom """
    return atom.get('source_indices', [atom['index']])


def _component(kind, value, atoms, rule):
    """บันทึกส่วนพยางค์ พร้อมเซลล์ที่ใช้ตรวจรูปเซลล์"""
    return dict(type=kind, value=value, rule=rule,
                indices=[i for atom in atoms for i in _source_indices(atom)],
                patterns=[p for atom in atoms for p in atom.get('patterns', [])])


def _cluster(left, right): #พยัญชนะควบกล้ำ
    return ((left == 'ห' and right in 'งญนมยรลว') or (left == 'อ' and right == 'ย')
            or (left in 'กขคตปผพทศส' and right in 'รลว')) #พยัญชนะควบแท้


def _is_written_o(atoms, index):
    """อ เป็นสระ ออ เมื่อไม่มีสระตามหลัง"""
    if not _is_consonant(atoms[index]) or atoms[index]['value'] != 'อ':
        return False
    end = index + 1
    while end < len(atoms) and atoms[end]['kind'] == 'tone':
        end += 1
    if end == len(atoms):
        return True
    if atoms[end]['kind'] == 'vowel' or atoms[end]['value'] == 'อ':
        return False  # อิ ออ ฯลฯ เริ่มพยางค์ใหม่ เช่น เต็มอิ่ม
    if atoms[end]['kind'] == 'leading':
        return end+1 < len(atoms) and _is_consonant(atoms[end+1])
    return True


def _starts_syllable(atoms, index, *, onset_context=False):
    """พยัญชนะที่มีสระของตัวเองตามหลังเป็น ต้นพยางค์ใหม่"""
    end = index + 1
    if end < len(atoms) and _is_consonant(atoms[end]) and _cluster(atoms[index]['value'], atoms[end]['value']):
        end += 1
    while end < len(atoms) and atoms[end]['kind'] == 'tone':
        end += 1
    if end >= len(atoms):
        return False
    if atoms[end]['kind'] == 'leading': # แ + ม เริ่มพยางค์ใหม่ ไม่ใช่สระของ ณ > คุณแม่
        return end+1 >= len(atoms) or not _is_consonant(atoms[end+1])
    # อ เป็นสระหลังพยัญชนะต้นได้ (กล + ่ + อ + ง)
    return (atoms[end]['kind'] in ('vowel', 'ambiguous_mark') or _is_written_o(atoms, end)
            or (onset_context and match_post_onset_vowel(atoms, end) is not None))


def _implicit_closed_span(atoms, index): 
    """พยัญชนะ 2 ตัวไม่มีสระ เป็น โอะลดรูป พยัญชนะต้น + ตัวสะกด return (จบต้น, ตำแหน่งตัวสะกด, จบพยางค์) 
    หรือ None หลังตัวสะกดต้องจบคำหรือเป็นพยางค์สมบูรณ์ 
    """
    if (index >= len(atoms) or not _is_consonant(atoms[index])
            or _starts_syllable(atoms, index, onset_context=True)):
        return None
    onset_end = index + 1
    if (onset_end+1 < len(atoms) and _is_consonant(atoms[onset_end])
            and _is_consonant(atoms[onset_end+1])
            and _cluster(atoms[index]['value'], atoms[onset_end]['value'])
            and not _starts_syllable(atoms, onset_end, onset_context=True)):
        onset_end += 1
    final_index = onset_end
    while final_index < len(atoms) and atoms[final_index]['kind'] == 'tone':
        final_index += 1
    if (final_index >= len(atoms) or not _is_consonant(atoms[final_index])
            or atoms[final_index]['value'] == 'อ'
            or _starts_syllable(atoms, final_index, onset_context=True)):
        return None
    end = final_index + 1
    # พยัญชนะที่มีวรรณยุกต์ตามหลังไม่ใช่ตัวสะกด กล้วย ไม่ใช่ ก้ล + วย
    if end < len(atoms) and atoms[end]['kind'] == 'tone':
        return None
    if end < len(atoms):
        preposed = (atoms[end]['kind'] == 'leading' and end+1 < len(atoms)
                    and _is_consonant(atoms[end+1]))
        explicit = _is_consonant(atoms[end]) and _starts_syllable(atoms, end, onset_context=True)
        if not (preposed or explicit or _implicit_closed_span(atoms, end) is not None):
            return None
    return onset_end, final_index, end


def _structural_error(units, source_indices=None):
    """ตรวจโครงสร้างการแยกที่ครบ โดยไม่แปลงเป็นข้อความและไม่ใช้พจนานุกรม"""
    if source_indices is not None:
        expected = Counter(source_indices)
        actual = Counter(i for unit in units for i in unit['indices'])
        if any(actual[i] > expected[i] for i in expected):
            return 'REUSED_SOURCE_CELL'
        if actual.keys() - expected.keys():
            return 'INVALID_SOURCE_CELLS'
        if actual != expected:
            return 'INCOMPLETE_PARSE'
    for unit in units:
        structural_status = unit.get('structural_status', unit.get('status'))
        if structural_status:
            return structural_status
        if 'literal' in unit: #เซลล์ซ้ำกับ สระมีตัวสะกดไม่ได้	
            if unit['literal'] == '�':# เครื่องหมายวรรคตอนเป็นข้อความ
                return 'UNKNOWN_TOKEN'
            if (unit['literal'] in CONSONANTS or unit['literal'] in TONES or unit['literal'] in LEADING
                    or normalize_vowel(unit['literal']) is not None):
                return 'INVALID_SYLLABLE'
            continue
        vowel = unit.get('vowel')
        if vowel is None:
            return 'UNRESOLVED_NUCLEUS'
        if (not unit.get('onset') or any(c not in CONSONANTS for c in unit['onset'])
                or (len(unit['onset']) > 1 and
                    (len(unit['onset']) != 2 or not _cluster(*unit['onset'])))
                or (unit.get('final') and (len(unit['final']) != 1 or unit['final'] not in CONSONANTS))):
            return 'INVALID_SYLLABLE'
        if vowel not in RULES:
            return 'INVALID_VOWEL'
        template = RULES[vowel][bool(unit.get('final'))]
        if template is None:
            return 'INVALID_FINAL'
        if unit.get('tone') and '็' in template:
            return 'UNSUPPORTED_COMBINATION'
        if vowel == 'เออะ' and unit.get('final'):
            return 'NEEDS_LEXICON'
    return None


def _compound_boundary(atoms, index, current):
    """เทียบ ตัวสะกด + พยัญชนะต้นถัดไป กับ พยัญชนะต้นถัดไป + สระประสม
    ไม่แน่ใจ ให้ ลองตัด 2 ทาง แล้วตรวจ ถ้าถูกทั้งคู่ให้ทำเครื่องหมายให้ตรวจ
    """
    if (_starts_syllable(atoms, index)
            or not _starts_syllable(atoms, index, onset_context=True)):
        return None
    alternatives = []
    source_indices = current['indices'] + [i for atom in atoms[index:] for i in _source_indices(atom)]
    for role, final, suffix in (
            ('CODA_CONTEXT', atoms[index]['value'], index+1),  # ทางที่ 1: เป็นตัวสะกด
            ('NEXT_EXPLICIT_ONSET', None, index)):  # ทางที่ 2: เป็นต้นพยางค์ถัดไป
        preceding = dict(onset=current['onset'], vowel=current['vowel'],
                         prefix=current['prefix'], tone=current['tone'], final=final,
                         status=current.get('status'),
                         components=deepcopy(current.get('components', [])),
                         indices=current['indices'] + (_source_indices(atoms[index]) if final else []))
        if final:
            preceding['components'].append(_component('FINAL', final, [atoms[index]], 'CODA_CONTEXT'))
        units = [preceding, *_parse_run(atoms[suffix:], compare_boundaries=False)]
        # ผิดได้ 2 แบบ: โครงสร้างพยางค์ผิด หรือใช้รูปเซลล์ผิดมาตรฐาน
        error = _structural_error(units, source_indices) or raw_encoding_error(units)
        alternatives.append(dict(role=role, syllables=units, error=error))
    valid = [candidate for candidate in alternatives if candidate['error'] is None]
    # คงสถานะสระประสมที่เลือกแล้วไว้แสดงผล 
    # คงการแสดงตัวสะกดที่มีอยู่
    # ใช้จัดอันดับตัวเลือกที่รอตรวจ 
    prefer_open = current['semantic'] and (
        current['vowel'] not in SHORT.values() or current['vowel_has_onset_alternative'])
    preferred = alternatives[int(prefer_open)]
    selected = (preferred if preferred in valid else valid[0]) if valid else preferred
    # สระแกนที่ยังไม่ชัด หรือเส้นทางที่ต้องใช้พจนานุกรม 
    # ถ้าใส่ผิดสระ ตัวสะกด หรือวรรณยุกต์ที่ชัด ถึงตัดตัวเลือกออก
    hard_errors = {'INVALID_FINAL', 'INVALID_SYLLABLE', 'UNSUPPORTED_COMBINATION',
                   'INVALID_VOWEL', 'INVALID_SOURCE_CELLS', 'REUSED_SOURCE_CELL', 'INCOMPLETE_PARSE',
                   'NONSTANDARD_UA_CLOSED_FORM'}
    unique = len(valid) == 1 and all(candidate['error'] in hard_errors
                                   for candidate in alternatives if candidate['error'])
    # พยางค์ไม่ครบ ให้คงไว้เพื่อรอตรวจ
    # ไม่เรียกพยางค์กำกวมของการตัดพยางค์ที่พิสูจน์แล้ว
    status = (None if unique else 'AMBIGUOUS_SEGMENTATION' if len(valid) >= 2
              else 'UNRESOLVED_SEGMENTATION')
    return dict(role=selected['role'], review=not unique, status=status,
                indices=_source_indices(atoms[index]))


def _final_role(atoms, index, vowel, *, current=None, compare_boundaries=True):
    """ตัดสินขอบพยางค์ก่อนที่ renderer ลดรูป"""
    # ไม้หันอากาศ ั ต้องมีตัวสะกดเสมอ
    if vowel == 'ั':
        return dict(role='CODA_CONTEXT', review=False)
    # พยัญชนะที่มีวรรณยุกต์ตามหลังเป็นต้นพยางค์ใหม่ เช่น ม ใน มะม่วง
    if index+1 < len(atoms) and atoms[index+1]['kind'] == 'tone':
        return dict(role='NEXT_EXPLICIT_ONSET', review=False)
    if compare_boundaries and current is not None:
        decision = _compound_boundary(atoms, index, current)
        if decision is not None:
            return decision
    if _starts_syllable(atoms, index, onset_context=not compare_boundaries):
        return dict(role='NEXT_EXPLICIT_ONSET', review=False)
    if _implicit_closed_span(atoms, index) is not None:
        # การลงท้ายด้วย ะ สนับสนุนพยางค์เปิด เมื่อรูปสระ
        # ไม่เปลี่ยน ให้เลือกพยางค์ถัดไปที่สมบูรณ์ด้วย
        # ไม่อย่างนั้นขอบพยางค์อาจเปลี่ยนรูปสระที่อ่านได้
        if vowel == 'อะ' or vowel in SHORT.values() or vowel not in CODA_CHANGES_VOWEL:
            return dict(role='NEXT_IMPLICIT_ONSET', review=False)
        return dict(role='AMBIGUOUS_BOUNDARY', review=False)
    return dict(role='CODA_CONTEXT', review=False)


def _parse_run(atoms, *, compare_boundaries=True): #ตัดท่อนเป็นพยางค์: สระหน้า, ต้น, สระ, ตัวสะกด
    """atom หนึ่งช่วง แล้วตัดเป็นพยางค์ทีละพยางค์ ใช้ yield ส่งพยางค์ออกไปทีละตัว"""
    index = 0
    while index < len(atoms): #1รอบ  1 พยางค์
        start = index # จำจุดเริ่มพยางค์
        prefix = '' # สระหน้า (เ แ โ ไ) 
        components, prefix_atoms, vowel_atoms = [], [], []
        vowel_rule, ambiguous_mark = 'VOWEL', False # กฎสระเริ่มต้น, ไม้ไต่คู้
        if atoms[index]['kind'] == 'leading' and index+1 < len(atoms) and _is_consonant(atoms[index+1]):
            prefix = atoms[index]['value'] # เป็นสระหน้า ตัวถัดไปเป็นพยัญชนะ
            prefix_atoms.append(atoms[index]) # เก็บ เ แ โ ไ ไว้ก่อน
            index += 1 # เลื่อนไปที่พยัญชนะ
        if not _is_consonant(atoms[index]): #ไม่ใช่พยัญชนะ
            atom = atoms[index]
            status = ('AMBIGUOUS_SEQUENCE' if atom['kind'] == 'ambiguous_mark' else
                      'AMBIGUOUS_SYLLABLE' if atom['semantic'] else None)
            yield dict(indices=_source_indices(atom), literal=atom['value'], status=status) #ส่งออกแบบไม่ใช่พยางค์
            index += 1
            continue
        onset_start = index
        implicit = _implicit_closed_span(atoms, index) if not prefix else None #พยัญชนะต้นและควบกล้ำ
        onset = atoms[index]['value']
        index += 1
        if (index < len(atoms) and _is_consonant(atoms[index])
                and _cluster(onset, atoms[index]['value'])
                and (prefix or _starts_syllable(atoms, index, onset_context=True)
                     or (implicit is not None and implicit[0] > index))):
            onset += atoms[index]['value']
            index += 1
        components.append(_component('ONSET', onset, atoms[onset_start:index], 'ONSET_CONTEXT'))
        vowel, tone, semantic, status = None, None, False, None
        while index < len(atoms): #วนเก็บสระและวรรณยุกต์
            atom = atoms[index]
            matched = match_post_onset_vowel(atoms, index) if vowel is None and not prefix else None
            if atom['kind'] == 'tone': #เก็บวรรณยุกต์
                if tone:
                    status = 'UNSUPPORTED_COMBINATION' #ถ้าซ้ำ
                tone = atom['value']
                components.append(_component('TONE', tone, [atom], 'SYLLABLE_TONE'))
            elif matched:
                vowel, span = matched
                semantic, vowel_rule = True, 'POST_ONSET_SEQUENCE'
                vowel_atoms.extend(atoms[index:index+span])
                index += span
                continue
            elif atom['kind'] == 'leading' and not prefix and vowel is None:
                if index+1 < len(atoms) and _is_consonant(atoms[index+1]):
                    break
                prefix = atom['value']
                prefix_atoms.append(atom)
            elif vowel is None and _is_written_o(atoms, index):
                # อ ที่เขียนหลังพยัญชนะต้น อ่านเป็นสระ ออ
                vowel = 'ออ'
                vowel_atoms.append(atom)
                vowel_rule = 'WRITTEN_O_AFTER_ONSET'
            elif atom['kind'] == 'ambiguous_mark' and vowel is None:
                vowel, ambiguous_mark = '็', True
                vowel_atoms.append(atom)
                vowel_rule = 'EXPLICIT_MAITAIKHU'
            elif atom['kind'] == 'vowel':
                if vowel is None:
                    vowel, semantic = atom['vowel'], atom['semantic']
                elif atom['vowel'] == 'อะ' and vowel in SHORT:
                    vowel, semantic = SHORT[vowel], True
                else:
                    break
                vowel_atoms.append(atom)
            else:
                break
            index += 1
        # สระหน้าคู่กับ ะ เป็นสระเสียงสั้น ส่วนสระอื่น (เ + ก + ิ) คงรูปที่เขียนไว้
        orthographic_prefix = ''
        if prefix:
            if vowel is None:
                vowel = LEADING[prefix]
                vowel_atoms = prefix_atoms
            elif vowel == 'อะ' and prefix in 'เแโ':
                vowel, semantic = SHORT[LEADING[prefix]], True
                vowel_atoms = prefix_atoms + vowel_atoms
            elif prefix == 'เ' and vowel == 'อา' and index < len(atoms) and atoms[index]['vowel'] == 'อะ':
                vowel, semantic = 'เอาะ', True
                vowel_atoms = prefix_atoms + vowel_atoms + [atoms[index]]
                index += 1
            else:
                orthographic_prefix = prefix
                components.append(_component('PRE_VOWEL', prefix, prefix_atoms, 'WRITTEN_PREFIX'))
        if vowel is None and not prefix and implicit is not None and index == implicit[1]:
            vowel, semantic, vowel_rule = 'โอะ', True, 'IMPLICIT_O_CLOSED'
        if vowel:
            components.append(_component('VOWEL', vowel, vowel_atoms, vowel_rule))
        final = None
        boundary_rule = 'RUN_END'
        boundary_decision = dict(review=False)
        if vowel and index < len(atoms) and _is_consonant(atoms[index]):
            if vowel_rule == 'IMPLICIT_O_CLOSED':
                boundary_rule = 'CODA_IMPLICIT_O'
            else:
                current = dict(onset=onset, vowel=vowel, tone=tone,
                               prefix=orthographic_prefix, semantic=semantic, status=status,
                               components=components,
                               vowel_has_onset_alternative=any(_is_consonant(a) for a in vowel_atoms),
                               indices=[i for a in atoms[start:index] for i in _source_indices(a)])
                boundary_decision = _final_role(atoms, index, vowel, current=current,
                                                compare_boundaries=compare_boundaries)
                boundary_rule = boundary_decision['role']
            if (not boundary_rule.startswith('NEXT_') and
                    (semantic or vowel not in INHERENT_FINAL)):
                if boundary_rule == 'AMBIGUOUS_BOUNDARY':
                    status = status or 'AMBIGUOUS_SYLLABLE'
                tail = index
                while tail < len(atoms) and _is_consonant(atoms[tail]):
                    tail += 1
                if (semantic and vowel_rule != 'IMPLICIT_O_CLOSED'
                        and tail-index > 1 and tail == len(atoms)
                        and not _starts_syllable(atoms, index+1)
                        and _implicit_closed_span(atoms, index+1) is None):
                    # พยัญชนะเปล่าหลายตัวอาจเป็นตัวสะกด ตัวการันต์
                    # หรือพยางค์ที่สระไม่เขียนรูป ไม่มีพจนานุกรมมาตัดสิน
                    status = status or 'AMBIGUOUS_SYLLABLE'
                final = atoms[index]['value']
                components.append(_component('FINAL', final, [atoms[index]], 'CODA_CONTEXT'))
                index += 1
                if index < len(atoms) and atoms[index]['kind'] == 'tone':
                    if tone:
                        status = 'UNSUPPORTED_COMBINATION'
                    tone = atoms[index]['value']
                    components.append(_component('TONE', tone, [atoms[index]], 'SYLLABLE_TONE'))
                    index += 1
        if ambiguous_mark:
            # ไม้ไต่คู้ที่เขียนเองในพยางค์เปิด (ก็) หรือ เ/แ ที่มีตัวสะกด อ่านได้ชัด กรณีอื่นรอตรวจ
            explicit_open = not orthographic_prefix and final is None and tone is None
            explicit_closed = orthographic_prefix in ('เ', 'แ') and final is not None
            if not (explicit_open or explicit_closed):
                status = status or 'AMBIGUOUS_SEQUENCE'
        unit = dict(onset=onset, vowel=vowel, tone=tone, final=final, prefix=orthographic_prefix,
                    indices=[i for a in atoms[start:index] for i in _source_indices(a)],
                    boundary_review=boundary_decision['review'],
                    boundary_status=boundary_decision.get('status'),
                    boundary_indices=boundary_decision.get('indices', []),
                    components=sorted(components, key=lambda c: c['indices'][0] if c['indices']
                                      else atoms[onset_start]['index']+.5),
                    status=status, structural_status=status)
        # ตรวจรูปเซลล์ดิบของทุกพยางค์ด้วย เช่น ส + 15 + น แบบเก่า
        unit['status'] = status or raw_encoding_error([unit])
        yield unit


def parse_tokens(tokens, atoms): #เก็บตัวอักษรที่ติดกัน เจอ ' ',ขึ้นบรรทัด หรือเซลล์เสีย ส่งตัด
    """เก็บ token ต้นทาง และคืนพยางค์หรือข้อความตามลำดับโดยไม่แปลงเป็นข้อความ"""
    raw = deepcopy(list(tokens))
    units, run = [], []  # run = atom ที่ติดกันรอแยกพยางค์

    def flush(): 
        units.extend(_parse_run(run)) #ส่งท่อนที่เก็บไป
        run.clear()

    def literal(value, indices, status=None): #เติมหน่วยที่ไม่ใช่พยางค์ 
        if value:
            units.append(dict(literal=value, indices=indices, status=status, trace=bool(status)))

    last_line = None
    for index, record in enumerate(raw):
        if record.get('consumed'):
            continue
        atom = deepcopy(atoms[index])
        line = record.get('line_id')
        line_break = record.get('line_break_before') or (line is not None and last_line is not None and line != last_line)
        if line is not None:
            last_line = line
        sequence_break = not atom.get('adjacent_before', True)  # ไม่ติดตัวก่อนหน้า = แยกช่วงใหม่
        if line_break or record.get('break_before') or sequence_break:
            flush()
            if units:
                if line_break:
                    while units and units[-1].get('literal') == ' ':
                        units.pop()
                    literal('\n', [])
                elif units[-1].get('literal') not in (' ', '\n'):
                    literal(' ', [])
        if atom['kind'] in ('literal', 'error'):
            flush()
            value = atom['value']
            indices = atom.get('source_indices', [index])
            if atom['kind'] == 'error':
                literal('�', indices, record.get('warning') or 'UNKNOWN_TOKEN')
            elif value and not (value == ' ' and (not units or units[-1].get('literal') in (' ', '\n'))):
                literal(value, indices)
        else:
            run.append(atom)
    flush()
    return dict(raw_tokens=raw, syllables=units)


def parse_braille(cells):
    """เซลล์ดิบ -> พยางค์ (ยังไม่สร้างข้อความไทย ให้ thai_renderer ทำต่อ)"""
    raw, tokens, atoms = lex_braille(cells)
    return dict(parse_tokens(tokens, atoms), raw_cells=raw)
