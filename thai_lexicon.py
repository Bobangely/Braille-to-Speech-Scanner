"""หลักฐานจากพจนานุกรมแบบเพิ่มความมั่นใจอย่างเดียว สำหรับเส้นทางไทยที่ผ่านการตรวจโครงสร้างแล้ว"""
from copy import deepcopy
from functools import lru_cache
import logging
import re
import unicodedata

from text_to_braille import text_to_braille_cells

logger = logging.getLogger(__name__)
_PLAIN_WORD = re.compile('[ก-ฮะ-ูเ-๎]+')
_LONGEST_PIECE = 30  # เซลล์ของคำย่อยยาวสุดที่ลองตัด คำยาวกว่านี้ค้นตรงได้อยู่แล้ว


@lru_cache(maxsize=1)
def default_lexicon():
    """โหลดรายการคำที่มากับแพ็กเกจครั้งเดียว และไม่ดึงข้อมูลตอนรันจริง"""
    try:
        from pythainlp.corpus import thai_words
        return frozenset(unicodedata.normalize('NFC', word) for word in thai_words() if word)
    except (ImportError, OSError) as exc:
        logger.warning('Thai lexicon unavailable; keeping structural candidates: %s', exc)
        return frozenset()


def _is_plain_word(word):
    """ตัด ฯ จุด ตัวเลข และพินทุ เพราะไม่มีเซลล์ของตัวเอง ทำให้ชนกับคำอื่น"""
    return bool(_PLAIN_WORD.fullmatch(word))


@lru_cache(maxsize=1)
def _cell_table():
    """เซลล์เบรลล์ของทุกคำในพจนานุกรม -> คำนั้น"""
    return {tuple(frozenset(cell) for cell in text_to_braille_cells(word, 'thai')): word
            for word in default_lexicon() if _is_plain_word(word)}


def word_from_cells(cells):
    """คำจากพจนานุกรมที่มีเซลล์ตรงกันทั้งหมด หรือ None ถ้าไม่รู้จัก

    กฎอย่างเดียวแยก ะ กับ ั ไม่ได้เพราะใช้เซลล์เดียวกัน คำประสมที่ไม่อยู่ในพจนานุกรม
    จะตัดเป็นคำย่อยให้ครบทุกเซลล์ โดยใช้จำนวนคำน้อยที่สุด
    """
    table, cells = _cell_table(), tuple(cells)
    if cells in table:
        return table[cells]
    best = [None] * len(cells) + [(0, '')]  # ตำแหน่งนี้ถึงท้ายคำ: (จำนวนคำ, ข้อความ)
    for start in range(len(cells) - 1, -1, -1):
        for end in range(start + 1, min(len(cells), start + _LONGEST_PIECE) + 1):
            word = table.get(cells[start:end])
            if word and best[end] and (best[start] is None or best[end][0] + 1 < best[start][0]):
                best[start] = (best[end][0] + 1, word + best[end][1])
    return best[0][1] if best[0] else None


def _coverage(parts, words):
    """ความครอบคลุมตัวอักษรที่รู้จักสูงสุดที่ขอบพยางค์ โดยยอมให้มีส่วนที่ไม่รู้จัก

    ไม่จับคู่สตริงย่อยภายในพยางค์ และไม่หักคะแนนชื่อใหม่
    คำเต็มอาจคร่อมหลายพยางค์ได้ แต่ห้ามข้ามขอบ run ของ parser
    """
    best = [(0, []) for _ in range(len(parts) + 1)]
    for start in range(len(parts)-1, -1, -1):
        best[start] = best[start+1]
        word = ''
        for end in range(start+1, len(parts)+1):
            word += parts[end-1]
            if word in words:
                score = len(word) + best[end][0]
                if score > best[start][0]:
                    best[start] = (score, [dict(word=word, start=start, end=end), *best[end][1]])
    return best[0]


def rank_segmentations(parsed, lexicon, render_unit, confidence_threshold=None):
    """จัดอันดับเส้นทางที่ผ่านการตรวจ และตัดสินเฉพาะเมื่อพจนานุกรมยืนยันครบหลายคำ

    ตัวเลือกอื่นยังเก็บไว้ใน trace รวมถึงชื่อที่ไม่เคยเห็น
    หลักฐานคำเดียว ไม่ครบ เสมอกัน หรือยังไม่ยืนยันเซลล์ดิบ จะยังอยู่ในสถานะรอตรวจ
    """
    runs = parsed.get('segmentation_runs', [])
    if not runs:
        return parsed['syllables'], [], dict(source='unused', available=False)
    words = default_lexicon() if lexicon is None else frozenset(
        unicodedata.normalize('NFC', word) for word in lexicon if isinstance(word, str) and word)
    info = dict(source='pythainlp.corpus.thai_words' if lexicon is None else 'supplied',
                available=bool(words), word_count=len(words))
    replacements, rankings = {}, []
    for run in runs:
        original = parsed['syllables'][run['unit_start']:run['unit_start']+run['unit_count']]
        candidates = []
        if run['generation_complete']:
            for path in run['candidates']:
                if not path['valid']:
                    continue
                try:
                    parts = [render_unit(unit) for unit in path['syllables']]
                except ValueError:
                    # ความผิดพลาดเชิงโครงสร้างระดับ renderer
                    # จะกลายเป็นถูกเพราะพจนานุกรมไม่ได้
                    continue
                score, matches = _coverage(parts, words)
                text = ''.join(parts)
                candidates.append(dict(text=text, valid=True, plausible=True,
                                       accepted_mode=path.get('accepted_mode', 'standard'),
                                       acceptance_reason=path.get('acceptance_reason'),
                                       standard_valid=path.get('standard_valid', True),
                                       standard_error=path.get('standard_error'),
                                       compatibility_valid=path.get('compatibility_valid', True),
                                       compatibility_error=path.get('compatibility_error'),
                                       known_characters=score, matches=matches,
                                       coverage=score / max(1, len(text)),
                                       syllables=deepcopy(path['syllables'])))
        # ใช้ lexicon จัดอันดับภายใน mode เดียวกันเท่านั้น
        candidates.sort(key=lambda candidate: (candidate['accepted_mode'] != 'standard',
                                                -candidate['coverage']))
        selected = None
        original_text = None
        reason = 'INCOMPLETE_CANDIDATE_SEARCH' if not run['generation_complete'] else 'NO_VALID_CANDIDATE'
        if candidates:
            highest = candidates[0]['coverage']
            best = [c for c in candidates if c['coverage'] == highest
                    and c['accepted_mode'] == candidates[0]['accepted_mode']]
            reason = ('UNIQUE_STRUCTURAL_PATH' if len(candidates) == 1 else
                      'NO_LEXICAL_EVIDENCE' if highest == 0 else
                      'LEXICON_TIE' if len(best) > 1 else 'LEXICON_SUPPORTED_DISPLAY')
            # ถ้าไม่มีหลักฐาน ให้คงตัวเลือกการแสดงผลตามโครงสร้างไว้
            # ถ้าคะแนนเสมอ ให้คงไว้เมื่ออยู่ในกลุ่มตัวเลือกที่แข็งแรงที่สุด
            try:
                original_text = ''.join(render_unit(unit) for unit in original)
            except ValueError:
                pass
            if highest or any(c['accepted_mode'] != candidates[0]['accepted_mode'] for c in candidates):
                selected = next((c for c in best if c['text'] == original_text), best[0])
        # ต้องมีบริบทวลีและคำที่คร่อมหลายพยางค์ ไม่ใช่ชื่อเดี่ยว
        # หรือคำเล็กๆ จากพจนานุกรมเรียงกัน
        complete = [c for c in candidates if c['standard_valid'] and c['accepted_mode'] == 'standard'
                    and c['coverage'] == 1 and len(c['matches']) >= 2
                    and any(m['end']-m['start'] > 1 for m in c['matches'])]
        raw = parsed.get('raw_cells', [])
        evidence = [raw[i] for i in run['indices'] if i < len(raw)]
        resolved = (run['generation_complete'] and len(complete) == 1
                    and sum(c['coverage'] == 1 for c in candidates) == 1
                    and len(evidence) == len(run['indices'])
                    and all(c.get('pattern') and not c.get('warning') for c in evidence)
                    and all(confidence_threshold is None or c.get('confidence') is None
                            or c['confidence'] >= confidence_threshold for c in evidence))
        if resolved:
            selected = complete[0]
            reason = 'UNIQUE_COMPLETE_LEXICAL_SUPPORT'
        # คงหลักฐานเดิมไว้เมื่อพจนานุกรมยืนยันไม่พอ
        units = deepcopy(selected['syllables'] if selected and selected['text'] != original_text else original)
        review = not resolved and (not run['generation_complete'] or len(candidates) > 1
                  or any(u.get('boundary_review') for u in original))
        if resolved:
            units = deepcopy(selected['syllables'])
            for unit in units:
                unit['boundary_review'] = False
                unit['boundary_status'] = None
                unit['boundary_evidence'] = 'UNIQUE_COMPLETE_LEXICAL_SUPPORT'
        if review and units:
            # คงหลักฐานเดิมไว้ คะแนนจัดอันดับลบคำเตือนไม่ได้
            origin = next((u for u in original if u.get('boundary_candidates')), original[0])
            owner = next((u for u in units if origin['indices'][0] in u['indices']), units[0])
            for key in ('boundary_indices', 'boundary_patterns', 'boundary_source_cells',
                        'boundary_candidates', 'boundary_evidence'):
                if key in origin:
                    owner[key] = deepcopy(origin[key])
            owner['boundary_review'] = True
            owner['boundary_status'] = (origin.get('boundary_status') or
                                           ('AMBIGUOUS_SEGMENTATION' if len(candidates) > 1
                                            else 'UNRESOLVED_SEGMENTATION'))
        replacements[run['unit_start']] = (run['unit_count'], units)
        rankings.append(dict(indices=run['indices'], generation_complete=run['generation_complete'],
                             reason=reason, candidates=candidates, needs_review=review,
                             selected_text=selected['text'] if selected else None))
    result = []
    index = 0
    while index < len(parsed['syllables']):
        if index in replacements:
            count, units = replacements[index]
            result.extend(units)
            index += count
        else:
            result.append(parsed['syllables'][index])
            index += 1
    return result, rankings, info
