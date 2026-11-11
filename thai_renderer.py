"""แปลงพยางค์ไทยที่แยกแล้วเป็นข้อความ โดยไม่ค้นเบรลล์และไม่ตัดพยางค์"""

from functools import lru_cache
import json
from pathlib import Path
import unicodedata


CONSONANTS = set('กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ')
TONES = set('่้๊๋')
LEADING = {'เ': 'เอ', 'แ': 'แอ', 'โ': 'โอ', 'ไ': 'ไอ', 'ใ': 'ใอ'}
# ทุกกฎวางวรรณยุกต์ชัดเจน หลังสระที่ประกอบกับพยัญชนะและก่อน
# สระหลังหรือตัวสะกด O รวมพยัญชนะต้นทั้งหมด ตัวสุดท้ายเป็นที่วาง T
RULES = {
    'อะ': ('{O}{T}ะ', '{O}ั{T}{F}'),
    'อา': ('{O}{T}า', '{O}{T}า{F}'),
    'อิ': ('{O}ิ{T}', '{O}ิ{T}{F}'),
    'อี': ('{O}ี{T}', '{O}ี{T}{F}'),
    'อึ': ('{O}ึ{T}', '{O}ึ{T}{F}'),
    'อือ': ('{O}ื{T}อ', '{O}ื{T}{F}'),
    'อุ': ('{O}ุ{T}', '{O}ุ{T}{F}'),
    'อู': ('{O}ู{T}', '{O}ู{T}{F}'),
    'เอะ': ('เ{O}{T}ะ', 'เ{O}็{T}{F}'),
    'เอ': ('เ{O}{T}', 'เ{O}{T}{F}'),
    'แอะ': ('แ{O}{T}ะ', 'แ{O}็{T}{F}'),
    'แอ': ('แ{O}{T}', 'แ{O}{T}{F}'),
    'โอะ': ('โ{O}{T}ะ', '{O}{T}{F}'),
    'โอ': ('โ{O}{T}', 'โ{O}{T}{F}'),
    'เอาะ': ('เ{O}{T}าะ', '{O}็{T}อ{F}'),
    'ออ': ('{O}{T}อ', '{O}{T}อ{F}'),
    'เออ': ('เ{O}{T}อ', 'เ{O}ิ{T}{F}'),
    'เออะ': ('เ{O}{T}อะ', 'เ{O}ิ{T}{F}'),
    'เอีย': ('เ{O}ี{T}ย', 'เ{O}ี{T}ย{F}'),
    'เอือ': ('เ{O}ื{T}อ', 'เ{O}ื{T}อ{F}'),
    'อัว': ('{O}ั{T}ว', '{O}{T}ว{F}'),
    'เอียะ': ('เ{O}ี{T}ยะ', None),
    'เอือะ': ('เ{O}ื{T}อะ', None),
    'อัวะ': ('{O}ั{T}วะ', None),
    'อำ': ('{O}{T}ำ', None),
    'ไอ': ('ไ{O}{T}', None),
    'ใอ': ('ใ{O}{T}', None),
    'เอา': ('เ{O}{T}า', None),
    # token ตัวสะกดที่อ่านจากหน้าแล้ว ไม่ใช่ชื่อสระเชิงความหมาย
    'ั': ('{O}ั{T}', '{O}ั{T}{F}'),
    'ื': ('{O}ื{T}', '{O}ื{T}{F}'),
    '็': ('{O}็{T}', '{O}็{T}{F}'),
    'ํ': ('{O}ํ{T}', '{O}ํ{T}{F}'),
}
ALIASES = {name: name for name in RULES}
ALIASES.update({
    'ะ': 'อะ', '-ะ': 'อะ', 'า': 'อา', '-า': 'อา',
    'ิ': 'อิ', '-ิ': 'อิ', 'ี': 'อี', '-ี': 'อี',
    'ึ': 'อึ', '-ึ': 'อึ', '-ือ': 'อือ', 'ุ': 'อุ', '-ุ': 'อุ',
    'ู': 'อู', '-ู': 'อู', 'เ-': 'เอ', 'แ-': 'แอ', 'โ-': 'โอ',
    'เ-ะ': 'เอะ', 'แ-ะ': 'แอะ', 'โ-ะ': 'โอะ', 'เ-าะ': 'เอาะ',
    '-อ': 'ออ', 'เ-อ': 'เออ', 'เ-อะ': 'เออะ',
    'เ-ีย': 'เอีย', 'เ-ือ': 'เอือ', '-ัว': 'อัว',
    'เ-ียะ': 'เอียะ', 'เ-ือะ': 'เอือะ', '-ัวะ': 'อัวะ',
    'ำ': 'อำ', '-ำ': 'อำ', 'ไ-': 'ไอ', 'ใ-': 'ใอ', 'เ-า': 'เอา',
})
SHORT = {'เอ': 'เอะ', 'แอ': 'แอะ', 'โอ': 'โอะ', 'เออ': 'เออะ',
         'เอา': 'เอาะ', 'เอีย': 'เอียะ', 'เอือ': 'เอือะ', 'อัว': 'อัวะ'}
INHERENT_FINAL = {'อำ', 'ไอ', 'ใอ', 'เอา'}
SHORT_COMPOUNDS = {'เอียะ', 'เอือะ', 'อัวะ'}
CODA_CHANGES_VOWEL = {vowel for vowel, (opened, closed) in RULES.items()
                      if closed is not None and opened != closed.replace('{F}', '')}


class CompositionError(ValueError):
    def __init__(self, status):
        self.status = status
        super().__init__(status)


def normalize_vowel(value):
    """ปรับชื่อและสัญลักษณ์แทนที่ให้เป็นมาตรฐาน โดยไม่เขียนข้อความที่ตรวจพบใหม่"""
    return ALIASES.get(value.removeprefix('สระ').replace('◌', '-'))


@lru_cache(maxsize=1)
def _exceptions():
    path = Path(__file__).with_name('thai_word_exceptions.json')
    with path.open(encoding='utf-8') as source:
        return {(r['onset'], r['vowel'], r.get('final'), r.get('tone')): r['text']
                for r in json.load(source)}


def compose_syllable(onset, vowel, final=None, tone=None):
    """แปลงพยางค์ที่แยกชัดแล้ว การประกอบที่ผิดจะเกิดสถานะแจ้ง"""
    vowel = normalize_vowel(vowel)
    if vowel is None:
        raise CompositionError('UNKNOWN_VOWEL')
    if not onset or any(c not in CONSONANTS for c in onset):
        raise CompositionError('AMBIGUOUS_SYLLABLE')
    if final and (len(final) != 1 or final not in CONSONANTS):
        raise CompositionError('INVALID_FINAL')
    if tone and tone not in TONES:
        raise CompositionError('UNSUPPORTED_COMBINATION')
    exception = _exceptions().get((onset, vowel, final or None, tone or None))
    if exception is not None:
        return exception
    if vowel == 'เออ' and final == 'ย':
        template = 'เ{O}{T}ย'
    else:
        template = RULES[vowel][bool(final)]
    if template is None:
        raise CompositionError('NEEDS_LEXICON' if vowel in SHORT_COMPOUNDS else 'INVALID_FINAL')
    # ไม้ไต่คู้กับวรรณยุกต์ต้องใช้หลักฐานจากพจนานุกรม อย่าใส่สองเครื่องหมายที่แข่งกัน
    if tone and '็' in template:
        raise CompositionError('UNSUPPORTED_COMBINATION')
    return unicodedata.normalize('NFC', template.format(O=onset, T=tone or '', F=final or ''))


def render_syllable(syllable):
    """แปลงพยางค์เชิงความหมายหนึ่งพยางค์ รวมส่วนนำที่เขียนไว้แล้ว"""
    if 'literal' in syllable:
        return syllable['literal']
    if syllable.get('vowel'):
        return syllable.get('prefix', '') + compose_syllable(
            syllable['onset'], syllable['vowel'], syllable.get('final'), syllable.get('tone'))
    return syllable['onset'] + (syllable.get('tone') or '')


def render_thai(parsed, *, confidence_threshold=None):
    """แปลงและตรวจหน่วยที่แยกแล้ว โดยเก็บตัวเลือกที่ถูกปฏิเสธไว้ใน trace"""
    records = [dict(token=t) if isinstance(t, str) else t for t in parsed['raw_tokens']]
    pieces, traces = [], []
    for unit in parsed['syllables']:
        syllable = dict(unit)
        status = syllable.pop('status', None)
        traced = syllable.pop('trace', True)
        try:
            candidate = render_syllable(syllable)
            if syllable.get('vowel') == 'เออะ' and syllable.get('final'):
                status = status or 'NEEDS_LEXICON'
        except CompositionError as exc:
            candidate, status = None, status or exc.status
        if status and syllable.get('literal') == '�':
            candidate = None
        confidence = [records[i]['confidence'] for i in syllable['indices']
                      if records[i].get('confidence') is not None]
        minimum = min(confidence) if confidence else None
        if confidence_threshold is not None and minimum is not None and minimum < confidence_threshold:
            status = status or 'LOW_CONFIDENCE'
        text = '�' if status else candidate
        pieces.append(text)
        # การตัดพยางค์ที่กำกวมเก็บไว้เป็นตัวเลือกแสดงผล ไม่ใช่การยืนยัน
        # กรณีผิดหรือความมั่นใจต่ำอื่นๆ ยังแสดง � เหมือนเดิม
        if syllable.get('boundary_review'):
            status = status or syllable['boundary_status']
        if traced or status:
            events = []
            if candidate and '็' in candidate:
                explicit = '3' in syllable.get('raw_patterns', []) or syllable.get('vowel') == '็'
                reasons = {'เอะ': 'VOWEL_E_SHORT_CLOSED', 'แอะ': 'VOWEL_AE_SHORT_CLOSED',
                           'เอาะ': 'VOWEL_O_SHORT_CLOSED'}
                reason = 'EXPLICIT_MAITAIKHU' if explicit else reasons.get(syllable.get('vowel'))
                if reason:
                    events.append(dict(mark='็', inserted_mark=None if explicit else '็',
                                       reason=reason, source=syllable.get('raw_patterns', [])))
            traces.append(dict(syllable, candidate=candidate, text=text, status=status or 'OK',
                               needs_review=bool(status), min_confidence=minimum, render_events=events))
    return dict(parsed, text=unicodedata.normalize('NFC', ''.join(pieces).strip()),
                syllables=traces, needs_review=any(t['needs_review'] for t in traces),
                status='NEEDS_REVIEW' if any(t['needs_review'] for t in traces) else 'VALID')
