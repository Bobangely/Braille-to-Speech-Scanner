"""แปลงพยางค์ไทยที่แยกแล้วเป็นข้อความ โดยไม่ค้นเบรลล์และไม่ตัดพยางค์"""

from functools import lru_cache
import json
from pathlib import Path
import unicodedata


CONSONANTS = set('กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ')
TONES = set('่้๊๋')
LEADING = {'เ': 'เอ', 'แ': 'แอ', 'โ': 'โอ', 'ไ': 'ไอ', 'ใ': 'ใอ'}
# สระ (รูปไม่มีตัวสะกด, รูปมีตัวสะกด) None = มีตัวสะกดไม่ได้
# O พยัญชนะต้น  T วรรณยุกต์  F ตัวสะกด
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
ALIASES.update({  #ค่าสระจาก config_thai -> ชื่อสระใน RULES
    'ะ': 'อะ', 'า': 'อา', 'ิ': 'อิ', 'ี': 'อี', 'ึ': 'อึ', 'ุ': 'อุ', 'ู': 'อู', 'ำ': 'อำ',
    'เ-า': 'เอา', 'เ-อ': 'เออ', 'เ-ีย': 'เอีย', 'เ-ือ': 'เอือ', '-ัว': 'อัว',
})
SHORT = {'เอ': 'เอะ', 'แอ': 'แอะ', 'โอ': 'โอะ', 'เออ': 'เออะ',#สระยาว สระสั้น
         'เอา': 'เอาะ', 'เอีย': 'เอียะ', 'เอือ': 'เอือะ', 'อัว': 'อัวะ'}
INHERENT_FINAL = {'อำ', 'ไอ', 'ใอ', 'เอา'} #สระที่มีเสียงตัวสะกดในตัว
SHORT_COMPOUNDS = {'เอียะ', 'เอือะ', 'อัวะ'} #สระสั้นประสม มีตัวสะกด
CODA_CHANGES_VOWEL = {vowel for vowel, (opened, closed) in RULES.items() #สระที่เปลี่ยนรูปเมื่อมีตัวสะกด
                      if closed is not None and opened != closed.replace('{F}', '')}


class CompositionError(ValueError):
    def __init__(self, status):
        self.status = status
        super().__init__(status)


def normalize_vowel(value):#แปลงรูปสระเป็นชื่อสระ
    return ALIASES.get(value.replace('◌', '-'))


@lru_cache(maxsize=1)
def _exceptions():
    path = Path(__file__).with_name('thai_word_exceptions.json') #ท + เออ + ม
    with path.open(encoding='utf-8') as source:
        return {(r['onset'], r['vowel'], r.get('final'), r.get('tone')): r['text']
                for r in json.load(source)}


def compose_syllable(onset, vowel, final=None, tone=None):
    """แปลงพยางค์ที่แยกแล้ว การประกอบที่ผิดจะเกิดสถานะแจ้ง"""
    vowel = normalize_vowel(vowel)
    if vowel is None:
        raise CompositionError('UNKNOWN_VOWEL')
    if not onset or any(c not in CONSONANTS for c in onset):
        raise CompositionError('AMBIGUOUS_SYLLABLE') #พยัญชนะต้น
    if final and (len(final) != 1 or final not in CONSONANTS):
        raise CompositionError('INVALID_FINAL') #ตัวสะกด
    if tone and tone not in TONES:
        raise CompositionError('UNSUPPORTED_COMBINATION') #วรรณยุกต์
    exception = _exceptions().get((onset, vowel, final or None, tone or None))
    if exception is not None: #คำยกเว้น
        return exception
    if vowel == 'เออ' and final == 'ย':
        template = 'เ{O}{T}ย'
    else:
        template = RULES[vowel][bool(final)]
    if template is None:
        raise CompositionError('NEEDS_LEXICON' if vowel in SHORT_COMPOUNDS else 'INVALID_FINAL')
    # ไม้ไต่คู้กับวรรณยุกต์ต้องใช้ค่าจากพจนานุกรม 
    if tone and '็' in template:
        raise CompositionError('UNSUPPORTED_COMBINATION')
    return unicodedata.normalize('NFC', template.format(O=onset, T=tone or '', F=final or ''))


def render_syllable(syllable): 
    #เช็คพยางค์ สัญลักษณ์ กับ สระ 
    if 'literal' in syllable:
        return syllable['literal']
    if syllable.get('vowel'):
        return syllable.get('prefix', '') + compose_syllable(
            syllable['onset'], syllable['vowel'], syllable.get('final'), syllable.get('tone'))
    return syllable['onset'] + (syllable.get('tone') or '')


def render_thai(parsed):
    """แปลงทุกพยางค์เป็นข้อความ พยางค์ที่ผิดแสดงเป็น �"""
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
        text = '�' if status else candidate
        pieces.append(text)
        # ตัดพยางค์กำกวม: ยังแสดงข้อความ แต่ส่งให้ตรวจ
        if syllable.get('boundary_review'):
            status = status or syllable['boundary_status']
        if traced or status: #เก็บรายละเอียด: text, status, needs_review
            traces.append(dict(syllable, text=text, status=status or 'OK', needs_review=bool(status)))
    return dict(parsed, text=unicodedata.normalize('NFC', ''.join(pieces).strip()),
                syllables=traces, needs_review=any(t['needs_review'] for t in traces))
