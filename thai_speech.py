"""ชื่อพยัญชนะไทยสำหรับการพูด โดยไม่ขึ้นกับ TTS"""
import re
import unicodedata


THAI_CONSONANT_NAMES = {
    'ก': 'กอไก่', 'ข': 'ขอไข่', 'ฃ': 'ขอขวด', 'ค': 'คอควาย',
    'ฅ': 'คอคน', 'ฆ': 'คอระฆัง', 'ง': 'งองู', 'จ': 'จอจาน',
    'ฉ': 'ฉอฉิ่ง', 'ช': 'ชอช้าง', 'ซ': 'ซอโซ่', 'ฌ': 'ชอเฌอ',
    'ญ': 'ยอหญิง', 'ฎ': 'ดอชฎา', 'ฏ': 'ตอปฏัก', 'ฐ': 'ถอฐาน',
    'ฑ': 'ทอมณโฑ', 'ฒ': 'ทอผู้เฒ่า', 'ณ': 'นอเณร', 'ด': 'ดอเด็ก',
    'ต': 'ตอเต่า', 'ถ': 'ถอถุง', 'ท': 'ทอทหาร', 'ธ': 'ทอธง',
    'น': 'นอหนู', 'บ': 'บอใบไม้', 'ป': 'ปอปลา', 'ผ': 'ผอผึ้ง',
    'ฝ': 'ฝอฝา', 'พ': 'พอพาน', 'ฟ': 'ฟอฟัน', 'ภ': 'พอสำเภา',
    'ม': 'มอม้า', 'ย': 'ยอยักษ์', 'ร': 'รอเรือ', 'ล': 'ลอลิง',
    'ว': 'วอแหวน', 'ศ': 'สอศาลา', 'ษ': 'สอฤๅษี', 'ส': 'สอเสือ',
    'ห': 'หอหีบ', 'ฬ': 'ลอจุฬา', 'อ': 'อออ่าง', 'ฮ': 'ฮอนกฮูก',
}
_LETTERS = re.compile('[' + ''.join(THAI_CONSONANT_NAMES) + ']')
_LESSON_NUMBER = re.compile(r'(พยัญชนะตัวที่[ \t]*)([0-9๐-๙]{1,2})(?!\w|[\u0e31\u0e34-\u0e3a\u0e47-\u0e4e])')
_DIGITS = ('', 'หนึ่ง', 'สอง', 'สาม', 'สี่', 'ห้า', 'หก', 'เจ็ด', 'แปด', 'เก้า')


def _lesson_number(match):
    number = int(match[2])
    if not 1 <= number <= 44:
        return match[0]
    tens, units = divmod(number, 10)
    prefix = '' if not tens else 'สิบ' if tens == 1 else 'ยี่สิบ' if tens == 2 else _DIGITS[tens]+'สิบ'
    suffix = 'เอ็ด' if units == 1 and tens else _DIGITS[units]
    return match[1] + prefix + suffix


def expand_thai_speech_units(text):
    def consonant(match):
        # เครื่องหมายที่ประกอบกับตัวหลักเป็นส่วนของตัวหลัก แม้ Unicode \w จะไม่นับ
        neighbors = (text[match.start()-1] if match.start() else '', text[match.end():match.end()+1])
        if any(c and (c.isalpha() or unicodedata.category(c).startswith('M')) for c in neighbors):
            return match[0]
        return THAI_CONSONANT_NAMES[match[0]]
    return _LESSON_NUMBER.sub(_lesson_number, _LETTERS.sub(consonant, text))


# ชื่อที่พูดสำหรับสัญลักษณ์เดียวใต้ปลายนิ้ว พยัญชนะใช้ THAI_CONSONANT_NAMES
SYMBOL_NAMES = {
    '่': 'ไม้เอก', '้': 'ไม้โท', '๊': 'ไม้ตรี', '๋': 'ไม้จัตวา', '์': 'ไม้ทัณฑฆาต',
    '็': 'ไม้ไต่คู้', 'ะ': 'สระอะ', 'ั': 'ไม้หันอากาศ', 'า': 'สระอา', 'ำ': 'สระอำ',
    'ิ': 'สระอิ', 'ี': 'สระอี', 'ึ': 'สระอึ', 'ื': 'สระอือ', 'ุ': 'สระอุ', 'ู': 'สระอู',
    'เ': 'สระเอ', 'แ': 'สระแอ', 'โ': 'สระโอ', 'ใ': 'สระใอ', 'ไ': 'สระไอ',
    'เ◌อ': 'สระเออ', 'เ◌า': 'สระเอา', 'เ◌ีย': 'สระเอีย', 'เ◌ือ': 'สระเอือ', '◌ัว': 'สระอัว',
}


# คำสะกดที่ส่งให้เสียง Piper สำหรับชื่อที่ออกเสียงผิด (ทีมฟังเอง)
# ชื่อในส่วนอื่นคงเดิม ช่องว่างแยกพยางค์ที่เสียงพูดติดกัน
TTS_SPELLING = {
    'งองู': 'งอ งู', 'ชอช้าง': 'ชอ ช้าง', 'ชอเฌอ': 'ชอ เชอ', 'ถอฐาน': 'ถอ สัน ถาน',
    'ทอมณโฑ': 'ทอ นาง มน โท', 'รอเรือ': 'รอ เรือ', 'หอหีบ': 'หอ หีบ', 'อออ่าง': 'ออ อ่าง',
    'ฮอนกฮูก': 'ฮอ นก ฮูก', 'ไม้เอก': 'ไม้ เอก', 'ไม้โท': 'ไม้ โท', 'ไม้จัตวา': 'ไม้ จัด ตา วา',
    **{name: 'สะ ละ ' + name[3:] for name in SYMBOL_NAMES.values() if name.startswith('สระ')},
}


# แบบเดียวกันสำหรับเสียง Edge ซึ่งอ่านชื่อส่วนใหญ่ถูก (ทีมฟังเอง)
EDGE_SPELLING = {
    'ผอผึ้ง': 'พ๋อ ผึ้ง', 'ฮอนกฮูก': 'ฮอ นก ฮูก',
    **{name: 'สะ ละ ' + name[3:] for name in SYMBOL_NAMES.values() if name.startswith('สระ')},
}


def tts_spelling(text, spelling=TTS_SPELLING):
    """ข้อความที่ใช้สังเคราะห์ของ ``text``: เปลี่ยนชื่อที่รู้จักทุกตัวเป็นคำสะกดของเสียงนั้น"""
    return ' '.join(spelling.get(part, part) for part in text.split(' '))


def symbol_name(text):
    """ชื่อที่พูดของสัญลักษณ์เบรลล์หนึ่งตัว ข้อความที่ไม่รู้จักจะพูดตามเดิม"""
    return SYMBOL_NAMES.get(text) or THAI_CONSONANT_NAMES.get(text, text)
