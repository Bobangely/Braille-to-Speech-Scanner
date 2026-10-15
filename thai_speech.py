"""Speech-only names for independent Thai consonants, without TTS dependencies."""
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
        # Combining marks belong to their base even when Unicode \w excludes them.
        neighbors = (text[match.start()-1] if match.start() else '', text[match.end():match.end()+1])
        if any(c and (c.isalpha() or unicodedata.category(c).startswith('M')) for c in neighbors):
            return match[0]
        return THAI_CONSONANT_NAMES[match[0]]
    return _LESSON_NUMBER.sub(_lesson_number, _LETTERS.sub(consonant, text))
