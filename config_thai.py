"""ตารางเบรลล์ไทยแบบไม่ย่อ สำหรับข้อความใหม่และชุดข้อมูล

พยัญชนะและเครื่องหมาย: liblouis tables/th-g0.utb (2023-06-01)
สระประสม: World Braille Usage, 3rd ed. (2013), printed pp. 140-141
"""


def dots(numbers):
    return frozenset(int(n) for n in numbers)


THAI_CONSONANTS = {dots(k): v for k, v in {
    '1245': 'ก', '13': 'ข', '136': 'ค', '12456': 'ง',
    '245': 'จ', '34': 'ฉ', '346': 'ช', '2346': 'ซ',
    '145': 'ด', '1256': 'ต', '2345': 'ถ', '23456': 'ท', '1345': 'น',
    '1236': 'บ', '12346': 'ป', '1234': 'ผ', '1346': 'ฝ', '1456': 'พ', '1246': 'ฟ', '134': 'ม',
    '13456': 'ย', '1235': 'ร', '123': 'ล', '2456': 'ว', '234': 'ส', '125': 'ห', '135': 'อ', '123456': 'ฮ',
}.items()}
# พยัญชนะ 2 เซลล์: เซลล์นำ (prefix) ตามด้วยเซลล์ในตารางนี้
PREFIX_6, PREFIX_36, PREFIX_356 = dots('6'), dots('36'), dots('356')
THAI_CONSONANTS_PREFIX6 = {dots(k): v for k, v in {
    '136': 'ฆ', '346': 'ฌ', '13456': 'ญ', '145': 'ฎ', '1256': 'ฏ', '2345': 'ฐ',
    '23456': 'ฑ', '1345': 'ณ', '1456': 'ภ', '234': 'ศ', '123': 'ฬ',
}.items()}
THAI_CONSONANTS_PREFIX36 = {dots(k): v for k, v in {'136': 'ฅ', '23456': 'ฒ', '234': 'ษ'}.items()}
THAI_CONSONANTS_PREFIX356 = {dots(k): v for k, v in {'13': 'ฃ', '23456': 'ธ'}.items()}
NUMBER_INDICATOR = dots('3456')
THAI_DIGIT_MAP = {dots(k): v for k, v in {
    '1': '1', '12': '2', '14': '3', '145': '4', '15': '5',
    '124': '6', '1245': '7', '125': '8', '24': '9', '245': '0',
}.items()}
THAI_VOWELS ={dots(k): v for k, v in {
    '1': 'ะ', '345': 'ั', '16': 'า', '12': 'ิ', '23': 'ี',
    '246': 'ึ', '26': 'ื', '14': 'ุ', '25': 'ู',
    '124': 'เ', '126': 'แ', '24': 'โ', '156': 'ไ', '1356': 'ำ',
    '235': 'เ◌า', '146': 'เ◌อ', '12356': 'เ◌ีย',
    '12345': 'เ◌ือ', '15': '◌ัว',
}.items()}
THAI_TONE_MARKS = {dots(k): v for k, v in {
    '35': '่', '256': '้', '2356': '๊', '236': '๋',
}.items()}
THAI_SPECIAL_MARKS = {dots(k): v for k, v in {
    '356': '์', '3': '็', '2': 'ๆ', '5': 'ํ',
    # ขีดกลางเดี่ยว ถ้ามีพยัญชนะที่ขึ้นต้นด้วย 36 ครบอยู่ติดกันจะใช้พยัญชนะก่อน
    '36': '-',
}.items()}
THAI_MULTI_CELL = {}
for prefix, mapping in [(PREFIX_6, THAI_CONSONANTS_PREFIX6),
                        (PREFIX_36, THAI_CONSONANTS_PREFIX36),
                        (PREFIX_356, THAI_CONSONANTS_PREFIX356)]:
    THAI_MULTI_CELL.update({(prefix, pattern): char for pattern, char in mapping.items()})
THAI_MULTI_CELL.update({(dots(a), dots(b)): char for a, b, char in [
    ('156', '2', 'ใ'), ('1235', '2', 'ฤ'), ('123', '2', 'ฦ'),
    ('56', '23', 'ฯ'), ('4', '12', '฿'), ('5', '16', 'ๅ'),
]})
# หนังสือใช้เครื่องหมายวรรคตอนแบบมีตัวนำของไทย (Liblouis tables/th-g1.utb)
# คงคำศัพท์ชุดข้อมูลจำลองเดิม และเพิ่มเฉพาะตารางสำหรับอ่าน
THAI_READING_MULTI_CELL = dict(THAI_MULTI_CELL)
THAI_READING_MULTI_CELL.update({(dots('456'), dots(body)): char for body, char in {
    '2': ',', '23': ';', '25': ':', '256': '.', '235': '!', '236': '?',
}.items()})

# ข้อความที่แสดงกับชื่อที่พูดใช้นิยามสัญลักษณ์ชุดเดียวกัน เพิ่มเข้า
# ตารางอ่านเดิม เพื่อให้ทุกส่วนเห็นการจับคู่หลายเซลล์แบบเดียวกัน
THAI_MULTI_CELL_SYMBOLS = {
    (dots('5'), dots('126')): dict(
        id='open_parenthesis', output='(', tts='วงเล็บเปิด', type='punctuation'),
    (dots('5'), dots('345')): dict(
        id='close_parenthesis', output=')', tts='วงเล็บปิด', type='punctuation'),
}
THAI_READING_MULTI_CELL.update({sequence: symbol['output']
                               for sequence, symbol in THAI_MULTI_CELL_SYMBOLS.items()})

LEADING_VOWELS = {'เ', 'แ', 'โ', 'ไ', 'ใ'}
COMPOUND_VOWELS = {'เ◌า', 'เ◌อ', 'เ◌ีย', 'เ◌ือ', '◌ัว'}
COMBINING_VOWELS = set(THAI_VOWELS.values()) - LEADING_VOWELS - COMPOUND_VOWELS
CONSONANT_KEYS = set(THAI_CONSONANTS)
VOWEL_KEYS = set(THAI_VOWELS)
TONE_KEYS = set(THAI_TONE_MARKS)
SPECIAL_KEYS = set(THAI_SPECIAL_MARKS)
THAI_BRAILLE_TO_CHAR = {**THAI_CONSONANTS, **THAI_VOWELS,
                       **THAI_TONE_MARKS, **THAI_SPECIAL_MARKS}
THAI_CHAR_TO_BRAILLE = {char: pattern for pattern, char in THAI_BRAILLE_TO_CHAR.items()}
THAI_CHAR_TO_BRAILLE.update({char: patterns for patterns, char in THAI_MULTI_CELL.items()})
