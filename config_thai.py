"""ตารางเบรลล์ไทย

หนึ่งเซลล์มี 6 จุด 
    (1) (4)
    (2) (5)
    (3) (6)

"""

THAI_CONSONANTS = {
    # ก - ง
    frozenset({1, 2, 4, 5}):       'ก',  # กอไก่
    frozenset({1, 3}):             'ข',  # ขอไข่
    frozenset({1, 3, 6}):          'ค',  # คอควาย
    frozenset({1, 2, 4, 5, 6}):    'ง',  # งองู
    # จ - ซ
    frozenset({2, 4, 5}):          'จ',  # จอจาน
    frozenset({3, 4}):             'ฉ',  # ฉอฉิ่ง
    frozenset({3, 4, 6}):          'ช',  # ชอช้าง
    frozenset({2, 3, 4, 6}):       'ซ',  # ซอโซ่
    # ด - น
    frozenset({1, 4, 5}):          'ด',  # ดอเด็ก
    frozenset({1, 2, 5, 6}):       'ต',  # ตอเต่า
    frozenset({2, 3, 4, 5}):       'ถ',  # ถอถุง
    frozenset({2, 3, 4, 5, 6}):    'ท',  # ทอทหาร
    frozenset({1, 3, 4, 5}):       'น',  # นอหนู
    # บ - ม
    frozenset({1, 2, 3, 6}):       'บ',  # บอใบไม้
    frozenset({1, 2, 3, 4, 6}):    'ป',  # ปอปลา
    frozenset({1, 2, 3, 4}):       'ผ',  # ผอผึ้ง
    frozenset({1, 3, 4, 6}):       'ฝ',  # ฝอฝา
    frozenset({1, 4, 5, 6}):       'พ',  # พอพาน
    frozenset({1, 2, 4, 6}):       'ฟ',  # ฟอฟัน
    frozenset({1, 3, 4}):          'ม',  # มอม้า
    # ย - ฮ
    frozenset({1, 3, 4, 5, 6}):    'ย',  # ยอยักษ์
    frozenset({1, 2, 3, 5}):       'ร',  # รอเรือ
    frozenset({1, 2, 3}):          'ล',  # ลอลิง
    frozenset({2, 4, 5, 6}):       'ว',  # วอแหวน
    frozenset({2, 3, 4}):          'ส',  # สอเสือ
    frozenset({1, 2, 5}):          'ห',  # หอหีบ
    frozenset({1, 3, 5}):          'อ',  # อออ่าง
    frozenset({1, 2, 3, 4, 5, 6}): 'ฮ',  # ฮอนกฮูก
}

# พยัญชนะ 2 เซลล์: เซลล์นำ (prefix) ตามด้วยเซลล์ในตารางนี้
PREFIX_6 = frozenset({6})
PREFIX_36 = frozenset({3, 6})
PREFIX_356 = frozenset({3, 5, 6})
THAI_CONSONANTS_PREFIX6 = {
    frozenset({1, 3, 6}):       'ฆ',  # คอระฆัง
    frozenset({3, 4, 6}):       'ฌ',  # ชอเฌอ
    frozenset({1, 3, 4, 5, 6}): 'ญ',  # ยอหญิง
    frozenset({1, 4, 5}):       'ฎ',  # ดอชฎา
    frozenset({1, 2, 5, 6}):    'ฏ',  # ตอปฏัก
    frozenset({2, 3, 4, 5}):    'ฐ',  # ถอฐาน
    frozenset({2, 3, 4, 5, 6}): 'ฑ',  # ทอมณโฑ
    frozenset({1, 3, 4, 5}):    'ณ',  # นอเณร
    frozenset({1, 4, 5, 6}):    'ภ',  # พอสำเภา
    frozenset({2, 3, 4}):       'ศ',  # สอศาลา
    frozenset({1, 2, 3}):       'ฬ',  # ลอจุฬา
}
THAI_CONSONANTS_PREFIX36 = {
    frozenset({1, 3, 6}):       'ฅ',  # คอคน
    frozenset({2, 3, 4, 5, 6}): 'ฒ',  # ทอผู้เฒ่า
    frozenset({2, 3, 4}):       'ษ',  # สอฤๅษี
}
THAI_CONSONANTS_PREFIX356 = {
    frozenset({1, 3}):          'ฃ',  # ขอขวด
    frozenset({2, 3, 4, 5, 6}): 'ธ',  # ทอธง
}

# ตัวเลข: เซลล์นำตัวเลข
NUMBER_INDICATOR = frozenset({3, 4, 5, 6})
THAI_DIGIT_MAP = {
    frozenset({1}):          '1',
    frozenset({1, 2}):       '2',
    frozenset({1, 4}):       '3',
    frozenset({1, 4, 5}):    '4',
    frozenset({1, 5}):       '5',
    frozenset({1, 2, 4}):    '6',
    frozenset({1, 2, 4, 5}): '7',
    frozenset({1, 2, 5}):    '8',
    frozenset({2, 4}):       '9',
    frozenset({2, 4, 5}):    '0',
}

THAI_VOWELS = {
    frozenset({1}):             'ะ',  # สระอะ
    frozenset({3, 4, 5}):       'ั',  # ไม้หันอากาศ
    frozenset({1, 6}):          'า',  # สระอา
    frozenset({1, 2}):          'ิ',  # สระอิ
    frozenset({2, 3}):          'ี',  # สระอี
    frozenset({2, 4, 6}):       'ึ',  # สระอึ
    frozenset({2, 6}):          'ื',  # สระอือ
    frozenset({1, 4}):          'ุ',  # สระอุ
    frozenset({2, 5}):          'ู',  # สระอู
    frozenset({1, 2, 4}):       'เ',  # สระเอ
    frozenset({1, 2, 6}):       'แ',  # สระแอ
    frozenset({2, 4}):          'โ',  # สระโอ
    frozenset({1, 5, 6}):       'ไ',  # สระไอ
    frozenset({1, 3, 5, 6}):    'ำ',  # สระอำ
    frozenset({2, 3, 5}):       'เ◌า',  # สระเอา
    frozenset({1, 4, 6}):       'เ◌อ',  # สระเออ
    frozenset({1, 2, 3, 5, 6}): 'เ◌ีย',  # สระเอีย
    frozenset({1, 2, 3, 4, 5}): 'เ◌ือ',  # สระเอือ
    frozenset({1, 5}):          '◌ัว',  # สระอัว
}

THAI_TONE_MARKS = {
    frozenset({3, 5}):       '่',  # ไม้เอก
    frozenset({2, 5, 6}):    '้',  # ไม้โท
    frozenset({2, 3, 5, 6}): '๊',  # ไม้ตรี
    frozenset({2, 3, 6}):    '๋',  # ไม้จัตวา
}

THAI_SPECIAL_MARKS = {
    frozenset({3, 5, 6}): '์',  # ไม้ทัณฑฆาต
    frozenset({3}):       '็',  # ไม้ไต่คู้
    frozenset({2}):       'ๆ',  # ไม้ยมก
    frozenset({5}):       'ํ',  # นิคหิต
    frozenset({3, 6}):    '-',  # ขีดกลาง # ขีดกลางเดี่ยว ถ้ามีพยัญชนะที่ขึ้นต้นด้วย 36 ครบอยู่ติดกันจะใช้พยัญชนะก่อน
}

# ตัวอักษรหลายเซลล์: รวมพยัญชนะ 2 เซลล์จากตารางด้านบน แล้วเพิ่มตัวอื่น
THAI_MULTI_CELL = {}
for prefix, mapping in [(PREFIX_6, THAI_CONSONANTS_PREFIX6),
                        (PREFIX_36, THAI_CONSONANTS_PREFIX36),
                        (PREFIX_356, THAI_CONSONANTS_PREFIX356)]:
    THAI_MULTI_CELL.update({(prefix, pattern): char for pattern, char in mapping.items()})
THAI_MULTI_CELL.update({
    (frozenset({1, 5, 6}), frozenset({2})):    'ใ',  # สระใอ
    (frozenset({1, 2, 3, 5}), frozenset({2})): 'ฤ',  # รึ
    (frozenset({1, 2, 3}), frozenset({2})):    'ฦ',  # ลึ
    (frozenset({5, 6}), frozenset({2, 3})):    'ฯ',  # ไปยาลน้อย
    (frozenset({4}), frozenset({1, 2})):       '฿',  # เครื่องหมายบาท
    (frozenset({5}), frozenset({1, 6})):       'ๅ',  # ลากข้าง
})

# หนังสือใช้เครื่องหมายวรรคตอนแบบมีตัวนำของไทย 
# คงคำศัพท์ชุดข้อมูลจำลองเดิม และเพิ่มเฉพาะตารางสำหรับอ่าน
THAI_READING_MULTI_CELL = dict(THAI_MULTI_CELL)
THAI_READING_MULTI_CELL.update({
    (frozenset({4, 5, 6}), frozenset({2})):       ',',  # จุลภาค
    (frozenset({4, 5, 6}), frozenset({2, 3})):    ';',  # อัฒภาค
    (frozenset({4, 5, 6}), frozenset({2, 5})):    ':',  # ทวิภาค
    (frozenset({4, 5, 6}), frozenset({2, 5, 6})): '.',  # มหัพภาค
    (frozenset({4, 5, 6}), frozenset({2, 3, 5})): '!',  # อัศเจรีย์
    (frozenset({4, 5, 6}), frozenset({2, 3, 6})): '?',  # ปรัศนี
})

# ข้อความที่แสดงกับชื่อที่พูดใช้นิยามสัญลักษณ์ชุดเดียวกัน เพิ่มเข้า
THAI_MULTI_CELL_SYMBOLS = {
    (frozenset({5}), frozenset({1, 2, 6})): dict(
        id='open_parenthesis', output='(', tts='วงเล็บเปิด', type='punctuation'),
    (frozenset({5}), frozenset({3, 4, 5})): dict(
        id='close_parenthesis', output=')', tts='วงเล็บปิด', type='punctuation'),
}
THAI_READING_MULTI_CELL.update({sequence: symbol['output']  
                               for sequence, symbol in THAI_MULTI_CELL_SYMBOLS.items()})

THAI_BRAILLE_TO_CHAR = {**THAI_CONSONANTS, **THAI_VOWELS, # ** รวมหลาย dict เป็นตัวเดียว ใช้ค้นเซลล์เดี่ยว
                       **THAI_TONE_MARKS, **THAI_SPECIAL_MARKS}
THAI_CHAR_TO_BRAILLE = {char: pattern for pattern, char in THAI_BRAILLE_TO_CHAR.items()}
THAI_CHAR_TO_BRAILLE.update({char: patterns for patterns, char in THAI_MULTI_CELL.items()})
