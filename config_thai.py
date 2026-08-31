"""
Thai Braille Configuration (Standard: thisAble.me / Thailand Association of the Blind / Genevieve Caulfield)
==========================================================================================================
มาตรฐานอักษรเบรลล์ไทยสากล (ระบบเบรลล์ไทยมาตรฐาน)

Braille Cell Layout:
    (1) (4)
    (2) (5)
    (3) (6)

การแยก Dictionary:
  - THAI_CONSONANTS          = พยัญชนะ single-cell (ก-ฮ ที่ไม่ต้องมี prefix)
  - THAI_CONSONANTS_PREFIX6  = พยัญชนะที่ต้องมี prefix จุด 6
  - THAI_CONSONANTS_PREFIX36 = พยัญชนะที่ต้องมี prefix จุด 3,6
  - THAI_CONSONANTS_PREFIX356= พยัญชนะที่ต้องมี prefix จุด 3,5,6
  - THAI_VOWELS              = สระ
  - THAI_TONE_MARKS          = วรรณยุกต์ 4 ตัว (มาตรฐานไทย)
  - THAI_SPECIAL_MARKS       = การันต์, ไม้ยมก, ไม้ไต่คู้, ไปยาลน้อย
  - THAI_DIGIT_MAP           = ตัวเลข 0-9
"""

# =============================================================================
# ส่วนที่ 1: พยัญชนะ (Consonants) — Single Cell (ตรงตามตารางมาตรฐานไทย 100%)
# =============================================================================
THAI_CONSONANTS = {
    # --- ก - ง ---
    frozenset({1, 2, 4, 5}):       'ก',    # k (dots 1,2,4,5)
    frozenset({1, 3}):             'ข',    # kh (high, dots 1,3)
    frozenset({1, 3, 6}):          'ค',    # kh (low, dots 1,3,6)
    frozenset({1, 2, 4, 5, 6}):    'ง',    # ng (dots 1,2,4,5,6)

    # --- จ - ซ ---
    frozenset({2, 4, 5}):          'จ',    # ch (dots 2,4,5)
    frozenset({3, 4}):             'ฉ',    # ch (high, dots 3,4)
    frozenset({3, 4, 6}):          'ช',    # ch (low, dots 3,4,6)
    frozenset({2, 3, 4, 6}):       'ซ',    # s (low, dots 2,3,4,6)

    # --- ด - น ---
    frozenset({1, 4, 5}):          'ด',    # d (dots 1,4,5)
    frozenset({1, 2, 5, 6}):       'ต',    # t (dots 1,2,5,6)
    frozenset({2, 3, 4, 5}):       'ถ',    # th (high, dots 2,3,4,5)
    frozenset({2, 3, 4, 5, 6}):    'ท',    # th (low, dots 2,3,4,5,6)
    frozenset({1, 3, 4, 5}):       'น',    # n (dots 1,3,4,5)

    # --- บ - ม ---
    frozenset({1, 2, 3, 6}):       'บ',    # b (dots 1,2,3,6)
    frozenset({1, 2, 3, 4, 6}):    'ป',    # p (dots 1,2,3,4,6)
    frozenset({1, 2, 3, 4}):       'ผ',    # ph (high, dots 1,2,3,4)
    frozenset({1, 3, 4, 6}):       'ฝ',    # f (high, dots 1,3,4,6)
    frozenset({1, 4, 5, 6}):       'พ',    # ph (low, dots 1,4,5,6)
    frozenset({1, 2, 4, 6}):       'ฟ',    # f (low, dots 1,2,4,6)
    frozenset({1, 3, 4}):          'ม',    # m (dots 1,3,4)

    # --- ย - ฮ ---
    frozenset({1, 3, 4, 5, 6}):    'ย',    # y (dots 1,3,4,5,6)
    frozenset({1, 2, 3, 5}):       'ร',    # r (dots 1,2,3,5)
    frozenset({1, 2, 3}):          'ล',    # l (dots 1,2,3)
    frozenset({2, 4, 5, 6}):       'ว',    # w (dots 2,4,5,6)
    frozenset({2, 3, 4}):          'ศ',    # s (high, dots 2,3,4) — หมายเหตุ: ศ/ส ในตารางใช้ dots 2,3,4
    frozenset({1, 2, 5}):          'ห',    # h (high, dots 1,2,5)
    frozenset({1, 3, 5}):          'อ',    # o (dots 1,3,5)
    frozenset({1, 2, 3, 4, 5, 6}): 'ฮ',    # h (low, dots 1,2,3,4,5,6)
}

# พยัญชนะที่มี prefix (2 เซลล์)
# Prefix dot-6 (⠠)
THAI_CONSONANTS_PREFIX6 = {
    frozenset({1, 3, 6}):          'ฆ',    # 6 + ค
    frozenset({3, 4, 6}):          'ฌ',    # 6 + ช
    frozenset({1, 3, 4, 5, 6}):    'ญ',    # 6 + ย
    frozenset({1, 4, 5}):          'ฎ',    # 6 + ด
    frozenset({1, 2, 5, 6}):       'ฏ',    # 6 + ต
    frozenset({2, 3, 4, 5}):       'ฐ',    # 6 + ถ
    frozenset({2, 3, 4, 5, 6}):    'ฑ',    # 6 + ท
    frozenset({1, 3, 4, 5}):       'ณ',    # 6 + น
    frozenset({1, 4, 5, 6}):       'ภ',    # 6 + พ
    frozenset({2, 3, 4}):          'ศ',    # 6 + ส (ศ)
    frozenset({1, 2, 3}):          'ฬ',    # 6 + ล
}

# Prefix dot-36 (⠤)
THAI_CONSONANTS_PREFIX36 = {
    frozenset({1, 3, 6}):          'ฅ',    # 36 + ค
    frozenset({2, 3, 4, 5, 6}):    'ฒ',    # 36 + ท
    frozenset({2, 3, 4}):          'ษ',    # 36 + ศ (ษ)
}

# Prefix dot-356 (⠴)
THAI_CONSONANTS_PREFIX356 = {
    frozenset({1, 3}):             'ฃ',    # 356 + ข
    frozenset({2, 3, 4, 5, 6}):    'ธ',    # 356 + ท
}

# เพิ่ม alias ส ให้ใช้ dots {2,3,4} โดยตรง
THAI_CONSONANTS[frozenset({2, 3, 4})] = 'ส'

# =============================================================================
# ส่วนที่ 2: สระ (Vowels) — ตรงตามตารางมาตรฐานไทย 100%
# =============================================================================
THAI_VOWELS = {
    # --- สระเดี่ยว ---
    frozenset({1}):                'ะ',     # สระอะ (dots 1)
    frozenset({3, 4, 5}):          'ั',     # ไม้หันอากาศ (dots 3,4,5)
    frozenset({1, 6}):             'า',     # สระอา (dots 1,6)
    frozenset({1, 2}):             'ิ',     # สระอิ (dots 1,2)
    frozenset({2, 3}):             'ี',     # สระอี (dots 2,3)
    frozenset({2, 6}):             'ึ',     # สระอึ (dots 2,6)
    frozenset({2, 5, 6}):          'ื',     # สระอือ (dots 2,5,6)
    frozenset({1, 4}):             'ุ',     # สระอุ (dots 1,4)
    frozenset({2, 5}):             'ู',     # สระอู (dots 2,5)

    # --- สระนำ ---
    frozenset({1, 2, 4}):          'เ',     # สระเอ (dots 1,2,4)
    frozenset({1, 2, 6}):          'แ',     # สระแอ (dots 1,2,6)
    frozenset({2, 4}):             'โ',     # สระโอ (dots 2,4)
    frozenset({1, 5, 6}):          'ไ',     # สระไอ (dots 1,5,6)
    # สระ 'ใ' = 2 cells: dots 1,5,6 + dot 2 (จัดการใน decoder)

    # --- สระผสมเดี่ยว (Single-cell representations) ---
    frozenset({1, 3, 5, 6}):       'ำ',     # สระอำ (dots 1,3,5,6)
    frozenset({2, 3, 5}):          'เ◌า',   # สระเอา (dots 2,3,5)
    frozenset({1, 4, 6}):          'เ◌อ',   # สระเออ (dots 1,4,6)
    frozenset({1, 2, 3, 5, 6}):    'เ◌ิ◌',  # สระเอิ (dots 1,2,3,5,6)
    frozenset({4, 5, 6}):          'เ◌ีย',  # สระเอีย (dots 4,5,6)
    frozenset({1, 5}):             'เ◌ือ',  # สระเอือ (dots 1,5)
    frozenset({4, 5}):             '◌ัว',   # สระอัว (dots 4,5)
}

# =============================================================================
# ส่วนที่ 3: วรรณยุกต์ (Tone marks) — ตามตารางมาตรฐานไทย (thisAble.me / มูลนิธิฯ)
# =============================================================================
THAI_TONE_MARKS = {
    frozenset({3}):                '่',     # ไม้เอก (dot 3)
    frozenset({3, 6}):             '้',     # ไม้โท (dots 3,6)
    frozenset({2, 3, 6}):          '๊',     # ไม้ตรี (dots 2,3,6)
    frozenset({2, 3, 5, 6}):       '๋',     # ไม้จัตวา (dots 2,3,5,6)
}

# =============================================================================
# ส่วนที่ 4: เครื่องหมายพิเศษ (Special symbols) — ตามตารางมาตรฐานไทย
# =============================================================================
THAI_SPECIAL_MARKS = {
    frozenset({3, 5, 6}):          '์',     # การันต์ / ทัณฑฆาต (dots 3,5,6)
    frozenset({3, 5}):             '็',     # ไม้ไต่คู้ (dots 3,5)
    frozenset({2}):                'ๆ',     # ไม้ยมก (dot 2)
    frozenset({5, 6}):             'ฯ',     # ไปยาลน้อย (dots 5,6)
}

# =============================================================================
# ส่วนที่ 5: ตัวเลข (Digits: 0-9)
# =============================================================================
THAI_DIGIT_MAP = {
    frozenset({1}):                '1',
    frozenset({1, 2}):             '2',
    frozenset({1, 4}):             '3',
    frozenset({1, 4, 5}):          '4',
    frozenset({1, 5}):             '5',
    frozenset({1, 2, 4}):          '6',
    frozenset({1, 2, 4, 5}):       '7',
    frozenset({1, 2, 5}):          '8',
    frozenset({2, 4}):             '9',
    frozenset({2, 4, 5}):          '0',
}

# =============================================================================
# ส่วนที่ 6: Prefix Markers
# =============================================================================
PREFIX_6   = frozenset({6})
PREFIX_36  = frozenset({3, 6})
PREFIX_356 = frozenset({3, 5, 6})
NUMBER_INDICATOR = frozenset({3, 4, 5, 6})

# =============================================================================
# ส่วนที่ 7: Context-Aware Lookup Helpers
# =============================================================================
CONSONANT_KEYS = set(THAI_CONSONANTS.keys())
VOWEL_KEYS = set(THAI_VOWELS.keys())
TONE_KEYS = set(THAI_TONE_MARKS.keys())
SPECIAL_KEYS = set(THAI_SPECIAL_MARKS.keys())

LEADING_VOWELS = {'เ', 'แ', 'โ', 'ไ', 'ใ'}
COMPOUND_VOWELS = {'เ◌า', 'เ◌อ', 'เ◌ิ◌', 'เ◌ีย', 'เ◌ือ', '◌ัว'}
COMBINING_VOWELS = {'ะ', 'ั', 'า', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'ำ'}

THAI_BRAILLE_TO_CHAR = {}
THAI_BRAILLE_TO_CHAR.update(THAI_CONSONANTS)
THAI_BRAILLE_TO_CHAR.update(THAI_VOWELS)
THAI_BRAILLE_TO_CHAR.update(THAI_TONE_MARKS)
THAI_BRAILLE_TO_CHAR.update(THAI_SPECIAL_MARKS)

# =============================================================================
# ส่วนที่ 8: Reverse mapping (Char -> Braille dots)
# =============================================================================
THAI_CHAR_TO_BRAILLE = {}
for dots, ch in THAI_CONSONANTS.items():
    THAI_CHAR_TO_BRAILLE[ch] = dots
for dots, ch in THAI_CONSONANTS_PREFIX6.items():
    THAI_CHAR_TO_BRAILLE[ch] = (frozenset({6}), dots)
for dots, ch in THAI_CONSONANTS_PREFIX36.items():
    THAI_CHAR_TO_BRAILLE[ch] = (frozenset({3, 6}), dots)
for dots, ch in THAI_CONSONANTS_PREFIX356.items():
    THAI_CHAR_TO_BRAILLE[ch] = (frozenset({3, 5, 6}), dots)
for dots, ch in THAI_VOWELS.items():
    THAI_CHAR_TO_BRAILLE[ch] = dots
for dots, ch in THAI_TONE_MARKS.items():
    THAI_CHAR_TO_BRAILLE[ch] = dots
for dots, ch in THAI_SPECIAL_MARKS.items():
    THAI_CHAR_TO_BRAILLE[ch] = dots

# สระ 'ใ' (dots 156 + dot 2)
THAI_CHAR_TO_BRAILLE['ใ'] = (frozenset({1, 5, 6}), frozenset({2}))
