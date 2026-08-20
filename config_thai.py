"""
Thai Braille Configuration
============================
อักษรเบรลล์ภาษาไทย — ตาราง mapping พยัญชนะ (ก-ฮ), สระ, และวรรณยุกต์

อ้างอิง:
- มาตรฐานอักษรเบรลล์ไทย (พัฒนาโดย Miss Genevieve Caulfield)
- Wikipedia: Thai and Lao Braille
- มูลนิธิช่วยคนตาบอดแห่งประเทศไทย ในพระบรมราชินูปถัมภ์

Braille Cell Layout (เหมือนระบบสากล):
    (1) (4)
    (2) (5)
    (3) (6)

หลักการ:
- พยัญชนะส่วนใหญ่ใช้ 1 เซลล์
- พยัญชนะบางตัว (อักษรที่ไม่ค่อยใช้) ใช้ 2 เซลล์ (มีจุดนำ dot-6 นำหน้า)
- สระเขียนตามลำดับเสียง (Linear) ไม่ใช่ตำแหน่งบน/ล่าง/หน้า/หลัง
- วรรณยุกต์เขียนหลังพยัญชนะ+สระ
"""

# =============================================================================
# จุดนำ (Prefix markers) — ใช้นำหน้าพยัญชนะ 2 เซลล์
# =============================================================================
THAI_PREFIX_DOT6 = frozenset({6})        # จุดนำสำหรับพยัญชนะกลุ่ม B

# =============================================================================
# ส่วนที่ 1: พยัญชนะ (Consonants) — กลุ่ม A: 1 เซลล์
# =============================================================================
THAI_CONSONANTS = {
    # --- กลุ่ม ก (เสียง /k/) ---
    frozenset({1, 2, 4, 5}):       'ก',    # ko kai
    frozenset({1, 3}):             'ข',    # kho khai
    frozenset({1, 3, 6}):          'ค',    # kho khwai

    # --- กลุ่ม ง (เสียง /ŋ/) ---
    frozenset({1, 2, 4, 5, 6}):    'ง',    # ngo ngu

    # --- กลุ่ม จ (เสียง /tɕ/) ---
    frozenset({2, 4, 5}):          'จ',    # cho chan
    frozenset({3, 4}):             'ฉ',    # cho ching
    frozenset({3, 4, 6}):          'ช',    # cho chang

    # --- กลุ่ม ซ (เสียง /s/) ---
    frozenset({2, 3, 4, 6}):       'ซ',    # so so

    # --- กลุ่ม ญ/ย (เสียง /j/) ---
    frozenset({1, 3, 5, 6}):       'ญ',    # yo ying
    frozenset({1, 3, 4, 5, 6}):    'ย',    # yo yak

    # --- กลุ่ม ด (เสียง /d/) ---
    frozenset({1, 4, 5}):          'ด',    # do dek

    # --- กลุ่ม ต (เสียง /t/) ---
    frozenset({2, 3, 4, 5}):       'ต',    # to tao
    frozenset({2, 3, 4, 5, 6}):    'ท',    # tho thahan

    # --- กลุ่ม น (เสียง /n/) ---
    frozenset({1, 3, 4, 5}):       'น',    # no nu
    frozenset({1, 3, 4}):          'ม',    # mo ma

    # --- กลุ่ม บ (เสียง /b/) ---
    frozenset({1, 2}):             'บ',    # bo baimai

    # --- กลุ่ม ป (เสียง /p/) ---
    frozenset({1, 2, 3, 4}):       'ป',    # po pla
    frozenset({1, 2, 3, 4, 5, 6}): 'พ',    # pho phan

    # --- กลุ่ม ฟ (เสียง /f/) ---
    frozenset({1, 4}):             'ฟ',    # fo fan

    # --- กลุ่ม ร ล (เสียง /r/ /l/) ---
    frozenset({1, 2, 3, 5}):       'ร',    # ro ruea
    frozenset({1, 2, 3}):          'ล',    # lo ling

    # --- กลุ่ม ว (เสียง /w/) ---
    frozenset({2, 4, 5, 6}):       'ว',    # wo waen

    # --- กลุ่ม ส (เสียง /s/) ---
    frozenset({2, 3, 4}):          'ส',    # so suea

    # --- กลุ่ม ห อ ฮ ---
    frozenset({1, 2, 5}):          'ห',    # ho hip
    frozenset({1, 5}):             'อ',    # o ang
    frozenset({2, 4, 6}):          'ฮ',    # ho nokhuk
}

# =============================================================================
# ส่วนที่ 1b: พยัญชนะ — กลุ่ม B: 2 เซลล์ (จุดนำ dot-6 + ตัวอักษร)
# =============================================================================
THAI_CONSONANTS_PREFIX6 = {
    frozenset({1, 3}):             'ฃ',    # kho khuad
    frozenset({1, 3, 6}):          'ฅ',    # kho khon
    frozenset({1, 2, 4, 5}):       'ฆ',    # kho rakhang
    frozenset({3, 4, 6}):          'ฌ',    # cho choe
    frozenset({1, 4, 5}):          'ฎ',    # do chada
    frozenset({2, 3, 4, 5}):       'ฏ',    # to patak
    frozenset({3, 4}):             'ฐ',    # tho than
    frozenset({2, 3, 4, 6}):       'ฑ',    # tho montho
    frozenset({2, 3, 4, 5, 6}):    'ฒ',    # tho phuthao
    frozenset({1, 3, 4, 5}):       'ณ',    # no nen
    frozenset({2, 3, 4}):          'ถ',    # tho thung
    frozenset({1, 4}):             'ธ',    # tho thong
    frozenset({1, 2, 3, 4}):       'ผ',    # pho phueng
    frozenset({1, 2, 4}):          'ฝ',    # fo fa
    frozenset({1, 2, 3, 4, 5, 6}): 'ภ',    # pho samphao
    frozenset({1, 5}):             'ศ',    # so sala
    frozenset({1, 2, 3, 5}):       'ษ',    # so ruesi
    frozenset({1, 2, 3}):          'ฬ',    # lo chula
}

# =============================================================================
# ส่วนที่ 2: สระ (Vowels) และเครื่องหมายพิเศษ
# =============================================================================
THAI_VOWELS = {
    # --- สระหลัง/บน/ล่างพยัญชนะ ---
    frozenset({1, 2, 3, 5}):       'ะ',     # สระอะ
    frozenset({3, 4, 5}):          'า',     # สระอา
    frozenset({3, 5}):             'ิ',     # สระอิ
    frozenset({3, 5, 6}):          'ี',     # สระอี
    frozenset({1, 4, 6}):          'ึ',     # สระอึ
    frozenset({3, 4, 6}):          'ื',     # สระอือ
    frozenset({1, 5, 6}):          'ุ',     # สระอุ
    frozenset({1, 2, 6}):          'ู',     # สระอู

    # --- สระหน้าพยัญชนะ ---
    frozenset({1, 2, 4}):          'เ',     # สระเอ (dots 1,2,4)
    frozenset({1, 2, 3, 6}):       'แ',     # สระแอ
    frozenset({2, 4, 6}):          'โ',     # สระโอ
    frozenset({2, 6}):             'ไ',     # สระไอ ไม้มลาย
    frozenset({3, 6}):             'ใ',     # สระใอ ไม้ม้วน

    # --- สระประสมและเครื่องหมายพิเศษ ---
    frozenset({2, 4}):             'ำ',     # สระอำ
    frozenset({1, 4, 5, 6}):       'ั',     # ไม้หันอากาศ
    frozenset({2, 5}):             '็',     # ไม้ไต่คู้
    frozenset({1, 2, 4, 6}):       '์',     # การันต์ (ทัณฑฆาต)
    frozenset({5, 6}):             'ๆ',     # ไม้ยมก
}

# =============================================================================
# ส่วนที่ 3: วรรณยุกต์ (Tone marks)
# =============================================================================
THAI_TONE_MARKS = {
    frozenset({3}):                '่',     # ไม้เอก
    frozenset({2, 5, 6}):          '้',     # ไม้โท (dots 2,5,6)
    frozenset({6}):                '้',     # ไม้โท (alternative dot 6)
    frozenset({3, 6}):             '๊',     # ไม้ตรี
    frozenset({2, 3, 6}):          '๋',     # ไม้จัตวา
}

# =============================================================================
# ส่วนที่ 4: ตัวเลข (Digits: 0-9)
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

# Lookup รวมสำหรับ single-cell
THAI_BRAILLE_TO_CHAR = {}
THAI_BRAILLE_TO_CHAR.update(THAI_VOWELS)
THAI_BRAILLE_TO_CHAR.update(THAI_CONSONANTS)

# Reverse mapping: Char -> Braille dots
THAI_CHAR_TO_BRAILLE = {}
for dots, ch in THAI_CONSONANTS.items():
    THAI_CHAR_TO_BRAILLE[ch] = dots
for dots, ch in THAI_CONSONANTS_PREFIX6.items():
    THAI_CHAR_TO_BRAILLE[ch] = (frozenset({6}), dots)
for dots, ch in THAI_VOWELS.items():
    THAI_CHAR_TO_BRAILLE[ch] = dots
for dots, ch in THAI_TONE_MARKS.items():
    THAI_CHAR_TO_BRAILLE[ch] = dots
