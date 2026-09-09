"""
Braille Reader - Configuration
================================
อักษรเบรลล์ Grade 1 (English) mapping

Braille Cell Layout:
    (1) (4)
    (2) (5)
    (3) (6)

แต่ละ cell มี 6 ตำแหน่ง จุดที่มี/ไม่มี จะกำหนดตัวอักษร
"""

# =============================================================================
# Braille Grade 1 - ตัวอักษร a-z
# key = frozenset ของเลข dot ที่มีอยู่
# value = ตัวอักษรที่ตรงกัน
# =============================================================================
BRAILLE_TO_CHAR = {
    frozenset({1}):          'a',
    frozenset({1, 2}):       'b',
    frozenset({1, 4}):       'c',
    frozenset({1, 4, 5}):    'd',
    frozenset({1, 5}):       'e',
    frozenset({1, 2, 4}):    'f',
    frozenset({1, 2, 4, 5}): 'g',
    frozenset({1, 2, 5}):    'h',
    frozenset({2, 4}):       'i',
    frozenset({2, 4, 5}):    'j',
    frozenset({1, 3}):       'k',
    frozenset({1, 2, 3}):    'l',
    frozenset({1, 3, 4}):    'm',
    frozenset({1, 3, 4, 5}): 'n',
    frozenset({1, 3, 5}):    'o',
    frozenset({1, 2, 3, 4}): 'p',
    frozenset({1, 2, 3, 4, 5}): 'q',
    frozenset({1, 2, 3, 5}): 'r',
    frozenset({2, 3, 4}):    's',
    frozenset({2, 3, 4, 5}): 't',
    frozenset({1, 3, 6}):    'u',
    frozenset({1, 2, 3, 6}): 'v',
    frozenset({2, 4, 5, 6}): 'w',
    frozenset({1, 3, 4, 6}): 'x',
    frozenset({1, 3, 4, 5, 6}): 'y',
    frozenset({1, 3, 5, 6}): 'z',
    # สัญลักษณ์พิเศษ
    frozenset({3, 4, 5, 6}): '#',   # number indicator
    frozenset({6}):          ',',
    frozenset({2, 6}):       '!',
    frozenset({2, 3, 6}):    '?',
    frozenset({2, 5, 6}):    '.',
}

# Reverse mapping: ตัวอักษร -> dot positions
CHAR_TO_BRAILLE = {v: k for k, v in BRAILLE_TO_CHAR.items()}
