"""Thai uncontracted Braille tables for NEW input and datasets.

Consonants and signs: liblouis tables/th-g0.utb (2023-06-01).
Compound vowels: World Braille Usage, 3rd ed. (2013), printed pp. 140-141.
Legacy synthetic images use config_thai_legacy explicitly; never mix labels.
"""

from config_thai_legacy import (
    THAI_CONSONANTS, THAI_CONSONANTS_PREFIX6, THAI_CONSONANTS_PREFIX36,
    THAI_CONSONANTS_PREFIX356, THAI_DIGIT_MAP, PREFIX_6, PREFIX_36,
    PREFIX_356, NUMBER_INDICATOR,
)


def dots(numbers):
    return frozenset(int(n) for n in numbers)


THAI_VOWELS = {dots(k): v for k, v in {
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
