"""Keep Thai short-i plus tone distinct from long-i through decoding and TTS."""

import unittest

from decoder import decode_cells, decode_cells_verbose, normalize_thai_text


def cells(patterns):
    # Explicit patterns, not reverse-generated from the mapping under test.
    return [dict(dots=frozenset(map(int, pattern))) for pattern in patterns]


WORDS = [('วิ่ง', '2456'), ('กิ่ง', '1245'), ('นิ่ง', '1345'),
         ('สิ่ง', '234'), ('ยิ่ง', '13456')]
SENTENCE = 'คุณแม่ตกใจวิ่งหนีจนลืม'
SENTENCE_PATTERNS = [
    '136', '14', '6', '1345',       # คุณ
    '126', '134', '35',             # แม่
    '1256', '1245', '156', '2', '245',  # ตกใจ
    '2456', '12', '35', '12456',    # วิ่ง
    '125', '1345', '23',            # หนี
    '245', '1345', '123', '26', '134',  # จนลืม
]


class ThaiCompositionTests(unittest.TestCase):
    def test_short_i_and_mai_ek_words(self):
        for expected, consonant in WORDS:
            for marks in [('12', '35'), ('35', '12')]:
                with self.subTest(word=expected, marks=marks):
                    result = decode_cells(cells([consonant, *marks, '12456']), 'thai')
                    self.assertEqual(result, expected)
                    self.assertEqual([ord(c) for c in result[1:3]], [0x0E34, 0x0E48])
                    self.assertNotIn('\u0e35', result)

    def test_verbose_tokens_preserve_vowel_and_tone_separately(self):
        # Same C13-C16 patterns as the captured diag_023 manifest.
        tokens = decode_cells_verbose(cells(['2456', '12', '35', '12456']), 'thai')
        self.assertEqual([token['char'] for token in tokens], ['ว', 'ิ', '่', 'ง'])
        self.assertTrue(all(not token['consumed'] for token in tokens))
        self.assertTrue(all(token['warning'] is None for token in tokens))

    def test_complete_sentence(self):
        for language in ('thai', 'th'):
            with self.subTest(language=language):
                self.assertEqual(decode_cells(cells(SENTENCE_PATTERNS), language), SENTENCE)

    def test_normalization_preserves_marks_and_is_idempotent(self):
        for word, _ in WORDS:
            for source in (word, word.replace('ิ่', '่ิ')):
                with self.subTest(source=source):
                    result = normalize_thai_text(source)
                    self.assertEqual(result, word)
                    self.assertEqual(normalize_thai_text(result), result)

    def test_long_i_is_not_guessed_to_be_short_i_with_tone(self):
        for patterns, expected in [(['2456', '23', '12456'], 'วีง'),
                                   (['2456', '23', '35', '12456'], 'วี่ง'),
                                   (['2456', '12', '12456'], 'วิง')]:
            with self.subTest(expected=expected):
                self.assertEqual(decode_cells(cells(patterns), 'thai'), expected)
                self.assertEqual(normalize_thai_text(expected), expected)

if __name__ == '__main__':
    unittest.main()
