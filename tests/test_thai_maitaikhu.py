"""Mai taikhu requires a coda in this syllable, not the next onset."""

from copy import deepcopy
import unittest

from decoder import decode_cells, decode_cells_verbose
from thai_parser import compose_tokens
from thai_decoder import decode_thai_details, tokenize_thai


def book_cells(lines):
    """Explicit physical patterns; preserve real word gaps and line boundaries."""
    cells = []
    for line_id, words in enumerate(lines):
        x = 0
        for word_index, word in enumerate(words):
            if word_index:
                x += 95
            for pattern in word:
                cells.append(dict(dots=frozenset(map(int, pattern)), line_id=line_id,
                                  reading_x=x, cell_pitch=95, adjacency_limit=145,
                                  x=x, y=line_id*180, center=(x, line_id*180)))
                x += 95
    return cells


BOX = ['1245', '123', '35', '135', '12456']  # กล่อง
GIFT = ['13', '135', '12456', '13', '2456', '345', '6', '13456']  # ของขวัญ
BOOK = [
    [['1245', '126', '1'], ['1245', '126', '1', *BOX, *GIFT], ['156', '2', '1345']],
    [[*BOX, *GIFT, '134', '23', '1245', '1236']],
]
EXPECTED = 'แกะ แกะกล่องของขวัญ ใน\nกล่องของขวัญมีกบ'


class ThaiMaiTaikhuTests(unittest.TestCase):
    def test_open_short_vowel_before_next_syllable_with_written_o(self):
        next_words = [(['ก', 'ล', '่', 'อ', 'ง'], 'กล่อง'),
                      (['ข', 'อ'], 'ขอ'), (['บ', 'อ', 'ก'], 'บอก'),
                      (['อ', 'อ', 'ก'], 'ออก'), (['ท', '่', 'อ'], 'ท่อ')]
        for start, word in [(['ก', 'แ', 'ะ'], 'แกะ'), (['แ', 'ก', 'ะ'], 'แกะ'),
                            (['ก', 'แ-ะ'], 'แกะ'), (['ก', 'เ', 'ะ'], 'เกะ'),
                            (['ก', 'เ-ะ'], 'เกะ'), (['ก', 'ะ'], 'กะ'),
                            (['ก', 'เ-าะ'], 'เกาะ')]:
            for tail, following in next_words:
                with self.subTest(start=start, following=following):
                    tokens = start + tail
                    original = deepcopy(tokens)
                    result = compose_tokens(tokens)
                    self.assertEqual(result['text'], word+following)
                    self.assertNotIn('็', result['text'])
                    self.assertFalse(result['needs_review'])
                    self.assertEqual(result['syllables'][0]['final'], None)
                    self.assertEqual(result['raw_tokens'], original)
                    self.assertEqual(tokens, original)

    def test_written_o_and_tone_belong_to_the_following_cluster(self):
        result = compose_tokens(['ก', 'แ', 'ะ', 'ก', 'ล', '่', 'อ', 'ง'])
        self.assertEqual(result['text'], 'แกะกล่อง')
        self.assertEqual(len(result['syllables']), 2)
        second = result['syllables'][1]
        self.assertEqual((second['onset'], second['vowel'], second['tone'], second['final']),
                         ('กล', 'ออ', '่', 'ง'))
        self.assertFalse(any(a['value'] == '็' for a in result['normalized_tokens']))

    def test_closed_short_vowels_still_generate_required_mai_taikhu(self):
        for tokens, text in [(['ต', 'เ-ะ', 'ม'], 'เต็ม'),
                             (['ข', 'แ-ะ', 'ง'], 'แข็ง'),
                             (['ก', 'เ-ะ', 'บ'], 'เก็บ'),
                             (['น', 'เ-าะ', 'ต'], 'น็อต')]:
            with self.subTest(text=text):
                result = compose_tokens(tokens)
                self.assertEqual(result['text'], text)
                self.assertIn('็', result['text'])
                self.assertNotIn('็', result['raw_tokens'])
                self.assertIsNotNone(result['syllables'][0]['final'])
                self.assertFalse(result['needs_review'])

    def test_o_onset_in_next_syllable_does_not_steal_the_previous_coda(self):
        for tokens, text in [(['ต', 'เ-ะ', 'ม', 'อ', 'ิ', '่', 'ม'], 'เต็มอิ่ม'),
                             (['ข', 'แ-ะ', 'ง', 'อ', '่', 'อ', 'น'], 'แข็งอ่อน'),
                             (['ก', 'เ-ะ', 'บ', 'อ', 'อ', 'ม'], 'เก็บออม'),
                             (['ค', 'เ-อ', 'ย', 'อ', '่', 'อ', 'น'], 'เคยอ่อน')]:
            with self.subTest(text=text):
                result = compose_tokens(tokens)
                self.assertEqual(result['text'], text)
                self.assertFalse(result['needs_review'])
                self.assertEqual(result['raw_tokens'], tokens)
                self.assertIsNotNone(result['syllables'][0]['final'])

    def test_book_patterns_keep_raw_tokens_and_do_not_add_mai_taikhu(self):
        cells = book_cells(BOOK)
        original = deepcopy(cells)
        self.assertEqual(len(cells), 39)
        raw = tokenize_thai(cells)
        result = decode_thai_details(cells)
        self.assertEqual([t['char'] for t in raw[:3]], ['ก', 'แ', 'ะ'])
        self.assertFalse(any(t['char'] == '็' for t in raw))
        self.assertEqual(result['text'], EXPECTED)
        self.assertNotIn('็', result['text'])
        self.assertFalse(result['needs_review'])
        self.assertEqual(result['raw_tokens'], raw)
        self.assertEqual(decode_cells(cells, 'thai'), EXPECTED)
        self.assertEqual([t['char'] for t in decode_cells_verbose(cells, 'thai')],
                         [t['char'] for t in raw])
        self.assertEqual(cells, original)

    def test_physical_closed_and_explicit_mai_taikhu_patterns_are_preserved(self):
        for word, patterns in [('เต็ม', ['1256', '124', '1', '134']),
                               ('แข็ง', ['13', '126', '1', '12456']),
                               ('เต็ม', ['124', '1256', '3', '134']),
                               ('แข็ง', ['126', '13', '3', '12456'])]:
            with self.subTest(word=word, patterns=patterns):
                result = decode_thai_details(book_cells([[patterns]]))
                self.assertEqual(result['text'], word)
                self.assertIn('็', result['text'])
                self.assertFalse(result['needs_review'])

    def test_space_and_line_boundaries_do_not_supply_a_coda(self):
        for separator in (' ', '\n', '-'):
            with self.subTest(separator=separator):
                result = compose_tokens(['ก', 'แ-ะ', separator, 'ง'])
                self.assertEqual(result['text'], 'แกะ'+separator+'ง')
                self.assertNotIn('็', result['text'])


if __name__ == '__main__':
    unittest.main()
