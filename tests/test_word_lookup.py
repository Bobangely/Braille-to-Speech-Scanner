"""Reading a word from the dictionary by its braille cells."""
import unittest
from unittest import mock

import thai_lexicon
from text_to_braille import text_to_braille_cells


def cells(text):
    return tuple(frozenset(c or ()) for c in text_to_braille_cells(text, 'thai'))


class LookupTests(unittest.TestCase):
    def test_whole_word(self):
        self.assertEqual(thai_lexicon.word_from_cells(cells('พระองค์')), 'พระองค์')

    def test_compound_split_into_dictionary_words(self):
        self.assertEqual(thai_lexicon.word_from_cells(cells('เสด็จไป')), 'เสด็จไป')

    def test_unknown_word_is_not_guessed(self):
        self.assertIsNone(thai_lexicon.word_from_cells([frozenset({1, 2, 3, 4, 5, 6})] * 3))

    def test_no_word_list_means_no_lookup(self):
        thai_lexicon._cell_table.cache_clear()
        self.addCleanup(thai_lexicon._cell_table.cache_clear)
        with mock.patch('thai_lexicon.default_lexicon', return_value=frozenset()):
            self.assertIsNone(thai_lexicon.word_from_cells(cells('พระองค์')))

    def test_no_two_dictionary_words_share_their_cells(self):
        table = thai_lexicon._cell_table()
        self.assertEqual(len(table), sum(map(thai_lexicon._is_plain_word, thai_lexicon.default_lexicon())))


if __name__ == '__main__':
    unittest.main()
