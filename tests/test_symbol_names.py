"""Spoken names for single Braille symbols read under the fingertip."""
import unittest

from thai_speech import SYMBOL_NAMES, symbol_name


class SymbolNameTests(unittest.TestCase):
    def test_kai_symbols(self):
        self.assertEqual([symbol_name(c) for c in ('ไ', 'ก', '่')], ['สระไอ', 'กอไก่', 'ไม้เอก'])

    def test_all_vowels_and_tones_have_names(self):
        required = '่้๊๋์ะัาำิีึืุูเแโใไ็'
        self.assertEqual([c for c in required if c not in SYMBOL_NAMES], [])
        self.assertTrue(all(name.strip() for name in SYMBOL_NAMES.values()))

    def test_compound_vowels_from_decoder_have_names(self):
        self.assertEqual(symbol_name('เ◌อ'), 'สระเออ')
        self.assertEqual(symbol_name('◌ัว'), 'สระอัว')

    def test_unknown_symbol_returns_text_itself(self):
        self.assertEqual(symbol_name('?'), '?')


if __name__ == '__main__':
    unittest.main()
