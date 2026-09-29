"""Standalone dot-36 punctuation must not shadow complete Thai consonants."""

from copy import deepcopy
import unittest

from decoder import decode_cells, decode_cells_verbose
from tests.test_punctuation_recovery import logical_cells
from tests.test_thai_standard import logical_cells as image_cells


PAIRS = [('136', 'ฅ', 'ค'), ('23456', 'ฒ', 'ท'), ('234', 'ษ', 'ส')]


class ThaiHyphenTests(unittest.TestCase):
    def test_standalone_and_between_text_without_warning(self):
        for patterns, expected in [(['36'], '-'), (['36', '1245'], '-ก'),
                                   (['1245', '36'], 'ก-'),
                                   (['1245', '36', '13'], 'ก-ข')]:
            for language in ('thai', 'th'):
                with self.subTest(patterns=patterns, language=language):
                    cells = logical_cells(patterns)
                    original = deepcopy(cells)
                    self.assertEqual(decode_cells(cells, language), expected)
                    tokens = decode_cells_verbose(cells, language)
                    self.assertFalse(any(t['warning'] for t in tokens))
                    self.assertFalse(any(t['consumed'] for t in tokens))
                    self.assertEqual(cells, original)

    def test_adjacent_consonants_take_precedence_at_start_and_inside_text(self):
        for tail, consonant, _ in PAIRS:
            for prefix in ([], ['1245']):
                with self.subTest(consonant=consonant, prefix=prefix):
                    cells = logical_cells(prefix + ['36', tail])
                    self.assertEqual(decode_cells(cells, 'thai'), ('ก' if prefix else '')+consonant)
                    tokens = decode_cells_verbose(cells, 'thai')
                    self.assertEqual(tokens[-2]['char'], consonant)
                    self.assertEqual(tokens[-1]['char'], '')
                    self.assertTrue(tokens[-1]['consumed'])
                    self.assertFalse(any(t['warning'] for t in tokens))

    def test_pairs_never_cross_blank_gap_or_line(self):
        for tail, _, single in PAIRS:
            for boundary in ('blank', 'gap', 'line', 'image_gap', 'image_line'):
                with self.subTest(tail=tail, boundary=boundary):
                    cells = logical_cells(['36', tail])
                    separator = ' '
                    if boundary == 'blank':
                        cells = logical_cells(['36', '', tail])
                    elif boundary == 'gap':
                        cells[1]['reading_x'] = 190
                    elif boundary == 'line':
                        cells[1]['line_id'] = 1
                        cells[1]['reading_x'] = 0
                        separator = '\n'
                    else:
                        cells = image_cells(['36', tail])
                        cells[1].update(dict(x=1000) if boundary == 'image_gap' else dict(y=300))
                    self.assertEqual(decode_cells(cells, 'thai'), '-'+separator+single)
                    tokens = decode_cells_verbose(cells, 'thai')
                    self.assertEqual(tokens[0]['char'], '-')
                    self.assertFalse(tokens[-1]['consumed'])
                    self.assertFalse(any(t['warning'] for t in tokens))

    def test_hyphen_does_not_hide_unreadable_following_cell(self):
        for changes in (dict(row_ambiguous=True), dict(dots=frozenset(), crop_status='empty')):
            cells = logical_cells(['36', '136'])
            cells[1].update(changes)
            tokens = decode_cells_verbose(cells, 'thai')
            self.assertEqual(tokens[0]['char'], '-')
            self.assertIsNone(tokens[0]['warning'])
            self.assertTrue(tokens[1]['warning'])
            self.assertFalse(tokens[1]['consumed'])
            self.assertEqual(decode_cells(cells, 'thai'), '-�')

    def test_english_mapping_is_unchanged(self):
        self.assertEqual(decode_cells(logical_cells(['36']), 'english'), '[3,6]')
        self.assertEqual(decode_cells(logical_cells(['145', '1']), 'english'), 'da')


if __name__ == '__main__':
    unittest.main()
