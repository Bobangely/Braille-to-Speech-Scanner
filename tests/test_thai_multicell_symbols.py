"""Config-driven punctuation precedes singles without changing Thai spelling."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from config_thai import THAI_MULTI_CELL_SYMBOLS, THAI_READING_MULTI_CELL
from decoder import decode_cells
from thai_braille import LEXICAL_SEQUENCES, lex_braille
from thai_decoder import decode_thai_details, tokenize_thai


def cells(patterns):
    return [dict(dots=list(pattern), line_id=0, reading_x=index*95, cell_pitch=95)
            for index, pattern in enumerate(patterns)]


class ThaiMultiCellSymbolTests(unittest.TestCase):
    def test_open_parenthesis_uses_config_and_retains_both_cells(self):
        source = cells([[5], [1, 2, 6]])
        before = deepcopy(source)
        result = decode_thai_details(source)
        self.assertEqual(result['text'], '(')
        self.assertFalse(result['needs_review'])
        self.assertEqual(source, before)
        self.assertEqual(result['raw_patterns'], ['5', '126'])
        token, continuation = result['raw_tokens']
        self.assertEqual(token['char'], '(')
        self.assertEqual(token['symbol_id'], 'open_parenthesis')
        self.assertEqual(token['tts'], 'วงเล็บเปิด')
        self.assertEqual(token['symbol_type'], 'punctuation')
        self.assertEqual((token['cell_start'], token['cell_end']), (0, 1))
        self.assertTrue(continuation['consumed'])
        self.assertEqual(continuation['char'], '')
        rule = THAI_MULTI_CELL_SYMBOLS[(frozenset({5}), frozenset({1, 2, 6}))]
        self.assertEqual(THAI_READING_MULTI_CELL[(frozenset({5}), frozenset({1, 2, 6}))], rule['output'])

    def test_standalone_ae_and_other_single_symbols_are_preserved(self):
        self.assertEqual(decode_cells(cells([[1, 2, 6]]), 'thai'), 'แ')
        self.assertEqual(tokenize_thai(cells([[5]]))[0]['char'], 'ํ')
        self.assertEqual(decode_cells(cells([[1, 2, 6], [1, 3, 4], [3, 5]]), 'thai'), 'แม่')

    def test_dot_order_is_normalized_before_matching(self):
        for body in ([1, 2, 6], [6, 2, 1], [2, 1, 6]):
            with self.subTest(body=body):
                self.assertEqual(decode_cells(cells([[5], body]), 'thai'), '(')

    def test_symbol_does_not_consume_across_any_boundary(self):
        for boundary in ('line', 'blank', 'gap', 'break_before', 'line_break_before', 'invalid'):
            with self.subTest(boundary=boundary):
                source = cells([[5], [1, 2, 6]])
                if boundary == 'line':
                    source[1]['line_id'] = 1
                elif boundary == 'blank':
                    source.insert(1, dict(dots=[]))
                elif boundary == 'gap':
                    source[1]['reading_x'] = 400
                elif boundary == 'invalid':
                    source[1]['geometry_status'] = 'invalid'
                else:
                    source[1][boundary] = True
                result = decode_thai_details(source)
                self.assertNotIn('(', result['text'])
                self.assertTrue(all(not token['consumed'] for token in result['raw_tokens']))
                if boundary != 'invalid':
                    self.assertEqual(result['raw_tokens'][-1]['char'], 'แ')

    def test_longest_lexical_match_precedes_shorter_symbol(self):
        # Temporary rule exercises the existing generic matcher with 3 cells.
        longer = dict(kind='literal', value='LONG', vowel=None, semantic=False)
        with patch.dict(LEXICAL_SEQUENCES, {('5', '126', '1'): longer}):
            self.assertEqual(decode_cells(cells([[5], [1, 2, 6], [1]]), 'thai'), 'LONG')

    def test_unknown_sequence_logs_evidence_and_falls_back_to_singles(self):
        source = cells([[5], [2, 4, 6]])
        source[0]['confidence'] = .91
        with self.assertLogs('thai_braille', level='DEBUG') as logs:
            _, tokens, _ = lex_braille(source)
        self.assertEqual([t['char'] for t in tokens], ['ํ', 'ึ'])
        message = '\n'.join(logs.output)
        for field in ('UNKNOWN_BRAILLE_SEQUENCE', 'line=0', 'cell_index=0',
                      'raw_dots=', 'normalized_dots=', 'neighbor_cells=', 'confidence=0.91'):
            self.assertIn(field, message)
        with self.assertLogs('thai_braille', level='DEBUG') as unknown:
            result = decode_thai_details(cells([[4, 5, 6]]))
        self.assertTrue(result['needs_review'])
        self.assertIn('UNKNOWN_BRAILLE_SEQUENCE', unknown.output[0])

if __name__ == '__main__':
    unittest.main()
