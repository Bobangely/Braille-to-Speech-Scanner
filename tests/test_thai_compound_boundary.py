"""Preserve the next onset before reducing the previous Thai vowel."""
from copy import deepcopy
import unittest

from thai_composer import compose_tokens
from thai_decoder import decode_thai_details
from thai_parser import parse_braille
from thai_renderer import render_thai, render_syllable
from tests.test_thai_syllable_boundaries import physical


class ThaiCompoundBoundaryTests(unittest.TestCase):
    def test_book_raw_cells_keep_the_next_onset_and_compound_vowel(self):
        # diag_038 / screenshot C61-C65 (also C71-C75), not inferred from text.
        patterns = ['125', '15', '1235', '135', '1']
        cells = physical(patterns)
        for index, cell in enumerate(cells, 61):
            cell['cell_id'] = f'C{index}'
        original = deepcopy(cells)
        parsed = parse_braille(cells)
        self.assertEqual(cells, original)
        self.assertEqual(parsed['raw_patterns'], patterns)
        self.assertEqual([t['char'] for t in parsed['raw_tokens']], ['ห', '◌ัว', 'ร', 'อ', 'ะ'])
        self.assertEqual([(s['onset'], s['vowel'], s['final']) for s in parsed['syllables']],
                         [('ห', 'อัว', None), ('ร', 'เอาะ', None)])
        self.assertEqual(parsed['syllables'][0]['boundary_rule'], 'NEXT_EXPLICIT_ONSET')
        self.assertEqual(parsed['syllables'][1]['source_cells'], ['C63', 'C64', 'C65'])
        vowel = next(t for t in parsed['semantic_tokens'] if t['value'] == 'เอาะ')
        self.assertEqual(vowel['patterns'], ['135', '1'])
        self.assertEqual(render_thai(parsed)['text'], 'หัวเราะ')

    def test_compound_vowels_after_reducible_vowels_are_general_not_word_rules(self):
        cases = [
            (['125', '15', '1235', '135', '1'], 'หัวเราะ'),
            (['125', '15', '1235', '135', '1', '13456', '135', '1'], 'หัวเราะเยาะ'),
            (['23456', '1', '123', '135', '1'], 'ทะเลาะ'),
            (['1256', '15', '1245', '135', '1'], 'ตัวเกาะ'),
            (['1245', '1', '1245', '135', '1'], 'กะเกาะ'),
            (['1245', '124', '1', '1235', '135', '1'], 'เกะเราะ'),
            (['125', '15', '123', '146', '1'], 'หัวเลอะ'),
            (['125', '15', '245', '146'], 'หัวเจอ'),
        ]
        for patterns, expected in cases:
            with self.subTest(expected=expected):
                result = decode_thai_details(physical(patterns))
                if expected in ('ทะเลาะ', 'กะเกาะ', 'เกะเราะ'):
                    # Short vowel + coda and next compound onset are both
                    # structurally possible: preserve the former reading as
                    # an alternative, not as a falsely certain final result.
                    alternatives = result['syllables'][0]['boundary_candidates']
                    rendered = {''.join(render_syllable(s)
                                        for s in option['syllables']) for option in alternatives}
                    self.assertIn(expected, rendered)
                else:
                    self.assertEqual(result['text'], expected)
                if expected in ('หัวเลอะ', 'หัวเจอ', 'หัวเราะ', 'ตัวเกาะ'):
                    self.assertFalse(result['needs_review'])
                else:
                    self.assertTrue(result['needs_review'])
                    self.assertEqual(result['status'], 'NEEDS_REVIEW')

    def test_real_codas_and_independent_o_are_not_globally_replaced(self):
        cases = [
            # raw มาตรฐานเขียน ว หลังสระอัวลดรูป ไม่ใช่ ัว (15)
            (['234', '2456', '1345'], 'สวน'),
            (['13', '2456', '145'], 'ขวด'),
            (['1245', '12', '1345', '135', '1'], 'กินอะ'),
            (['135', '1'], 'อะ'),
            (['13', '135', '1345'], 'ขอน'),
            # Literal า + ะ is not 135-1: do not invent a leading เ.
            (['1235', '16', '1'], 'ราะ'),
            (['1235', '135', '1'], 'เราะ'),
        ]
        for patterns, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(decode_thai_details(physical(patterns))['text'], expected)

    def test_text_adapter_uses_the_same_segmentation_rule(self):
        result = compose_tokens(['ห', '◌ัว', 'ร', 'อ', 'ะ'])
        self.assertEqual(result['text'], 'หัวเราะ')

    def test_compound_vowel_cannot_cross_space_or_line(self):
        for boundary in ('space', 'line'):
            cells = physical(['125', '15', '1235', '135', '1'])
            if boundary == 'space':
                cells[4]['reading_x'] += 400
            else:
                cells[4]['line_id'] = 1
            parsed = parse_braille(cells)
            self.assertFalse(any(t['value'] == 'เอาะ' for t in parsed['semantic_tokens']))

    def test_review_debug_includes_raw_decoded_and_composed_data(self):
        cells = physical(['1235', '135', '1'])
        with self.assertLogs('thai_decoder', level='DEBUG') as captured:
            result = decode_thai_details(cells, confidence_threshold=1.)
        self.assertTrue(result['needs_review'])
        log = '\n'.join(captured.output)
        for field in ('raw_cells=', 'decoded_cells=', 'syllables=', 'composed_output='):
            self.assertIn(field, log)


if __name__ == '__main__':
    unittest.main()
