"""Segment source patterns before changing open/closed Thai vowel spelling."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from decoder import decode_cells, decode_cells_verbose
from thai_parser import compose_tokens
from thai_decoder import decode_thai_details
from thai_parser import parse_braille
from thai_renderer import render_thai


WORDS = {
    'กะ': ['1245', '1'], 'กัน': ['1245', '1', '1345'],
    'จะ': ['245', '1'], 'จม': ['245', '134'],
    'เกะ': ['1245', '124', '1'], 'เต็ม': ['1256', '124', '1', '134'],
    'แกะ': ['1245', '126', '1'], 'แข็ง': ['13', '126', '1', '12456'],
    'โกะ': ['1245', '24', '1'], 'คน': ['136', '24', '1', '1345'],
    'เจอ': ['245', '146'], 'เกิด': ['1245', '146', '145'],
    'เกาะ': ['1245', '135', '1'], 'เลอะ': ['123', '146', '1'],
    'ตัว': ['1256', '15'], 'สวน': ['234', '2456', '1345'],
    'ขอน': ['13', '135', '1345'], 'น้ำ': ['1345', '256', '1356'],
    'จน': ['245', '1345'], 'กบ': ['1245', '1236'],
    'กลม': ['1245', '123', '134'], 'ส่ง': ['234', '35', '12456'],
    'สะ': ['234', '1'], 'กด': ['1245', '145'],
    'ตก': ['1256', '1245'], 'ใจ': ['156', '2', '245'],
    'เพราะ': ['1456', '1235', '135', '1'],
}


def physical(patterns):
    return [dict(dots=list(map(int, p)), line_id=0, reading_x=i*95,
                 cell_pitch=95, cell_id=f'S{i}', confidence=.99)
            for i, p in enumerate(patterns)]


def sentence(words):
    return [p for word in words for p in WORDS[word]]


class ThaiSyllableBoundaryTests(unittest.TestCase):
    def test_all_requested_open_closed_vowel_examples(self):
        for word, patterns in WORDS.items():
            with self.subTest(word=word):
                result = decode_thai_details(physical(patterns))
                self.assertEqual(result['text'], word)
                self.assertFalse(result['needs_review'])

    def test_next_implicit_onset_is_not_stolen_as_a_final(self):
        phrases = [
            ['จะ', 'จม'], ['ขอน', 'จะ', 'จม', 'น้ำ'],
            ['กะ', 'คน'], ['เกะ', 'จม'], ['แกะ', 'คน'], ['โกะ', 'จน'],
            ['เกาะ', 'คน'], ['เลอะ', 'จม'], ['จะ', 'กลม'], ['จะ', 'ส่ง', 'น้ำ'],
            ['สะ', 'กด'], ['แกะ', 'ตก', 'ใจ'], ['เพราะ', 'คน'],
        ]
        for words in phrases:
            with self.subTest(words=words):
                result = decode_thai_details(physical(sentence(words)))
                self.assertEqual(result['text'], ''.join(words))
                self.assertFalse(result['needs_review'])
        result = decode_thai_details(physical(sentence(['ขอน', 'จะ', 'จม', 'น้ำ'])))
        self.assertEqual([s['onset'] for s in result['syllables']], ['ข', 'จ', 'จ', 'น'])
        self.assertEqual([s['final'] for s in result['syllables']], ['น', None, 'ม', None])
        self.assertEqual([s['indices'] for s in result['syllables']],
                         [[0, 1, 2], [3, 4], [5, 6], [7, 8, 9]])
        self.assertEqual(result['syllables'][1]['boundary_rule'], 'NEXT_IMPLICIT_ONSET')

    def test_closed_vowels_keep_their_final_before_an_implicit_syllable(self):
        for words in (['กัน', 'จม'], ['เต็ม', 'จน'], ['แข็ง', 'จม'],
                      ['คน', 'จน'], ['เกิด', 'จม'], ['สวน', 'คน']):
            with self.subTest(words=words):
                result = decode_thai_details(physical(sentence(words)))
                self.assertEqual(result['text'], ''.join(words))
                self.assertFalse(result['needs_review'])
                self.assertIsNotNone(result['syllables'][0]['final'])

    def test_multiple_implicit_syllables_and_tones(self):
        result = decode_thai_details(physical(sentence(['จะ', 'จม', 'จน', 'ส่ง', 'น้ำ'])))
        self.assertEqual(result['text'], 'จะจมจนส่งน้ำ')
        self.assertEqual(len(result['syllables']), 5)
        self.assertFalse(result['needs_review'])
        sent = result['syllables'][3]
        self.assertEqual((sent['onset'], sent['vowel'], sent['tone'], sent['final']),
                         ('ส', 'โอะ', '่', 'ง'))

    def test_parser_returns_segmentation_before_rendering_and_retains_raw_sources(self):
        source = physical(sentence(['จะ', 'จม']))
        original = deepcopy(source)
        with patch('thai_renderer.compose_syllable', side_effect=AssertionError('early render')):
            parsed = parse_braille(source)
        self.assertNotIn('text', parsed)
        self.assertEqual(source, original)
        self.assertEqual(parsed['raw_patterns'], sentence(['จะ', 'จม']))
        inferred = next(t for t in parsed['semantic_tokens'] if t['rule'] == 'IMPLICIT_O_CLOSED')
        self.assertEqual(inferred['value'], 'โอะ')
        self.assertEqual(inferred['patterns'], [])  # no invented detected vowel cell
        self.assertEqual(render_thai(parsed)['text'], 'จะจม')

    def test_unresolved_boundary_returns_review_instead_of_changed_spelling(self):
        result = decode_thai_details(physical(['1245', '146', '145', '1345']))
        self.assertTrue(result['needs_review'])
        self.assertEqual(result['syllables'][0]['status'], 'AMBIGUOUS_SYLLABLE')
        self.assertEqual(result['syllables'][0]['boundary_rule'], 'AMBIGUOUS_BOUNDARY')
        self.assertIn('�', result['text'])

    def test_implicit_vowel_never_joins_across_line_space_punctuation_or_error(self):
        for boundary in ('line', 'space', 'gap', 'punctuation', 'error'):
            with self.subTest(boundary=boundary):
                source = physical(WORDS['จม'])
                if boundary == 'line':
                    source[1]['line_id'] = 1
                elif boundary == 'space':
                    source.insert(1, dict(dots=[]))
                elif boundary == 'gap':
                    source[1]['reading_x'] = 500
                elif boundary == 'error':
                    source[1]['geometry_status'] = 'invalid'
                else:
                    source = physical(['245', '36', '134'])
                parsed = parse_braille(source)
                self.assertFalse(any(t['rule'] == 'IMPLICIT_O_CLOSED'
                                     for t in parsed['semantic_tokens']))

if __name__ == '__main__':
    unittest.main()
