"""Acceptance criteria and source traces from THAI_BRAILLE_G1_PARSER_RULES.md."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from colored_braille import ColoredCellStream
from decoder import decode_cells, decode_cells_verbose
from thai_braille import PATTERN_CANDIDATES
from thai_decoder import decode_thai_details
from thai_parser import parse_braille
from thai_renderer import render_syllable, render_thai
from tests.test_yolo_cell_stream import page
from tools.diagnostics.export_yolo_stream import export_stream


ACCEPTANCE = [
    (['1245', '1'], 'กะ'), (['1245', '124', '1'], 'เกะ'),
    (['1245', '126', '1'], 'แกะ'), (['1245', '24', '1'], 'โกะ'),
    (['1245', '135', '1'], 'เกาะ'), (['123', '146', '1'], 'เลอะ'),
    (['245', '146'], 'เจอ'), (['1245', '146', '145'], 'เกิด'),
    (['134', '12356'], 'เมีย'), (['234', '12345'], 'เสือ'),
    (['1256', '15'], 'ตัว'), (['13', '235'], 'เขา'),
    (['13', '135'], 'ขอ'), (['136', '1356'], 'คำ'),
    (['156', '12346'], 'ไป'), (['156', '2', '245'], 'ใจ'),
    (['135', '1'], 'อะ'), (['1256', '124', '1'], 'เตะ'),
    (['124', '1256', '3', '134'], 'เต็ม'),
    (['136', '26', '135'], 'คือ'), (['136', '26', '135', '1345'], 'คืน'),
]


def cells(patterns):
    return [dict(dots=frozenset(map(int, p)), cell_id=f'book-{i}', line_id=0,
                 reading_x=95*i, cell_pitch=95, adjacency_limit=145)
            for i, p in enumerate(patterns)]


class ThaiGradeOneParserTests(unittest.TestCase):
    def test_document_acceptance_patterns(self):
        for patterns, expected in ACCEPTANCE:
            with self.subTest(patterns=patterns):
                source = cells(patterns)
                original = deepcopy(source)
                result = decode_thai_details(source)
                self.assertEqual(result['text'], expected)
                self.assertEqual(result['raw_patterns'], patterns)
                self.assertEqual(result['status'], 'VALID')
                self.assertFalse(result['needs_review'])
                self.assertEqual(source, original)
                json.dumps(result, ensure_ascii=False)

    def test_all_contextual_short_vowels_retain_both_source_cells(self):
        for body, canonical, expected in [('124', 'เอะ', 'เกะ'), ('126', 'แอะ', 'แกะ'),
                                         ('24', 'โอะ', 'โกะ'), ('135', 'เอาะ', 'เกาะ'),
                                         ('146', 'เออะ', 'เกอะ'), ('12356', 'เอียะ', 'เกียะ'),
                                         ('12345', 'เอือะ', 'เกือะ'), ('15', 'อัวะ', 'กัวะ')]:
            with self.subTest(body=body):
                result = decode_thai_details(cells(['1245', body, '1']))
                self.assertEqual(result['text'], expected)
                vowel = next(t for t in result['semantic_tokens'] if t['type'] == 'VOWEL')
                self.assertEqual(vowel['value'], canonical)
                self.assertEqual(vowel['patterns'], [body, '1'])
                self.assertEqual(vowel['source_cells'], ['book-1', 'book-2'])
                self.assertEqual(vowel['rule'], 'POST_ONSET_SEQUENCE')

    def test_prefix_consonants_and_preposed_vowels_win_longest_match(self):
        cases = [('356', '13', 'ฃ'), ('36', '136', 'ฅ'), ('6', '136', 'ฆ'),
                 ('6', '346', 'ฌ'), ('6', '13456', 'ญ'), ('6', '145', 'ฎ'),
                 ('6', '1256', 'ฏ'), ('6', '2345', 'ฐ'), ('6', '23456', 'ฑ'),
                 ('36', '23456', 'ฒ'), ('6', '1345', 'ณ'), ('356', '23456', 'ธ'),
                 ('6', '1456', 'ภ'), ('6', '234', 'ศ'), ('36', '234', 'ษ'),
                 ('6', '123', 'ฬ')]
        for a, b, text in cases:
            with self.subTest(text=text):
                result = decode_thai_details(cells([a, b]))
                self.assertEqual(result['text'], text)
                self.assertTrue(result['raw_tokens'][1]['consumed'])
                self.assertEqual(result['semantic_tokens'][0]['patterns'], [a, b])
                self.assertEqual(result['syllables'][0]['indices'], [0, 1])
        self.assertEqual(decode_thai_details(cells(['36']))['text'], '-')
        self.assertEqual(decode_thai_details(cells(['156', '2', '245']))['text'], 'ใจ')

    def test_135_candidates_are_selected_by_syllable_context(self):
        self.assertEqual({c['value'] for c in PATTERN_CANDIDATES['135']}, {'อ', 'ออ', 'เอาะ'})
        for patterns, role, value in [(['135', '1'], 'ONSET', 'อ'),
                                      (['13', '135'], 'VOWEL', 'ออ'),
                                      (['1245', '135', '1'], 'VOWEL', 'เอาะ')]:
            parsed = parse_braille(cells(patterns))
            chosen = next(t for t in parsed['semantic_tokens'] if t['type'] == role)
            self.assertEqual(chosen['value'], value)
        parsed = parse_braille(['1245', '135', '1'])
        match = next(t for t in parsed['parser_trace'] if t['rule'] == 'POST_ONSET_SEQUENCE')
        self.assertEqual((match['state_before'], match['state_after']), ('ONSET', 'POST_VOWEL'))
        self.assertEqual(match['patterns'], ['135', '1'])

    def test_parser_does_not_reparse_display_labels_or_call_renderer(self):
        with patch('thai_parser._atom', side_effect=AssertionError('display label parsed')):
            with patch('thai_renderer.compose_syllable', side_effect=AssertionError('renderer called')):
                parsed = parse_braille(['1245', '135', '1'])
        self.assertNotIn('text', parsed)
        self.assertEqual(parsed['syllables'][0]['vowel'], 'เอาะ')
        parsed['raw_tokens'][1]['char'] = 'display label only'
        self.assertEqual(render_thai(parsed)['text'], 'เกาะ')
        self.assertEqual(render_syllable(dict(onset='ก', vowel='เอาะ')), 'เกาะ')

    def test_word_line_and_failed_cell_boundaries_stop_sequences(self):
        for patterns in (['1245', '135', '1'], ['1245', '124', '1'], ['6', '145']):
            for boundary in ('line', 'gap', 'error'):
                with self.subTest(patterns=patterns, boundary=boundary):
                    source = cells(patterns)
                    if boundary == 'line':
                        source[-1]['line_id'] = 1
                    elif boundary == 'gap':
                        source[-1]['reading_x'] += 500
                    else:
                        source[-1]['geometry_status'] = 'invalid'
                    result = decode_thai_details(source)
                    self.assertFalse(any(len(c['patterns']) > 1 for c in result['semantic_tokens']))
        result = decode_thai_details(cells(['1245', '', '135', '1']))
        self.assertEqual(result['text'], 'ก อะ')

    def test_ambiguous_mark_and_invalid_patterns_need_review(self):
        for patterns in (['3'], ['1245', '3', '1345']):
            result = decode_thai_details(cells(patterns))
            self.assertTrue(result['needs_review'])
            self.assertEqual(result['status'], 'NEEDS_REVIEW')
            self.assertIn('AMBIGUOUS_SEQUENCE', [s['status'] for s in result['syllables']])
            self.assertIn('�', result['text'])
        for source in (['7'], [{}], [dict(dots=[1, 1])]):
            result = render_thai(parse_braille(source))
            self.assertEqual(result['text'], '�')
            self.assertTrue(result['needs_review'])

    def test_missing_pattern_inside_a_run_is_reported_without_crashing(self):
        source = cells(['1245']) + [{}] + cells(['135', '1'])
        result = decode_thai_details(source)
        self.assertTrue(result['needs_review'])
        self.assertEqual(result['raw_cells'][1]['warning'], 'invalid_raw_pattern')
        self.assertIn('�', result['text'])

    def test_contextual_matching_obeys_legacy_coordinate_adjacency(self):
        for body in ('124', '146', '135'):
            with self.subTest(body=body):
                source = [dict(dots=list(map(int, p)), x=x, y=0,
                               grid=dict(expected_cols=[0, 20]))
                          for p, x in zip(('1245', body, '1'), (0, 50, 130))]
                result = decode_thai_details(source)
                self.assertFalse(result['raw_cells'][2]['adjacent_before'])
                self.assertFalse(any(len(c['patterns']) > 1 for c in result['semantic_tokens']))
                self.assertIn(' ', result['text'])

    def test_explicit_open_written_mark_is_not_lost_to_previous_coda(self):
        for patterns, expected in [(['1245', '3'], 'ก็'),
                                   (['1245', '3', '24', '1256'], 'ก็โต'),
                                   (['134', '23', '1245', '3', '24', '1256'], 'มีก็โต')]:
            with self.subTest(patterns=patterns):
                result = decode_thai_details(cells(patterns))
                self.assertEqual(result['text'], expected)
                self.assertFalse(result['needs_review'])
                mark = next(s for s in result['syllables'] if s.get('vowel') == '็')
                self.assertEqual(mark['render_events'][0]['reason'], 'EXPLICIT_MAITAIKHU')
                self.assertIsNone(mark['render_events'][0]['inserted_mark'])

    def test_mai_taikhu_has_explicit_or_closed_vowel_evidence(self):
        for patterns, reason, inserted in [(['124', '1256', '3', '134'], 'EXPLICIT_MAITAIKHU', None),
                                            (['1256', '124', '1', '134'], 'VOWEL_E_SHORT_CLOSED', '็')]:
            result = decode_thai_details(cells(patterns))
            self.assertEqual(result['text'], 'เต็ม')
            event = result['syllables'][0]['render_events'][0]
            self.assertEqual(event['reason'], reason)
            self.assertEqual(event['inserted_mark'], inserted)
            self.assertEqual(event['source'], patterns)
        result = decode_thai_details(cells(['1245', '126', '1', '13', '135', '1345']))
        self.assertEqual(result['text'], 'แกะขอน')
        self.assertFalse(any(s['render_events'] for s in result['syllables']))

    def test_source_confidence_includes_consumed_cells_and_lexicon_only_validates(self):
        source = cells(['6', '145', '16'])
        for cell, confidence in zip(source, (.99, .4, .97)):
            cell['confidence'] = confidence
        result = decode_thai_details(source, confidence_threshold=.8)
        self.assertEqual(result['syllables'][0]['min_confidence'], .4)
        self.assertEqual(result['syllables'][0]['status'], 'LOW_CONFIDENCE')
        self.assertEqual(result['syllables'][0]['candidate'], 'ฎา')
        source = cells(['12456', '146', '1', '1345'])
        self.assertTrue(decode_thai_details(source)['needs_review'])
        self.assertEqual(decode_thai_details(source, lexicon={'เงิน'})['text'], 'เงิน')
        self.assertEqual(decode_thai_details(source, lexicon={'กิน'})['text'], '�')

    def test_punctuation_confidence_includes_its_continuation_cell(self):
        source = cells(['456', '236'])
        source[0]['confidence'], source[1]['confidence'] = .99, .1
        parsed = parse_braille(source)
        self.assertEqual(parsed['syllables'][0]['source_cells'], ['book-0', 'book-1'])
        result = render_thai(parsed, confidence_threshold=.8)
        self.assertEqual(result['text'], '�')
        self.assertEqual(result['syllables'][0]['candidate'], '?')
        self.assertEqual(result['syllables'][0]['min_confidence'], .1)
        self.assertEqual(result['syllables'][0]['status'], 'LOW_CONFIDENCE')

    def test_book_regression_and_raw_overlay_labels(self):
        fixture = Path(__file__).parent/'fixtures/thai_g1_book_cases.json'
        for case in json.loads(fixture.read_text(encoding='utf-8')):
            source = cells(case['raw_patterns'])
            for cell, identity in zip(source, case['cells']):
                cell['cell_id'] = identity
            result = decode_thai_details(source)
            self.assertEqual(result['text'], case['expected'])
            self.assertEqual([t['char'] for t in result['raw_tokens']][-6:], ['ก', 'อ', 'ะ', 'ข', 'อ', 'น'])
            vowel = next(t for t in result['semantic_tokens'] if t['value'] == 'เอาะ')
            self.assertEqual(vowel['source_cells'], ['C37', 'C38'])
            self.assertEqual([t['char'] for t in decode_cells_verbose(source, 'thai')],
                             [t['char'] for t in result['raw_tokens']])

    def test_diagnostic_exports_parser_and_renderer_trace(self):
        image, _, _ = page([['1245', '135', '1']])
        detected, debug = ColoredCellStream().detect(image)
        with tempfile.TemporaryDirectory() as folder:
            report = json.loads(export_stream(image, detected, debug, folder).read_text(encoding='utf-8'))
        trace = report['language_trace']
        self.assertEqual(trace['text'], 'เกาะ')
        self.assertEqual(trace['raw_patterns'], ['1245', '135', '1'])
        self.assertTrue(trace['parser_trace'])
        self.assertTrue(trace['semantic_tokens'])

    def test_english_stays_on_its_own_decoder(self):
        self.assertEqual(decode_cells(cells(['6', '145', '3456', '1', '12']), 'english'), 'D12')

    def test_diagnostic_standard_boundary_does_not_invalidate_known_cells(self):
        image, _, _ = page([['125', '15', '1235', '135', '1']])
        detected, debug = ColoredCellStream().detect(image)
        with tempfile.TemporaryDirectory() as folder:
            report = json.loads(export_stream(image, detected, debug, folder).read_text(encoding='utf-8'))
        self.assertEqual(report['text'], 'หัวเราะ')
        self.assertEqual(report['language_trace']['status'], 'VALID')
        self.assertEqual(report['warning_counts'], {})
        tokens = decode_cells_verbose(detected, 'thai')
        self.assertEqual((tokens[0]['char'], tokens[2]['char']), ('ห', 'ร'))
        self.assertFalse(any(t.get('warning') for t in tokens))
        self.assertFalse(any(t.get('composition_warning') for t in tokens))


if __name__ == '__main__':
    unittest.main()
