"""ทดสอบ raw encoding มาตรฐานแยกจากสระ semantic แบบเดิม"""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from thai_composer import compose_tokens
from thai_decoder import decode_thai_details
from tests.test_thai_syllable_boundaries import physical


class ThaiRawEncodingTests(unittest.TestCase):
    def test_standard_ua_keeps_the_written_vowel_before_the_next_onset(self):
        result = decode_thai_details(physical(['125', '15', '1235', '135', '1']),
                                     lexicon=set())
        self.assertEqual(result['text'], 'หัวเราะ')
        self.assertEqual(result['status'], 'VALID')
        self.assertEqual([(s['onset'], s['vowel'], s['final']) for s in result['syllables']],
                         [('ห', 'อัว', None), ('ร', 'เอาะ', None)])

    def test_rejected_semantic_coda_keeps_its_raw_provenance_and_reason(self):
        source = physical(['125', '15', '1235', '135', '1'])
        original = deepcopy(source)
        result = decode_thai_details(source)
        paths = result['segmentation_runs'][0]['candidates']
        self.assertEqual(len(paths), 2)
        closed = next(p for p in paths if p['syllables'][0]['final'] == 'ร')
        self.assertFalse(closed['valid'])
        self.assertFalse(closed['standard_valid'])
        self.assertTrue(closed['compatibility_valid'])
        self.assertEqual(closed['standard_error'], 'NONSTANDARD_UA_CLOSED_FORM')
        self.assertEqual(closed['validation_error'], 'NONSTANDARD_UA_CLOSED_FORM')
        self.assertEqual(closed['syllables'][0]['source_cells'], ['S0', 'S1', 'S2'])
        self.assertEqual(source, original)
        self.assertEqual(result['raw_patterns'], ['125', '15', '1235', '135', '1'])

    def test_lexicon_cannot_restore_a_rejected_standard_raw_form(self):
        result = decode_thai_details(physical(['125', '15', '1235', '135', '1']),
                                     lexicon={'หวร', 'อะ'})
        self.assertEqual((result['text'], result['status']), ('หัวเราะ', 'VALID'))
        self.assertEqual([c['text'] for c in result['lexicon_ranking'][0]['candidates']],
                         ['หัวเราะ'])

    def test_explicit_compatibility_conflict_keeps_standard_display_and_review(self):
        result = decode_thai_details(physical(['125', '15', '1235', '135', '1']),
                                     compatibility=True, lexicon={'หวร', 'อะ'})
        self.assertEqual((result['text'], result['status']), ('หัวเราะ', 'NEEDS_REVIEW'))
        paths = result['lexicon_ranking'][0]['candidates']
        self.assertEqual({p['text'] for p in paths}, {'หัวเราะ', 'หวรอะ'})
        self.assertEqual(paths[0]['accepted_mode'], 'standard')
        self.assertEqual(paths[1]['accepted_mode'], 'compatibility')
        self.assertTrue(all(p['plausible'] for p in paths))

    def test_standard_closed_ua_uses_written_wo_without_an_onset_whitelist(self):
        cases = [(['234', '2456', '1345'], 'สวน'),
                 (['136', '2456', '1235'], 'ควร'),
                 (['13', '2456', '145'], 'ขวด')]
        for patterns, text in cases:
            with self.subTest(text=text):
                result = decode_thai_details(physical(patterns), lexicon=set())
                self.assertEqual((result['text'], result['status']), (text, 'VALID'))

    def test_legacy_closed_ua_requires_explicit_compatibility(self):
        for patterns, text in [(['234', '15', '1345'], 'สวน'),
                               (['136', '15', '1235'], 'ควร')]:
            with self.subTest(text=text):
                source = physical(patterns)
                standard = decode_thai_details(source)
                self.assertEqual(standard['status'], 'NEEDS_REVIEW')
                self.assertEqual(standard['syllables'][0]['status'], 'NONSTANDARD_UA_CLOSED_FORM')
                legacy = decode_thai_details(source, compatibility=True, lexicon=set())
                self.assertEqual((legacy['text'], legacy['status']), (text, 'VALID'))
                self.assertEqual(legacy['syllables'][0]['accepted_mode'], 'compatibility')

    def test_semantic_adapter_still_accepts_legacy_closed_ua(self):
        result = compose_tokens(['ส', 'อัว', 'น'], lexicon=set())
        self.assertEqual((result['text'], result['status']), ('สวน', 'VALID'))

    def test_rule_applies_to_other_onsets_and_multisyllable_runs(self):
        for patterns, text, status in [(['1256', '15', '1245', '135', '1'], 'ตัวเกาะ', 'VALID'),
                               (['125', '15', '1235', '135', '1', '13456', '135', '1'],
                                'หัวเราะเยาะ', 'NEEDS_REVIEW')]:
            result = decode_thai_details(physical(patterns), lexicon=set())
            # boundary ของ เราะ/เยาะ ยังมีสองทางตามกฎเดิม ห้ามปลด review ทั้ง run
            self.assertEqual((result['text'], result['status']), (text, status))

    def test_previous_p1_ambiguities_remain_unconfirmed(self):
        for patterns, text in [(['1245', '1', '1345', '135', '1'], 'กันอะ'),
                               (['1256', '124', '1', '134', '135', '1', '156', '1235'],
                                'เต็มอะไร')]:
            result = decode_thai_details(physical(patterns), lexicon=set())
            self.assertEqual((result['text'], result['status']), (text, 'NEEDS_REVIEW'))
            self.assertEqual(sum(p['valid'] for p in result['segmentation_runs'][0]['candidates']), 2)

    def test_compound_vowels_and_known_consonants_remain_valid(self):
        for patterns, text in [(['1245', '135', '1'], 'เกาะ'),
                               (['1245', '126', '1'], 'แกะ'),
                               (['123', '146', '1'], 'เลอะ'),
                               (['1235', '135', '1'], 'เราะ'),
                               (['135', '1'], 'อะ'),
                               (['125'], 'ห'), (['1235'], 'ร')]:
            result = decode_thai_details(physical(patterns))
            self.assertEqual((result['text'], result['status']), (text, 'VALID'))

    def test_captured_diagnostic_raw_patterns_now_validate_without_mutation(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/diag_039_confirmation.json')
                             .read_text(encoding='utf-8'))
        source = []
        for line_id, line in enumerate(fixture['lines']):
            cells = physical(line['patterns'])
            for index, cell in enumerate(cells):
                cell.update(line_id=line_id, cell_id=f'C{len(source) + index + 1}')
                cell['break_before'] = index in line['gaps_before']
            source.extend(cells)
        original = deepcopy(source)
        expected = ('กล่องของขวัญมีกบ\nกบ กระโดดออกจากกล่องของขวัญ\n'
                    'แกะตกใจกระโดดกอด กา กับ ไก่\nกา กับ ไก่ หัวเราะ แกะ ก็หัวเราะ')
        for _ in range(3):
            result = decode_thai_details(source, lexicon=set())
            self.assertEqual((result['text'], result['status']), (expected, 'VALID'))
            self.assertEqual(result['raw_patterns'], [p for line in fixture['lines'] for p in line['patterns']])
            self.assertEqual(source, original)


if __name__ == '__main__':
    unittest.main()
