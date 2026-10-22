"""Lexical evidence ranks complete structural paths, never their validity."""
from collections import Counter
from copy import deepcopy
import unittest
import subprocess
import sys
from unittest.mock import patch

from thai_composer import compose_tokens
from thai_decoder import decode_thai_details
from thai_lexicon import default_lexicon
from tests.test_thai_syllable_boundaries import physical


class ThaiLexiconRankingTests(unittest.TestCase):
    def test_unique_complete_multisyllable_evidence_resolves_reported_line(self):
        source = physical(['1256', '1', '1245', '1235', '256', '16', '12456', '135', '1'])
        result = decode_thai_details(source, lexicon={'ตะกร้า', 'เงาะ'})
        self.assertEqual(result['text'], 'ตะกร้าเงาะ')
        self.assertEqual(result['status'], 'VALID')
        self.assertEqual(result['lexicon_ranking'][0]['reason'], 'UNIQUE_COMPLETE_LEXICAL_SUPPORT')
        self.assertTrue(all(c['valid'] and c['plausible'] for c in result['lexicon_ranking'][0]['candidates']))

    def test_dictionary_cannot_resolve_known_rival_or_incomplete_or_invalid_evidence(self):
        patterns = ['1256', '1', '1245', '1235', '256', '16', '12456', '135', '1']
        for words, flags in [({'ตะกร้า', 'เงาะ', 'ตะกร้าง', 'อะ'}, {}),
                             ({'ตะกร้า'}, {}),
                             ({'ตะกร้า', 'เงาะ'}, {'geometry_status': 'invalid'})]:
            with self.subTest(words=words, flags=flags):
                source = physical(patterns)
                source[0].update(flags)
                self.assertEqual(decode_thai_details(source, lexicon=words)['status'], 'NEEDS_REVIEW')

    def test_low_confidence_or_missing_dictionary_keeps_reported_phrase_reviewed(self):
        source = physical(['1256', '1', '1245', '1235', '256', '16', '12456', '135', '1'])
        source[0]['confidence'] = .2
        result = decode_thai_details(source, lexicon={'ตะกร้า', 'เงาะ'}, confidence_threshold=.8)
        self.assertEqual(result['status'], 'NEEDS_REVIEW')
        self.assertTrue(result['lexicon_ranking'][0]['needs_review'])
        self.assertEqual(decode_thai_details(source, lexicon=set())['status'], 'NEEDS_REVIEW')

    def test_single_rare_or_proper_name_entry_does_not_resolve_competitors(self):
        source = physical(['1256', '1', '1245', '1235', '256', '16', '12456', '135', '1'])
        for name in ('ตะกร้างอะ', 'ตะกร้าเงาะ'):
            result = decode_thai_details(source, lexicon={name})
            self.assertEqual(result['status'], 'NEEDS_REVIEW')
            self.assertTrue(all(c['valid'] and c['plausible'] for c in result['lexicon_ranking'][0]['candidates']))

    def test_lexicon_ranks_valid_paths_without_confirming_an_unknown_competitor(self):
        result = decode_thai_details(physical(['1245', '1', '1345', '135', '1']),
                                     lexicon={'กะ', 'เนาะ'})
        self.assertEqual(result['text'], 'กะเนาะ')
        self.assertEqual(result['status'], 'NEEDS_REVIEW')
        ranking = result['lexicon_ranking'][0]
        self.assertEqual(ranking['reason'], 'LEXICON_SUPPORTED_DISPLAY')
        self.assertEqual(ranking['candidates'][0]['text'], 'กะเนาะ')
        self.assertEqual(ranking['candidates'][0]['known_characters'], 6)
        self.assertTrue(all(c['valid'] for c in ranking['candidates']))
        unknown = next(c for c in ranking['candidates'] if c['text'] == 'กันอะ')
        self.assertEqual(unknown['known_characters'], 0)
        self.assertTrue(unknown['plausible'])

    def test_equal_lexical_support_keeps_the_p1_preference_and_review(self):
        result = decode_thai_details(physical(['1245', '1', '1345', '135', '1']),
                                     lexicon={'กัน', 'อะ', 'กะ', 'เนาะ'})
        self.assertEqual(result['text'], 'กันอะ')
        self.assertEqual(result['status'], 'NEEDS_REVIEW')
        self.assertEqual(result['lexicon_ranking'][0]['reason'], 'LEXICON_TIE')

    def test_missing_lexical_evidence_preserves_unknown_valid_candidates(self):
        result = decode_thai_details(physical(['125', '15', '1235', '135', '1']),
                                     compatibility=True, lexicon={'กะ'})
        self.assertEqual(result['text'], 'หัวเราะ')
        self.assertEqual(result['status'], 'NEEDS_REVIEW')
        self.assertEqual(result['lexicon_ranking'][0]['reason'], 'NO_LEXICAL_EVIDENCE')
        self.assertTrue(all(c['valid'] and c['plausible']
                            for c in result['lexicon_ranking'][0]['candidates']))

    def test_new_word_with_unique_structure_is_not_rejected(self):
        result = compose_tokens(['ฮ', 'อิ', 'ล'], lexicon={'กะ'})
        self.assertEqual(result['text'], 'ฮิล')
        self.assertEqual(result['status'], 'VALID')

    def test_invalid_closed_structure_cannot_win_even_with_a_lexicon_entry(self):
        result = decode_thai_details(physical(['1245', '1356', '1235', '135', '1']),
                                     lexicon={'กำร', 'อะ'})
        self.assertEqual(result['text'], 'กำเราะ')
        self.assertEqual(result['status'], 'VALID')
        run = result['segmentation_runs'][0]
        self.assertTrue(any(not c['valid'] and c['validation_error'] == 'INVALID_FINAL'
                            for c in run['candidates']))
        ranking = result['lexicon_ranking'][0]
        self.assertEqual(len(ranking['candidates']), 1)
        self.assertEqual(ranking['candidates'][0]['text'], 'กำเราะ')

    def test_all_nested_boundary_combinations_are_generated_before_ranking(self):
        patterns = ['1245', '1', '1345', '135', '1'] * 2
        source = physical(patterns)
        original = deepcopy(source)
        result = decode_thai_details(source, lexicon={'กันอะกะเนาะ'})
        self.assertEqual(result['text'], 'กันอะกะเนาะ')
        self.assertEqual(result['status'], 'NEEDS_REVIEW')
        ranking = result['lexicon_ranking'][0]
        self.assertEqual({c['text'] for c in ranking['candidates']},
                         {'กันอะกันอะ', 'กันอะกะเนาะ', 'กะเนาะกันอะ', 'กะเนาะกะเนาะ'})
        self.assertTrue(result['segmentation_runs'][0]['generation_complete'])
        for candidate in result['segmentation_runs'][0]['candidates']:
            self.assertEqual(Counter(i for s in candidate['syllables'] for i in s['indices']),
                             Counter(range(10)))
        self.assertEqual(source, original)
        self.assertEqual(result['raw_patterns'], patterns)
        self.assertTrue(result['parser_trace'])
        self.assertTrue(result['semantic_tokens'])

    def test_lexical_matches_do_not_cross_space_or_line_boundaries(self):
        for boundary in ('break_before', 'line_id'):
            source = physical(['1245', '1', '1345', '135', '1'] * 2)
            if boundary == 'line_id':
                for cell in source[5:]:
                    cell['line_id'] = 1
            else:
                source[5]['break_before'] = True
            result = decode_thai_details(source, lexicon={'กะเนาะกะเนาะ'})
            separator = '\n' if boundary == 'line_id' else ' '
            self.assertEqual(result['text'], 'กันอะ' + separator + 'กันอะ')
            self.assertEqual(len(result['lexicon_ranking']), 2)
            self.assertTrue(all(r['reason'] == 'NO_LEXICAL_EVIDENCE'
                                for r in result['lexicon_ranking']))

    def test_previous_p1_and_single_cell_regressions_remain_safe(self):
        cases = [(['1245', '1', '1345', '135', '1'], 'กันอะ', 'NEEDS_REVIEW'),
                 (['1256', '124', '1', '134', '135', '1', '156', '1235'], 'เต็มอะไร', 'NEEDS_REVIEW'),
                 (['125'], 'ห', 'VALID'), (['1235'], 'ร', 'VALID')]
        for patterns, text, status in cases:
            with self.subTest(text=text):
                result = decode_thai_details(physical(patterns), lexicon={text})
                self.assertEqual(result['text'], text)
                self.assertEqual(result['status'], status)

    def test_candidate_limit_fails_closed_without_ranking_a_partial_search(self):
        with patch('thai_parser.MAX_SEGMENTATION_CANDIDATES', 2):
            result = decode_thai_details(physical(['1245', '1', '1345', '135', '1'] * 2),
                                         lexicon={'กะเนาะกะเนาะ'})
        self.assertEqual(result['status'], 'NEEDS_REVIEW')
        self.assertEqual(result['text'], 'กันอะกันอะ')
        self.assertFalse(result['segmentation_runs'][0]['generation_complete'])
        self.assertFalse(result['lexicon_ranking'][0]['candidates'])
        self.assertEqual(result['lexicon_ranking'][0]['reason'], 'INCOMPLETE_CANDIDATE_SEARCH')

    def test_unavailable_default_lexicon_keeps_structural_output(self):
        default_lexicon.cache_clear()
        self.addCleanup(default_lexicon.cache_clear)
        with patch.dict('sys.modules', {'pythainlp.corpus': None}):
            with self.assertLogs('thai_lexicon', 'WARNING'):
                result = decode_thai_details(physical(['125', '15', '1235', '135', '1']),
                                             compatibility=True)
        self.assertEqual(result['text'], 'หัวเราะ')
        self.assertEqual(result['status'], 'NEEDS_REVIEW')
        self.assertFalse(result['lexicon_info']['available'])

    def test_packaged_lexicon_is_available_offline_and_only_changes_ranking(self):
        default_lexicon.cache_clear()
        self.addCleanup(default_lexicon.cache_clear)
        with patch('urllib.request.urlopen', side_effect=AssertionError('runtime network access')):
            result = decode_thai_details(physical(['125', '15', '1235', '135', '1']),
                                         compatibility=True)
        self.assertTrue(result['lexicon_info']['available'])
        self.assertEqual(result['text'], 'หัวเราะ')
        self.assertEqual(result['status'], 'NEEDS_REVIEW')
        candidates = result['lexicon_ranking'][0]['candidates']
        self.assertGreater(candidates[0]['coverage'], candidates[1]['coverage'])

    def test_lexicon_does_not_mutate_original_parser_trace(self):
        source = physical(['1245', '1', '1345', '135', '1'])
        baseline = decode_thai_details(source, lexicon=set())
        ranked = decode_thai_details(source, lexicon={'กะ', 'เนาะ'})
        self.assertEqual(ranked['parser_trace'], baseline['parser_trace'])
        self.assertEqual(ranked['semantic_tokens'], baseline['semantic_tokens'])
        self.assertEqual(ranked['raw_tokens'], baseline['raw_tokens'])
        self.assertEqual(ranked['structural_syllables'], baseline['structural_syllables'])
        self.assertEqual(ranked['syllables'][0]['raw_patterns'], ['1245', '1'])
        self.assertEqual(ranked['lexicon_ranking'][0]['indices'], [0, 1, 2, 3, 4])

    def test_ranked_review_is_attached_to_the_boundary_owner_not_a_previous_syllable(self):
        source = physical(['245', '1', '1245', '1', '1345', '135', '1'])
        result = decode_thai_details(source, lexicon={'จะ', 'กะ', 'เนาะ'})
        self.assertEqual(result['text'], 'จะกะเนาะ')
        self.assertFalse(result['syllables'][0]['needs_review'])
        self.assertTrue(result['syllables'][1]['needs_review'])
        self.assertEqual(result['syllables'][1]['boundary_indices'], [4])

    def test_long_competing_run_fails_closed_without_recursive_suffix_expansion(self):
        # Isolate the enumeration stack from unittest's own frames. The old
        # structural decoder handles this same run at this stack limit.
        code = """
import sys
from thai_decoder import decode_thai_details
from tests.test_thai_syllable_boundaries import physical
sys.setrecursionlimit(40)
result = decode_thai_details(physical(['1245', '1', '1345', '135', '1'] * 45), lexicon=set())
assert result['status'] == 'NEEDS_REVIEW'
assert not result['segmentation_runs'][0]['generation_complete']
assert not result['lexicon_ranking'][0]['candidates']
"""
        outcome = subprocess.run([sys.executable, '-X', 'utf8', '-c', code],
                                 capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(outcome.returncode, 0, outcome.stderr)


if __name__ == '__main__':
    unittest.main()
