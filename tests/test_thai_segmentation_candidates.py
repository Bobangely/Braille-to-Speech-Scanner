"""Competing Thai boundaries must not silently become a confirmed reading."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np

from camera_reader import AsyncBrailleWorker, RealTimeBrailleScanner
from decoder import decode_cells_verbose
from thai_composer import compose_tokens
from thai_decoder import decode_thai_details
from thai_braille import PATTERN_CANDIDATES
from thai_parser import parse_braille
from thai_renderer import render_syllable
from tests.test_camera_recovery import await_condition
from tests.test_thai_syllable_boundaries import physical


class ThaiSegmentationCandidateTests(unittest.TestCase):
    def test_requested_regressions_keep_coda_and_independent_o(self):
        cases = [
            (['1245', '1', '1345', '135', '1'], 'กันอะ', 'กะเนาะ'),
            (['1256', '124', '1', '134', '135', '1', '156', '1235'], 'เต็มอะไร', 'เตะเมาะไร'),
            (['13', '126', '1', '12456', '135', '1'], 'แข็งอะ', 'แขะเงาะ'),
        ]
        for patterns, preferred, other in cases:
            with self.subTest(preferred=preferred):
                source = physical(patterns)
                original = deepcopy(source)
                result = decode_thai_details(source)
                self.assertEqual(source, original)
                self.assertEqual(result['raw_patterns'], patterns)
                self.assertEqual(result['text'], preferred)
                self.assertEqual(result['status'], 'NEEDS_REVIEW')
                boundary = result['syllables'][0]
                self.assertEqual(boundary['boundary_evidence'], 'COMPETING_SEGMENTATIONS')
                self.assertEqual(boundary['status'], 'AMBIGUOUS_SEGMENTATION')
                candidates = boundary['boundary_candidates']
                self.assertEqual(len(candidates), 2)
                self.assertTrue(all(candidate['valid'] for candidate in candidates))
                texts = {''.join(render_syllable(unit) for unit in candidate['syllables'])
                         for candidate in candidates}
                self.assertEqual(texts, {preferred, other})

    def test_invalid_closed_vowel_path_cannot_override_valid_next_onset(self):
        source = physical(['1245', '1356', '1235', '135', '1'])
        result = decode_thai_details(source)
        self.assertEqual(result['text'], 'กำเราะ')
        self.assertEqual(result['status'], 'VALID')
        boundary = result['syllables'][0]
        self.assertEqual(boundary['boundary_evidence'], 'UNIQUE_VALID_SEGMENTATION')
        self.assertEqual([c['validation_error'] for c in boundary['boundary_candidates']],
                         ['INVALID_FINAL', None])

    def test_tone_and_vowel_validity_are_checked_before_selecting_boundary(self):
        result = decode_thai_details(physical(['1245', '35', '124', '1', '1235', '135', '1']))
        self.assertEqual(result['text'], 'เก่ะเราะ')
        self.assertEqual(result['status'], 'VALID')
        self.assertEqual(result['syllables'][0]['boundary_candidates'][0]['validation_error'],
                         'UNSUPPORTED_COMBINATION')

    def test_unresolved_alternative_is_not_treated_as_proven_invalid(self):
        source = physical(['125', '15', '1235', '135', '1'])
        with patch('thai_parser._segmentation_error', side_effect=['UNRESOLVED_NUCLEUS', None]):
            result = decode_thai_details(source)
        self.assertEqual(result['text'], 'หัวเราะ')
        self.assertEqual(result['status'], 'NEEDS_REVIEW')
        self.assertEqual(result['syllables'][0]['boundary_evidence'], 'UNRESOLVED_SEGMENTATIONS')
        self.assertEqual(result['syllables'][0]['status'], 'UNRESOLVED_SEGMENTATION')

    def test_single_125_has_no_invented_prefix_or_boundary_candidate(self):
        self.assertEqual([(c['kind'], c['value']) for c in PATTERN_CANDIDATES['125']],
                         [('consonant', 'ห')])
        for pattern in ('125', '521', '251'):
            source = physical([pattern])
            original = deepcopy(source)
            result = decode_thai_details(source)
            self.assertEqual(result['text'], 'ห')
            self.assertEqual(result['status'], 'VALID')
            self.assertEqual(result['syllables'][0]['boundary_candidates'], [])
            verbose = decode_cells_verbose(source, 'thai')
            self.assertEqual(verbose[0]['char'], 'ห')
            self.assertFalse(verbose[0].get('warning'))
            self.assertEqual(source, original)

    def test_125_at_each_sentence_position_keeps_its_single_cell_reading(self):
        for patterns, expected in [
                (['125', '', '1245', '1', '', '245', '1'], 'ห กะ จะ'),
                (['1245', '1', '', '125', '', '245', '1'], 'กะ ห จะ'),
                (['1245', '1', '', '245', '1', '', '125'], 'กะ จะ ห')]:
            with self.subTest(expected=expected):
                source = physical(patterns)
                result = decode_thai_details(source)
                self.assertEqual(result['text'], expected)
                self.assertEqual(result['status'], 'VALID')
                token = decode_cells_verbose(source, 'thai')[patterns.index('125')]
                self.assertEqual(token['char'], 'ห')
                self.assertFalse(token.get('warning'))

    def test_boundary_warning_does_not_blame_the_previous_onset(self):
        source = physical(['125', '15', '1235', '135', '1'])
        original = deepcopy(source)
        result = decode_thai_details(source)
        boundary = result['syllables'][0]
        self.assertEqual(result['text'], 'หัวเราะ')
        self.assertEqual(result['status'], 'NEEDS_REVIEW')
        self.assertEqual(boundary['boundary_indices'], [2])
        self.assertEqual(boundary['boundary_patterns'], ['1235'])
        self.assertEqual(boundary['boundary_source_cells'], ['S2'])
        verbose = decode_cells_verbose(source, 'thai')
        self.assertEqual(verbose[0]['char'], 'ห')
        self.assertFalse(verbose[0].get('warning'))
        self.assertEqual(verbose[2]['warning'], 'AMBIGUOUS_SEGMENTATION')
        self.assertEqual(verbose[2]['segmentation']['boundary_indices'], [2])
        self.assertEqual(source, original)

    def test_no_valid_segmentation_is_unresolved_not_ambiguous(self):
        source = physical(['125', '15', '1235', '135', '1'])
        with patch('thai_parser._segmentation_error',
                   side_effect=['UNRESOLVED_NUCLEUS', 'NEEDS_LEXICON']):
            result = decode_thai_details(source)
        self.assertEqual(result['status'], 'NEEDS_REVIEW')
        self.assertEqual(result['syllables'][0]['status'], 'UNRESOLVED_SEGMENTATION')
        self.assertFalse(any(c['valid'] for c in result['syllables'][0]['boundary_candidates']))

    def test_warning_location_is_general_and_uses_global_source_indices(self):
        for onset in ('125', '1245', '123', '234'):
            source = physical(['245', '1', '', onset, '15', '1235', '135', '1'])
            result = decode_thai_details(source)
            boundary = next(s for s in result['syllables'] if s.get('boundary_review'))
            self.assertEqual(boundary['boundary_indices'], [5])
            self.assertEqual(boundary['boundary_source_cells'], ['S5'])
            self.assertEqual(sum(c['valid'] for c in boundary['boundary_candidates']), 2)
            verbose = decode_cells_verbose(source, 'thai')
            self.assertFalse(verbose[3].get('warning'))
            self.assertEqual(verbose[5]['warning'], 'AMBIGUOUS_SEGMENTATION')

    def test_125_in_contiguous_syllables_has_no_prefix_ambiguity(self):
        for patterns, expected in [
                (['125', '1', '1245', '1'], 'หะกะ'),
                (['1245', '1', '125', '1', '245', '1'], 'กะหะจะ'),
                (['1245', '1', '125', '1'], 'กะหะ')]:
            result = decode_thai_details(physical(patterns))
            self.assertEqual(result['text'], expected)
            self.assertEqual(result['status'], 'VALID')
            self.assertFalse(any(t.get('warning') for t in
                                 decode_cells_verbose(physical(patterns), 'thai')))

    def test_explicit_boundaries_resolve_compound_vs_coda_without_guessing(self):
        for split, expected in ((3, 'กัน อะ'), (2, 'กะ เนาะ')):
            source = physical(['1245', '1', '1345', '135', '1'])
            source[split]['break_before'] = True
            result = decode_thai_details(source)
            self.assertEqual(result['text'], expected)
            self.assertEqual(result['status'], 'VALID')
            self.assertFalse(any(s['boundary_candidates'] for s in result['syllables']))

    def test_standalone_vowels_and_existing_compound_patterns_remain_valid(self):
        cases = [(['1245', '135', '1'], 'เกาะ'),
                 (['1245', '126', '1'], 'แกะ'),
                 (['123', '146', '1'], 'เลอะ'),
                 (['13', '135', '1345'], 'ขอน')]
        for patterns, expected in cases:
            with self.subTest(expected=expected):
                result = decode_thai_details(physical(patterns))
                self.assertEqual(result['text'], expected)
                self.assertEqual(result['status'], 'VALID')

    def test_ambiguous_compound_candidate_keeps_display_but_never_valid(self):
        source = physical(['125', '15', '1235', '135', '1'])
        parsed = parse_braille(source)
        self.assertNotIn('text', parsed)
        self.assertTrue(parsed['syllables'][0]['boundary_review'])
        result = decode_thai_details(source)
        self.assertEqual(result['text'], 'หัวเราะ')
        self.assertEqual(result['status'], 'NEEDS_REVIEW')
        self.assertTrue(any(t.get('warning') == 'AMBIGUOUS_SEGMENTATION'
                            for t in decode_cells_verbose(source, 'thai')))

    def test_text_token_adapter_obeys_the_same_review_contract(self):
        for tokens, expected in [(['ก', 'ะ', 'น', 'อ', 'ะ'], 'กันอะ'),
                                 (['ต', 'เ-ะ', 'ม', 'อ', 'ะ', 'ไ', 'ร'], 'เต็มอะไร')]:
            result = compose_tokens(tokens)
            self.assertEqual(result['text'], expected)
            self.assertEqual(result['status'], 'NEEDS_REVIEW')

    def test_worker_confirmation_cannot_accept_repeated_ambiguous_candidates(self):
        source = physical(['1245', '1', '1345', '135', '1'])

        class Detector:
            def detect(self, image, lang='thai'):
                return deepcopy(source), dict(dots=[], grid_pending=False)

        scanner = RealTimeBrailleScanner(res_preset='480p')
        worker = AsyncBrailleWorker(Detector())
        scanner.ai_worker = worker
        scanner._preview = Mock(motion_valid=True)
        scanner._preview.render.side_effect = lambda frame, *args: frame.copy()
        scanner._draw_top_hud = Mock()
        scanner._draw_notice = Mock()
        worker.start()
        try:
            for frame_id in range(1, 8):
                frame = np.full((120, 320, 3), 255, np.uint8)
                scanner._render_frame(frame, frame_id)
                result = await_condition(lambda: (r if (r := worker.get_latest_results())['frame_id'] == frame_id else None))
                self.assertEqual(result['decoded_text'], 'กันอะ')
                self.assertEqual(result['status'], 'uncertain')
                self.assertIn('AMBIGUOUS_SEGMENTATION', result['reason'])
                scanner._render_frame(frame, frame_id)
                self.assertFalse(scanner.history)
                self.assertFalse(scanner._confirmed_text)
        finally:
            self.assertTrue(worker.stop())


if __name__ == '__main__':
    unittest.main()
