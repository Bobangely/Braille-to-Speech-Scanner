"""Physical Braille cells -> raw tokens -> composer -> pre-TTS text."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from decoder import decode_cells, decode_cells_verbose
from tests.test_punctuation_recovery import logical_cells
from thai_composer import compose_tokens
from thai_decoder import decode_thai_details, tokenize_thai
from tts import TextToSpeech


PATTERNS = [
    ('เกิด', ['1245', '146', '145']),
    ('เคย', ['136', '146', '13456']),
    ('เต็ม', ['124', '1256', '1', '134']),
    ('เต็ม', ['1256', '124', '1', '134']),
    ('แข็ง', ['126', '13', '1', '12456']),
    ('คน', ['24', '136', '1', '1345']),
    ('กัน', ['1245', '1', '1345']),
    ('สวน', ['234', '15', '1345']),
    ('คืน', ['136', '26', '1345']),
    ('น็อต', ['124', '1345', '16', '1', '1256']),
    ('เทอม', ['23456', '146', '134']),
    ('เจอ', ['245', '146']),
    ('ตัว', ['1256', '15']),
    ('เรียน', ['1235', '12356', '1345']),
    ('เดือน', ['145', '12345', '1345']),
    ('เกิดมา', ['1245', '146', '145', '134', '16']),
    ('กะทิ', ['1245', '1', '23456', '12']),
]


class ThaiWordPipelineTests(unittest.TestCase):
    def test_real_patterns_keep_cells_and_overlay_labels(self):
        for expected, patterns in PATTERNS:
            with self.subTest(expected=expected, patterns=patterns):
                cells = logical_cells(patterns)
                original = deepcopy(cells)
                raw = tokenize_thai(cells)
                result = decode_thai_details(cells)
                self.assertEqual(result['text'], expected)
                self.assertEqual(result['raw_tokens'], raw)
                self.assertFalse(result['needs_review'])
                self.assertEqual(cells, original)
                verbose = decode_cells_verbose(cells, 'thai')
                self.assertEqual([t['char'] for t in verbose], [t['char'] for t in raw])
                self.assertFalse(any(t['warning'] for t in verbose))
                json.dumps(result, ensure_ascii=False)
                json.dumps(verbose, ensure_ascii=False)

    def test_ko_kai_oe_do_dek_trace(self):
        result = decode_thai_details(logical_cells(['1245', '146', '145']))
        self.assertEqual([t['char'] for t in result['raw_tokens']], ['ก', 'เ◌อ', 'ด'])
        self.assertEqual(result['text'], 'เกิด')
        syllable = result['syllables'][0]
        self.assertEqual([syllable[k] for k in ('onset', 'vowel', 'tone', 'final')], ['ก', 'เออ', None, 'ด'])
        self.assertEqual(syllable['indices'], [0, 1, 2])

    def test_space_line_and_detection_failure_stop_composition(self):
        for change, expected in [(dict(reading_x=500), 'เกอ ด'),
                                 (dict(line_id=1), 'เกอ\nด'),
                                 (dict(row_ambiguous=True), 'เกอ�'),
                                 (dict(dots=set(), crop_status='empty'), 'เกอ�')]:
            with self.subTest(change=change):
                cells = logical_cells(['1245', '146', '145'])
                cells[-1].update(change)
                self.assertEqual(decode_cells(cells, 'thai'), expected)
        self.assertEqual(decode_cells(logical_cells(['1245', '146', '', '145']), 'thai'), 'เกอ ด')

    def test_review_status_reaches_existing_confirmation_gate(self):
        # Closed short เออะ needs lexical validation, not an unmarked guess.
        cells = logical_cells(['12456', '146', '1', '1345'])
        self.assertIn('�', decode_cells(cells, 'thai'))
        self.assertIn('NEEDS_LEXICON', [t['warning'] for t in decode_cells_verbose(cells, 'thai')])
        ambiguous = compose_tokens(['ก', 'เ-อ', 'ด', 'น'])
        self.assertTrue(ambiguous['needs_review'])
        self.assertEqual(ambiguous['syllables'][0]['status'], 'AMBIGUOUS_SYLLABLE')
        self.assertEqual(ambiguous['raw_tokens'], ['ก', 'เ-อ', 'ด', 'น'])

    def test_composed_output_is_sent_to_tts(self):
        with patch.object(TextToSpeech, '_init_offline_engine'):
            speaker = TextToSpeech()
        for expected, patterns in PATTERNS:
            with self.subTest(expected=expected):
                with patch.object(speaker, '_speak_online', return_value=True) as online:
                    text = decode_cells(logical_cells(patterns), 'thai')
                    speaker.speak(text, lang='thai', method='online')
                    online.assert_called_once_with(expected, lang='th', save_file=None)

    def test_english_is_not_composed(self):
        self.assertEqual(decode_cells(logical_cells(['1245', '146', '145']), 'english'), 'g[1,4,6]d')
        self.assertTrue(all('composition' not in t for t in decode_cells_verbose(
            logical_cells(['1245', '146', '145']), 'english')))


if __name__ == '__main__':
    unittest.main()
