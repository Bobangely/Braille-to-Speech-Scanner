"""Acceptance examples from thai_braille_word_composition_rules.md (2026-09-30)."""

from copy import deepcopy
import unittest

from thai_parser import compose_tokens
from thai_renderer import CompositionError, compose_syllable


CASES = [
    ('ก', 'อะ', None, 'กะ'), ('ก', 'อะ', 'น', 'กัน'), ('ร', 'อะ', 'ก', 'รัก'),
    ('ม', 'อือ', None, 'มือ'), ('ค', 'อือ', 'น', 'คืน'),
    ('ก', 'เอะ', None, 'เกะ'), ('ต', 'เอะ', 'ม', 'เต็ม'), ('ก', 'เอะ', 'บ', 'เก็บ'),
    ('ก', 'แอะ', None, 'แกะ'), ('ข', 'แอะ', 'ง', 'แข็ง'),
    ('ก', 'โอะ', None, 'โกะ'), ('ค', 'โอะ', 'น', 'คน'), ('ม', 'โอะ', 'ด', 'มด'),
    ('ก', 'เอาะ', None, 'เกาะ'), ('น', 'เอาะ', 'ต', 'น็อต'), ('ล', 'เอาะ', 'ก', 'ล็อก'),
    ('จ', 'เออ', None, 'เจอ'), ('ก', 'เออ', 'ด', 'เกิด'), ('ด', 'เออ', 'น', 'เดิน'),
    ('ต', 'เออ', 'ม', 'เติม'), ('ค', 'เออ', 'ย', 'เคย'), ('ล', 'เออ', 'ย', 'เลย'),
    ('ต', 'อัว', None, 'ตัว'), ('ส', 'อัว', 'น', 'สวน'), ('ข', 'อัว', 'ด', 'ขวด'),
    ('ร', 'เอีย', 'น', 'เรียน'), ('ด', 'เอือ', 'น', 'เดือน'),
    ('ม', 'อา', None, 'มา'), ('ม', 'อา', 'ก', 'มาก'), ('ก', 'อิ', 'น', 'กิน'),
    ('จ', 'อี', 'น', 'จีน'), ('ด', 'อึ', 'ง', 'ดึง'), ('ข', 'อุ', 'น', 'ขุน'),
    ('พ', 'อู', 'ด', 'พูด'), ('ก', 'เอ', 'ม', 'เกม'), ('ด', 'แอ', 'ง', 'แดง'),
    ('ค', 'โอ', 'ม', 'โคม'), ('บ', 'ออ', 'ก', 'บอก'), ('ท', 'เออ', 'ม', 'เทอม'),
    ('ก', 'อำ', None, 'กำ'), ('ก', 'ไอ', None, 'ไก'), ('จ', 'ใอ', None, 'ใจ'),
    ('ก', 'เอา', None, 'เกา'), ('ล', 'เออะ', None, 'เลอะ'),
]


class ThaiWordComposerTests(unittest.TestCase):
    def test_specification_rules_and_parser(self):
        for onset, vowel, final, expected in CASES:
            with self.subTest(expected=expected):
                self.assertEqual(compose_syllable(onset, vowel, final), expected)
                tokens = [onset, vowel] + ([final] if final else [])
                result = compose_tokens(tokens)
                self.assertEqual(result['text'], expected)
                self.assertFalse(result['needs_review'])
                self.assertEqual(result['raw_tokens'], tokens)
                self.assertEqual(result['syllables'][0]['final'], final)

    def test_ko_kai_oe_do_dek_and_aliases(self):
        for alias in ('เออ', 'เ-อ', 'สระเออ', 'เ◌อ'):
            result = compose_tokens(['ก', alias, 'ด'])
            self.assertEqual(result['text'], 'เกิด')
            self.assertEqual(result['normalized_tokens'][1]['vowel'], 'เออ')
            self.assertEqual(result['raw_tokens'][1], alias)

    def test_contextual_o_short_a_is_a_semantic_short_o_vowel(self):
        for tokens, text in [(['ก', 'อ', 'ะ'], 'เกาะ'),
                             (['ก', 'อ', 'ะ', 'ข', 'อ', 'น'], 'เกาะขอน'),
                             (['ห', 'อ', 'ะ'], 'เหาะ'),
                             (['ก', 'ล', 'อ', 'ะ'], 'เกลาะ'),
                             (['ก', '่', 'อ', 'ะ'], 'เก่าะ'),
                             (['ก', 'อ', 'ะ', '่'], 'เก่าะ')]:
            with self.subTest(tokens=tokens):
                original = deepcopy(tokens)
                result = compose_tokens(tokens)
                self.assertEqual(result['text'], text)
                self.assertEqual(result['syllables'][0]['vowel'], 'เอาะ')
                self.assertFalse(result['needs_review'])
                self.assertEqual(result['raw_tokens'], original)
                self.assertEqual(tokens, original)

    def test_o_consonant_and_written_long_o_are_not_rewritten(self):
        for tokens, text in [(['ข', 'อ', 'น'], 'ขอน'),
                             (['อ', 'ะ'], 'อะ'), (['อ', 'ิ', '่', 'ม'], 'อิ่ม'),
                             (['อ', 'อ', 'ม'], 'ออม'),
                             (['ก', 'ะ', 'อ', 'ะ'], 'กะอะ'),
                             (['ก', 'ิ', 'น', 'อ', 'ะ'], 'กินอะ'),
                             (['เ', 'ก', 'อ', 'ะ'], 'เกอะ')]:
            with self.subTest(tokens=tokens):
                result = compose_tokens(tokens)
                self.assertEqual(result['text'], text)
                self.assertEqual(result['raw_tokens'], tokens)
                if text == 'กินอะ':
                    self.assertTrue(result['needs_review'])
                    self.assertEqual(result['status'], 'NEEDS_REVIEW')
                else:
                    self.assertFalse(result['needs_review'])

    def test_compound_o_short_a_cannot_cross_parsing_boundaries(self):
        for boundary in (' ', '\n', '-'):
            for tokens, text in [(['ก', boundary, 'อ', 'ะ'], 'ก'+boundary+'อะ'),
                                 (['ก', 'อ', boundary, 'ะ'], 'กอ'+boundary+'ะ')]:
                with self.subTest(tokens=tokens):
                    self.assertEqual(compose_tokens(tokens)['text'], text)
        for metadata, text in [(dict(break_before=True), 'กอ ะ'),
                               (dict(line_id=1), 'กอ\nะ'),
                               (dict(warning='unknown_or_incomplete_symbol'), 'กอ�')]:
            tokens = [dict(char=c, line_id=0) for c in ['ก', 'อ', 'ะ']]
            tokens[-1].update(metadata)
            original = deepcopy(tokens)
            result = compose_tokens(tokens)
            self.assertEqual(result['text'], text)
            self.assertEqual(result['raw_tokens'], original)
            self.assertEqual(tokens, original)

    def test_tones_and_cluster_anchor(self):
        for onset, vowel, final, tone, expected in [
            ('ก', 'อา', None, '่', 'ก่า'), ('ก', 'อี', None, '่', 'กี่'),
            ('ข', 'อัว', None, '่', 'ขั่ว'), ('กว', 'อา', 'ง', '้', 'กว้าง'),
            ('ก', 'เออ', 'ด', '่', 'เกิ่ด'), ('ค', 'เออ', 'ย', '้', 'เค้ย'),
            ('ส', 'เอือ', None, '้', 'เสื้อ'), ('ป', 'เอียะ', None, '๊', 'เปี๊ยะ'),
        ]:
            with self.subTest(expected=expected):
                self.assertEqual(compose_syllable(onset, vowel, final, tone), expected)
                for marks in ([vowel, tone], [tone, vowel]):
                    result = compose_tokens([onset, *marks] + ([final] if final else []))
                    self.assertEqual(result['text'], expected)
        result = compose_tokens(['ก', 'ว', 'อา', '้', 'ง'])
        self.assertEqual(result['text'], 'กว้าง')
        self.assertEqual(result['syllables'][0]['tone_anchor'], 1)

    def test_word_boundaries_and_existing_orthography(self):
        for tokens, expected in [
            (['ก', 'เ-อ', ' ', 'ด'], 'เกอ ด'),
            (['ก', 'เ-อ', '\n', 'ด'], 'เกอ\nด'),
            (['ก', 'เ-อ', '-', 'ด'], 'เกอ-ด'),
            (['ก', 'เ-อ', 'ด', 'ม', 'า'], 'เกิดมา'),
            (['จ', 'เ-อ', 'ห', 'น', 'ู'], 'เจอหนู'),
            (['ก', 'ะ', 'ท', 'ิ'], 'กะทิ'),
            (['เ', 'ก', 'ิ', 'ด'], 'เกิด'),
            (['ใ', 'ค', 'ร'], 'ใคร'),
            (['ก', 'ื'], 'กื'),
            (['ท', 'เ-อ', 'ม', ' ', 'จ', 'เ-อ', ' ', 'ค', 'เ-อ', 'ย'], 'เทอม เจอ เคย'),
        ]:
            with self.subTest(tokens=tokens):
                self.assertEqual(compose_tokens(tokens)['text'], expected)

    def test_invalid_and_lexicon_dependent_forms(self):
        for vowel, status in [('อำ', 'INVALID_FINAL'), ('ไอ', 'INVALID_FINAL'),
                              ('ใอ', 'INVALID_FINAL'), ('เอา', 'INVALID_FINAL'),
                              ('เอียะ', 'NEEDS_LEXICON'), ('เอือะ', 'NEEDS_LEXICON'),
                              ('อัวะ', 'NEEDS_LEXICON'), ('สระไม่รู้จัก', 'UNKNOWN_VOWEL')]:
            with self.subTest(vowel=vowel):
                with self.assertRaises(CompositionError) as error:
                    compose_syllable('ก', vowel, 'ด')
                self.assertEqual(error.exception.status, status)
        for vowel in ('อำ', 'ไอ', 'ใอ', 'เอา', 'เอียะ', 'เอือะ', 'อัวะ'):
            result = compose_tokens(['ก', vowel, 'ด'])
            self.assertTrue(result['needs_review'])
            self.assertEqual(result['text'], '�')
        for vowel in ('เอะ', 'แอะ', 'เอาะ'):
            with self.assertRaises(CompositionError):
                compose_syllable('ก', vowel, 'ด', '้')
        result = compose_tokens(['ง', 'เออะ', 'น'])
        self.assertEqual(result['syllables'][0]['candidate'], 'เงิน')
        self.assertEqual(result['syllables'][0]['status'], 'NEEDS_LEXICON')
        self.assertEqual(compose_tokens(['ง', 'เออะ', 'น'], lexicon={'เงิน'})['text'], 'เงิน')

    def test_boundary_metadata_and_source_preservation(self):
        for boundary, expected in [(dict(break_before=True), 'เกอ ด'),
                                   (dict(line_id=1), 'เกอ\nด')]:
            tokens = [dict(char=char, line_id=0, confidence=.99) for char in ['ก', 'เ◌อ', 'ด']]
            tokens[-1].update(boundary)
            original = deepcopy(tokens)
            self.assertEqual(compose_tokens(tokens)['text'], expected)
            self.assertEqual(tokens, original)
        tokens = [dict(token=c, confidence=confidence) for c, confidence in [('ก', .98), ('เออ', .63), ('ด', .97)]]
        result = compose_tokens(tokens, confidence_threshold=.8)
        self.assertEqual(result['syllables'][0]['candidate'], 'เกิด')
        self.assertEqual(result['syllables'][0]['status'], 'LOW_CONFIDENCE')
        self.assertEqual(result['syllables'][0]['min_confidence'], .63)


if __name__ == '__main__':
    unittest.main()
