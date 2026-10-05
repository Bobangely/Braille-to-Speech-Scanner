"""Keep Thai short-i plus tone distinct from long-i through decoding and TTS."""

import unittest
from unittest.mock import patch

from decoder import decode_cells, decode_cells_verbose, normalize_thai_text
from tts import TextToSpeech


def cells(patterns):
    # Explicit patterns, not reverse-generated from the mapping under test.
    return [dict(dots=frozenset(map(int, pattern))) for pattern in patterns]


WORDS = [('วิ่ง', '2456'), ('กิ่ง', '1245'), ('นิ่ง', '1345'),
         ('สิ่ง', '234'), ('ยิ่ง', '13456')]
SENTENCE = 'คุณแม่ตกใจวิ่งหนีจนลืม'
SENTENCE_PATTERNS = [
    '136', '14', '6', '1345',       # คุณ
    '126', '134', '35',             # แม่
    '1256', '1245', '156', '2', '245',  # ตกใจ
    '2456', '12', '35', '12456',    # วิ่ง
    '125', '1345', '23',            # หนี
    '245', '1345', '123', '26', '134',  # จนลืม
]


class ThaiCompositionTests(unittest.TestCase):
    def test_short_i_and_mai_ek_words(self):
        for expected, consonant in WORDS:
            for marks in [('12', '35'), ('35', '12')]:
                with self.subTest(word=expected, marks=marks):
                    result = decode_cells(cells([consonant, *marks, '12456']), 'thai')
                    self.assertEqual(result, expected)
                    self.assertEqual([ord(c) for c in result[1:3]], [0x0E34, 0x0E48])
                    self.assertNotIn('\u0e35', result)

    def test_verbose_tokens_preserve_vowel_and_tone_separately(self):
        # Same C13-C16 patterns as the captured diag_023 manifest.
        tokens = decode_cells_verbose(cells(['2456', '12', '35', '12456']), 'thai')
        self.assertEqual([token['char'] for token in tokens], ['ว', 'ิ', '่', 'ง'])
        self.assertTrue(all(not token['consumed'] for token in tokens))
        self.assertTrue(all(token['warning'] is None for token in tokens))

    def test_complete_sentence(self):
        for language in ('thai', 'th'):
            with self.subTest(language=language):
                self.assertEqual(decode_cells(cells(SENTENCE_PATTERNS), language), SENTENCE)

    def test_normalization_preserves_marks_and_is_idempotent(self):
        for word, _ in WORDS:
            for source in (word, word.replace('ิ่', '่ิ')):
                with self.subTest(source=source):
                    result = normalize_thai_text(source)
                    self.assertEqual(result, word)
                    self.assertEqual(normalize_thai_text(result), result)

    def test_long_i_is_not_guessed_to_be_short_i_with_tone(self):
        for patterns, expected in [(['2456', '23', '12456'], 'วีง'),
                                   (['2456', '23', '35', '12456'], 'วี่ง'),
                                   (['2456', '12', '12456'], 'วิง')]:
            with self.subTest(expected=expected):
                self.assertEqual(decode_cells(cells(patterns), 'thai'), expected)
                self.assertEqual(normalize_thai_text(expected), expected)

    def test_final_text_passed_to_tts_keeps_vowel_and_tone(self):
        # Mock only the audio backends: exercise the real pre-TTS normalization
        # without initializing devices, using the network, or producing files.
        with patch.object(TextToSpeech, '_init_offline_engine'):
            speaker = TextToSpeech()
        samples = [(cells([consonant, '12', '35', '12456']), word)
                   for word, consonant in WORDS]
        samples.append((cells(SENTENCE_PATTERNS), SENTENCE))
        for input_cells, expected in samples:
            text = decode_cells(input_cells, 'thai')
            for language in ('thai', 'th'):
                with self.subTest(expected=expected, language=language):
                    with patch.object(speaker, '_speak_online', return_value=True) as online:
                        self.assertTrue(speaker.speak(text, lang=language, method='online'))
                        online.assert_called_once_with(expected, lang='th', save_file=None)
                    with patch.object(speaker, '_speak_offline', return_value=True) as offline:
                        self.assertTrue(speaker.speak(text, lang=language, method='offline'))
                        offline.assert_called_once_with(expected, is_thai=True)

    def test_english_decode_and_tts_unchanged(self):
        for language in ('english', 'en'):
            with self.subTest(language=language):
                text = decode_cells(cells(['6', '145', '3456', '1', '12']), language)
                self.assertEqual(text, 'D12')
                with patch.object(TextToSpeech, '_init_offline_engine'):
                    speaker = TextToSpeech()
                with patch.object(speaker, '_speak_online', return_value=True) as online:
                    self.assertTrue(speaker.speak(text, lang=language, method='online'))
                    online.assert_called_once_with('D12', lang='en', save_file=None)


if __name__ == '__main__':
    unittest.main()
