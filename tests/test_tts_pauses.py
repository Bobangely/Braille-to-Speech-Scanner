import unittest

from edge_stream import plan_edge_chunks
from tts import prepare_thai_speech_text
from thai_speech import THAI_CONSONANT_NAMES


class SpeechPauseTests(unittest.TestCase):
    def test_spaces_lines_and_sentences_choose_longest_boundary(self):
        source = '  เขียด เขียว  ขอน\r\nไข่.\nกิน! คน? จบฯ ต่อ  '
        self.assertEqual(plan_edge_chunks(source, 'th-TH-NiwatNeural'), [
            ('เขียด', 30), ('เขียว', 80), ('ขอน', 130), ('ไข่.', 200),
            ('กิน!', 200), ('คน?', 200), ('จบฯ', 200), ('ต่อ', 0)])

    def test_no_split_inside_thai_words_or_decimal_numbers(self):
        self.assertEqual(plan_edge_chunks('กินคนเกี๊ยว 3.14 บาท', 'th-TH-NiwatNeural'),
                         [('กินคนเกี๊ยว', 30), ('3.14', 30), ('บาท', 0)])
        self.assertEqual(plan_edge_chunks('   \n\t', 'th-TH-NiwatNeural'), [])
        self.assertEqual(plan_edge_chunks('...!?', 'th-TH-NiwatNeural'), [])

    def test_consonant_names_and_source_text_are_preserved(self):
        source = '\n'.join(THAI_CONSONANT_NAMES)
        prepared = prepare_thai_speech_text(source)
        chunks = plan_edge_chunks(prepared, 'th-TH-NiwatNeural')
        self.assertEqual([text for text, _ in chunks], list(THAI_CONSONANT_NAMES.values()))
        self.assertEqual([pause for _, pause in chunks], [130]*43+[0])
        self.assertEqual(source, '\n'.join(THAI_CONSONANT_NAMES))

    def test_standalone_list_markers_stay_with_the_following_words(self):
        source = '- คำควบกล้ำตัว กล เช่น กล่อ\nง\nนิทานตัว ข'
        chunks = plan_edge_chunks(prepare_thai_speech_text(source), 'th-TH-NiwatNeural')
        self.assertEqual(chunks[0], ('- คำควบกล้ำตัว', 30))
        self.assertTrue(all(any(char.isalnum() for char in text) for text, _ in chunks))
        self.assertIn(('งองู', 130), chunks)
        self.assertEqual(chunks[-1], ('ขอไข่', 0))
        self.assertEqual(plan_edge_chunks('คำ\n• ต่อ -', 'th-TH-NiwatNeural'),
                         [('คำ', 130), ('• ต่อ -', 0)])

    def test_config_and_english_are_respected(self):
        self.assertEqual(plan_edge_chunks('กิน คน\nไข่.', 'th-TH-NiwatNeural',
                         dict(space=200, phrase=350, line=500, sentence=600)),
                         [('กิน', 200), ('คน', 500), ('ไข่.', 0)])
        self.assertEqual(plan_edge_chunks('Hello world.\nNext', 'en-US-JennyNeural'),
                         [('Hello world.\nNext', 0)])
        for invalid in (-1, float('nan'), '250'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                plan_edge_chunks('กิน คน', 'th-TH-NiwatNeural', dict(space=invalid))


if __name__ == '__main__':
    unittest.main()
