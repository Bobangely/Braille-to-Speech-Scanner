import unittest
from array import array

from edge_stream import pack_audio_frames, plan_edge_chunks, trim_edge_padding
from tts import prepare_thai_speech_text
from thai_speech import THAI_CONSONANT_NAMES


class SpeechPauseTests(unittest.TestCase):
    def test_default_reads_whole_lines_and_keeps_all_44_spoken_names(self):
        voice = 'th-TH-NiwatNeural'
        self.assertEqual(plan_edge_chunks('เขียด เขียว  ขอน\nไข่. คน กิน', voice),
                         [('เขียด เขียว ขอน', 10), ('ไข่.', 50), ('คน กิน', 0)])
        source = ' '.join(THAI_CONSONANT_NAMES)
        spoken = prepare_thai_speech_text(source)
        self.assertEqual(plan_edge_chunks(spoken, voice),
                         [(' '.join(THAI_CONSONANT_NAMES.values()), 0)])

    def test_playback_buffers_are_full_until_the_final_audio_block(self):
        def fragments():
            yield array('h')
            for samples in ([1], [2, 3], [4, 5]):
                yield array('h', samples)
        packed = pack_audio_frames(fragments())
        next(packed)
        self.assertEqual(packed.send(3).tolist(), [1, 2, 3])
        self.assertEqual(packed.send(3).tolist(), [4, 5])
        with self.assertRaises(StopIteration):
            packed.send(3)

    def test_only_exact_zero_padding_is_removed_and_internal_silence_is_preserved(self):
        blocks = [array('h', [0, 0, 1, 0]), array('h', [0, 0]), array('h', [-1, 0, 0])]
        result = array('h')
        for block in trim_edge_padding(blocks):
            result.extend(block)
        self.assertEqual(result.tolist(), [1, 0, 0, 0, -1])
        self.assertEqual(list(trim_edge_padding([array('h', [0, 0])])), [])

    def test_spaces_lines_and_sentences_choose_longest_boundary(self):
        source = '  เขียด เขียว  ขอน\r\nไข่.\nกิน! คน? จบฯ ต่อ  '
        self.assertEqual(plan_edge_chunks(source, 'th-TH-NiwatNeural'), [
            ('เขียด เขียว ขอน', 10), ('ไข่.', 50),
            ('กิน!', 50), ('คน?', 50), ('จบฯ', 50), ('ต่อ', 0)])

    def test_no_split_inside_thai_words_or_decimal_numbers(self):
        self.assertEqual(plan_edge_chunks('กินคนเกี๊ยว 3.14 บาท', 'th-TH-NiwatNeural'),
                         [('กินคนเกี๊ยว 3.14 บาท', 0)])
        self.assertEqual(plan_edge_chunks('   \n\t', 'th-TH-NiwatNeural'), [])
        self.assertEqual(plan_edge_chunks('...!?', 'th-TH-NiwatNeural'), [])

    def test_consonant_names_and_source_text_are_preserved(self):
        source = '\n'.join(THAI_CONSONANT_NAMES)
        prepared = prepare_thai_speech_text(source)
        chunks = plan_edge_chunks(prepared, 'th-TH-NiwatNeural')
        self.assertEqual([text for text, _ in chunks], list(THAI_CONSONANT_NAMES.values()))
        self.assertEqual([pause for _, pause in chunks], [10]*43+[0])
        self.assertEqual(source, '\n'.join(THAI_CONSONANT_NAMES))

    def test_standalone_list_markers_stay_with_the_following_words(self):
        source = '- คำควบกล้ำตัว กล เช่น กล่อ\nง\nนิทานตัว ข'
        chunks = plan_edge_chunks(prepare_thai_speech_text(source), 'th-TH-NiwatNeural')
        self.assertEqual(chunks[0], ('- คำควบกล้ำตัว กล เช่น กล่อ', 10))
        self.assertTrue(all(any(char.isalnum() for char in text) for text, _ in chunks))
        self.assertIn(('งองู', 10), chunks)
        self.assertEqual(chunks[-1], ('นิทานตัว ขอไข่', 0))
        self.assertEqual(plan_edge_chunks('คำ\n• ต่อ -', 'th-TH-NiwatNeural'),
                         [('คำ', 10), ('• ต่อ -', 0)])

    def test_config_and_english_are_respected(self):
        self.assertEqual(plan_edge_chunks('กิน คน', 'th-TH-NiwatNeural', dict(space=0)),
                         [('กิน คน', 0)])
        self.assertEqual(plan_edge_chunks('กิน คน\nไข่.', 'th-TH-NiwatNeural',
                         dict(space=200, phrase=350, line=500, sentence=600)),
                         [('กิน', 200), ('คน', 500), ('ไข่.', 0)])
        self.assertEqual(plan_edge_chunks('Hello world.\nNext', 'en-US-JennyNeural'),
                         [('Hello world.\nNext', 0)])
        for invalid in (-1, float('nan'), '250'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                plan_edge_chunks('กิน คน', 'th-TH-NiwatNeural', dict(space=invalid))

    def test_fractional_pause_is_supported_and_nonfinite_values_are_rejected(self):
        self.assertEqual(plan_edge_chunks('กิน คน', 'th-TH-NiwatNeural', dict(space=.5)),
                         [('กิน', .5), ('คน', 0)])
        for invalid in (float('inf'), float('-inf'), True, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                plan_edge_chunks('กิน คน', 'th-TH-NiwatNeural', dict(space=invalid))


if __name__ == '__main__':
    unittest.main()
