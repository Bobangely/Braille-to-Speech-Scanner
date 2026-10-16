from array import array
import unittest

from edge_stream import insert_word_pauses, word_pause_position


class EdgeWordPauseTests(unittest.TestCase):
    def test_metadata_maps_only_source_whitespace_not_internal_thai_syllables(self):
        text = 'กอไก่ ขอไข่'
        cursor, position = word_pause_position(text, dict(text='กอ', offset=0, duration=1000), 0)
        self.assertIsNone(position)
        cursor, position = word_pause_position(text, dict(text='ไก่', offset=1000, duration=1000), cursor)
        self.assertEqual(position, 5)
        cursor, position = word_pause_position(text, dict(text='ขอไข่', offset=2000, duration=1000), cursor)
        self.assertIsNone(position)
        with self.assertRaises(ValueError):
            word_pause_position(text, dict(text='ขอไข่', offset=0, duration=1000), 0)

    def test_two_milliseconds_adds_48_samples_without_losing_or_reordering_audio(self):
        stats = {}
        output = array('h')
        for block in insert_word_pauses([array('h', [1, 2, 3, 4])], [2], 2, stats):
            output.extend(block)
        self.assertEqual(output.tolist(), [1, 2]+[0]*48+[3, 4])
        self.assertEqual(stats['word_pauses_inserted'], 1)

    def test_late_metadata_is_skipped_instead_of_inserting_at_wrong_position(self):
        markers, stats = [], {}
        def blocks():
            yield array('h', [1, 2, 3])
            markers.append(1)
            yield array('h', [4, 5])
        output = array('h')
        for block in insert_word_pauses(blocks(), markers, 2, stats):
            output.extend(block)
        self.assertEqual(output.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(stats['late_word_boundaries'], 1)


if __name__ == '__main__':
    unittest.main()
