"""
Unit tests for finger_braille_reader module.
"""

import unittest
import numpy as np

from finger_braille_reader import (
    FingerBrailleTouchReader,
    IndexFingerSkeletonDetector,
    THAI_VOWEL_TONE_NAMES,
)
from thai_speech import THAI_CONSONANT_NAMES


class TestFingerBrailleReader(unittest.TestCase):
    def setUp(self):
        self.reader = FingerBrailleTouchReader()

    def tearDown(self):
        self.reader.close()

    def test_detector_initialization(self):
        self.assertTrue(self.reader.detector.available)

    def test_grid_cache_update_and_lookup(self):
        cells = [
            {'center': (100, 100), 'width': 30, 'height': 50, 'dots': [1, 2]},
            {'center': (200, 100), 'width': 30, 'height': 50, 'dots': [1, 3]}
        ]
        verbose = [{'char': 'ข'}, {'char': 'ก'}]

        self.reader.update_grid_cache(cells, verbose)
        self.assertEqual(len(self.reader.cached_cells), 2)

        # Point inside cell 0
        idx, item = self.reader.find_cell_at_point((102, 105))
        self.assertEqual(idx, 0)
        self.assertEqual(item['char'], 'ข')

        # Point inside cell 1
        idx, item = self.reader.find_cell_at_point((198, 95))
        self.assertEqual(idx, 1)
        self.assertEqual(item['char'], 'ก')

        # Point far away
        idx, item = self.reader.find_cell_at_point((500, 500))
        self.assertEqual(idx, -1)

    def test_pronunciation_mapping(self):
        # Consonant
        self.assertEqual(self.reader.get_spoken_name('ก'), 'กอไก่')
        self.assertEqual(self.reader.get_spoken_name('ข'), 'ขอไข่')

        # Vowel & Tone
        self.assertEqual(self.reader.get_spoken_name('ะ'), 'สระอะ')
        self.assertEqual(self.reader.get_spoken_name('า'), 'สระอา')
        self.assertEqual(self.reader.get_spoken_name('่'), 'ไม้เอก')

        # English
        self.assertEqual(self.reader.get_spoken_name('a'), 'Letter A')

    def test_word_mode_mapping(self):
        cells = [
            {'center': (100, 100), 'width': 30, 'height': 50, 'dots': [1]},
            {'center': (140, 100), 'width': 30, 'height': 50, 'dots': [1, 6]}
        ]
        verbose = [{'char': 'ก'}, {'char': 'า'}]
        self.reader.update_grid_cache(cells, verbose, decoded_text="กา")
        self.assertEqual(self.reader.cached_cells[0]['word'], 'กา')
        self.assertEqual(self.reader.cached_cells[1]['word'], 'กา')
        self.assertEqual(self.reader.cached_cells[0]['word_id'], self.reader.cached_cells[1]['word_id'])

    def test_grid_lock_and_hand_presence_prevents_overwrite(self):
        cells = [{'center': (100, 100), 'width': 30, 'height': 50, 'dots': [1]}]
        verbose = [{'char': 'ก'}]
        self.reader.update_grid_cache(cells, verbose)
        self.assertEqual(len(self.reader.cached_cells), 1)

        # 1. When grid is locked, updates must be ignored
        self.reader.lock_grid()
        self.reader.update_grid_cache([], [])
        self.assertEqual(len(self.reader.cached_cells), 1)
        self.reader.unlock_grid()

        # 2. When hand is present, updates must be ignored
        self.reader.is_hand_present = True
        self.reader.update_grid_cache([{'center': (200, 200), 'width': 30, 'height': 50, 'dots': [1, 2]}], [{'char': 'ข'}])
        self.assertEqual(self.reader.cached_cells[0]['char'], 'ก')
        self.reader.is_hand_present = False

        # 3. When hand leaves, cache can update
        self.reader.update_grid_cache([{'center': (200, 200), 'width': 30, 'height': 50, 'dots': [1, 2]}], [{'char': 'ข'}])
        self.assertEqual(self.reader.cached_cells[0]['char'], 'ข')

    def test_process_empty_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out_frame, touched = self.reader.process(frame)
        self.assertEqual(out_frame.shape, (480, 640, 3))
        self.assertIsNone(touched)

    def test_dwell_threshold_prevents_accidental_swipes(self):
        from unittest.mock import Mock
        self.reader.speaker.speak = Mock()
        cells = [{'center': (100, 100), 'width': 30, 'height': 50, 'dots': [1]}]
        verbose = [{'char': 'ก'}]
        self.reader.update_grid_cache(cells, verbose, decoded_text="กา")

        # Mock index finger hovering directly over cell center
        self.reader.detector.detect_index_finger = Mock(
            return_value=((100, 100), [(100, 100), (100, 110), (100, 120), (100, 130)])
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Frames under dwell_threshold must NOT trigger speech
        for _ in range(self.reader.dwell_threshold - 1):
            self.reader.process(frame)
            self.reader.speaker.speak.assert_not_called()

        # Reaching dwell_threshold MUST trigger speech
        self.reader.process(frame)
        self.reader.speaker.speak.assert_called_once_with('กา', interrupt=True)

    def test_hold_finger_dwell_spells_character_in_word_mode(self):
        from unittest.mock import Mock, call
        self.reader.speaker.speak = Mock()
        cells = [
            {'center': (100, 100), 'width': 30, 'height': 50, 'dots': [1]},
            {'center': (140, 100), 'width': 30, 'height': 50, 'dots': [1, 6]}
        ]
        verbose = [{'char': 'ก'}, {'char': 'า'}]
        self.reader.update_grid_cache(cells, verbose, decoded_text="กา")

        # 1. Hover on Cell 0 ('ก')
        self.reader.detector.detect_index_finger = Mock(
            return_value=((100, 100), [(100, 100)] * 4)
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Process up to frame 8: should speak the word 'กา'
        for _ in range(8):
            self.reader.process(frame)
        self.assertEqual(self.reader.speaker.speak.call_args_list, [call('กา', interrupt=True)])

        # 2. Continue holding on Cell 0 ('ก') up to frame 18: should spell 'กอไก่'
        for _ in range(10):
            self.reader.process(frame)
        self.assertEqual(self.reader.speaker.speak.call_args_list, [
            call('กา', interrupt=True),
            call('กอไก่', interrupt=True)
        ])

        # 3. Slide to Cell 1 ('า') in the same word
        self.reader.detector.detect_index_finger = Mock(
            return_value=((140, 100), [(140, 100)] * 4)
        )
        # Should not re-read word 'กา', but holding for 18 frames should spell 'สระอา'
        for _ in range(18):
            self.reader.process(frame)
        self.assertEqual(self.reader.speaker.speak.call_args_list, [
            call('กา', interrupt=True),
            call('กอไก่', interrupt=True),
            call('สระอา', interrupt=True)
        ])

    def test_grid_bbox_exact_matching(self):
        cells = [{
            'center': (100, 100),
            'grid': {'bbox': (85, 75, 115, 125)}
        }]
        verbose = [{'char': 'ก'}]
        self.reader.update_grid_cache(cells, verbose, scale_x=1.5, scale_y=2.0)
        cached = self.reader.cached_cells[0]
        # x1 = 85 * 1.5 = 127.5 -> 128, y1 = 75 * 2.0 = 150
        # x2 = 115 * 1.5 = 172.5 -> 172, y2 = 125 * 2.0 = 250
        self.assertEqual(cached['bbox'], (128, 150, 172, 250))
        self.assertEqual(cached['center'], ((128 + 172) // 2, (150 + 250) // 2))

    def test_miss_grace_preserves_dwell_counter(self):
        from unittest.mock import Mock
        self.reader.speaker.speak = Mock()
        cells = [{'center': (100, 100), 'grid': {'bbox': (80, 80, 120, 120)}}]
        verbose = [{'char': 'ก'}]
        self.reader.update_grid_cache(cells, verbose, decoded_text="ก")

        touch_mock = ((100, 100), [(100, 100)] * 4)
        miss_mock = ((500, 500), [(500, 500)] * 4)  # outside
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 3 frames touch
        self.reader.detector.detect_index_finger = Mock(return_value=touch_mock)
        for _ in range(3):
            self.reader.process(frame)
        self.assertEqual(self.reader.dwell_counter, 3)

        # 1 frame jitter / miss -> grace period keeps it active!
        self.reader.detector.detect_index_finger = Mock(return_value=miss_mock)
        self.reader.process(frame)
        self.assertEqual(self.reader.dwell_counter, 4)

        # Return to touch -> completes to dwell_threshold (6)
        self.reader.detector.detect_index_finger = Mock(return_value=touch_mock)
        self.reader.process(frame)
        self.reader.process(frame)
        self.reader.speaker.speak.assert_called_once_with('ก', interrupt=True)


if __name__ == '__main__':
    unittest.main()
