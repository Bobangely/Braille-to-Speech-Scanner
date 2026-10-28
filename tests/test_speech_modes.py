"""Speech modes: 'spell' names every cell, 'word' names only short words' cells."""
import unittest

from finger_state import EnterCell, FinishWord
from page_layout import CellSlot, LinePair, PageMap
from speech_queue import SpeechQueue
from tests.test_speech_queue import RecordingSpeaker

NAMES = ('กอไก่', 'ซอโซ่', 'สระอา', 'ขอไข่', 'คอควาย', 'งองู', 'จอจาน', 'ฉอฉิ่ง')


def page():
    """Word 0 'ก' = cell 0; word 1 'กา' = cells 1-2; word 2 'กระดาน' = cells 3-7."""
    words = [0, 1, 1, 2, 2, 2, 2, 2]
    cells = tuple(CellSlot(i, i * 6.0, i * 6.0 + 6.0, frozenset({1}), NAMES[i], w)
                  for i, w in enumerate(words))
    return PageMap((LinePair(0, 10.0, 20.0, cells),), ('ก', 'กา', 'กระดาน'))


def spoken(speaker):
    return [c[1] for c in speaker.calls if c != 'stop']


class SpeechModeTests(unittest.TestCase):
    def setUp(self):
        self.speaker = RecordingSpeaker()
        self.queue = SpeechQueue(self.speaker, mode='word')
        self.queue.prepare_page(page())

    def test_spell_mode_unchanged(self):
        self.queue.mode = 'spell'
        self.queue.handle([EnterCell(0, 3, NAMES[3])])
        self.assertEqual(spoken(self.speaker), [NAMES[3]])

    def test_word_mode_long_word_cells_silent(self):
        for i in range(3, 8):
            self.queue.handle([EnterCell(0, i, NAMES[i])])
        self.assertEqual(self.speaker.calls, [])

    def test_word_mode_long_word_spoken_at_end(self):
        self.queue.handle([EnterCell(0, 7, NAMES[7])])
        self.queue.handle([FinishWord(2, 'กระดาน')])
        self.assertEqual(spoken(self.speaker), ['กระดาน'])

    def test_word_mode_short_word_spelled(self):
        self.queue.handle([EnterCell(0, 1, NAMES[1])])
        self.queue.handle([EnterCell(0, 2, NAMES[2])])
        self.queue.handle([FinishWord(1, 'กา')])
        self.assertEqual(spoken(self.speaker), [NAMES[1], NAMES[2], 'กา'])

    def test_word_mode_threshold_one(self):
        self.queue.spell_max_cells = 1
        self.queue.handle([EnterCell(0, 1, NAMES[1])])
        self.queue.handle([EnterCell(0, 0, NAMES[0])])
        self.assertEqual(spoken(self.speaker), [NAMES[0]])

    def test_word_mode_unready_word_spells(self):
        self.speaker.prepare_many = lambda texts, lang: {t for t in texts if t != 'กระดาน'}
        queue = SpeechQueue(self.speaker, mode='word')
        queue.prepare_page(page())
        queue.handle([EnterCell(0, 4, NAMES[4])])
        self.assertEqual(spoken(self.speaker), [NAMES[4]])

    def test_word_mode_unknown_cell(self):
        self.queue.handle([EnterCell(0, 1, None)])
        self.assertEqual(spoken(self.speaker), ['ไม่ทราบ'])

    def test_mode_switch_mid_word(self):
        self.queue.mode = 'spell'
        self.queue.handle([EnterCell(0, 3, NAMES[3])])
        self.queue.mode = 'word'
        self.queue.handle([EnterCell(0, 4, NAMES[4])])
        self.assertEqual(spoken(self.speaker), [NAMES[3]])

    def test_last_spoken(self):
        self.assertIsNone(SpeechQueue(self.speaker).last_spoken)
        self.queue.handle([FinishWord(2, 'กระดาน')])
        self.assertEqual(self.queue.last_spoken, 'กระดาน')

    def test_cell_without_page_spelled(self):
        queue = SpeechQueue(self.speaker, mode='word')
        queue.handle([EnterCell(0, 4, NAMES[4])])
        self.assertEqual(spoken(self.speaker), [NAMES[4]])


if __name__ == '__main__':
    unittest.main()
