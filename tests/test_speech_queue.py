"""Speech for finger events: newest event wins, unknown cells are never guessed."""
import unittest

from finger_state import EnterCell, FinishWord
from page_layout import CellSlot, LinePair, PageMap
from speech_queue import SpeechQueue


class RecordingSpeaker:
    def __init__(self):
        self.calls = []
        self.prepared = []

    def speak(self, text, lang):
        self.calls.append(('speak', text))
        return True

    def stop(self):
        self.calls.append('stop')

    def prepare(self, text, lang):
        self.prepared.append(text)


def page(words=('ไก่', 'ม')):
    cell = CellSlot(0, 0.0, 6.0, frozenset({1}), 'สระไอ', 0)
    return PageMap((LinePair(0, 10.0, 20.0, (cell,)),), words)


class SpeechQueueTests(unittest.TestCase):
    def setUp(self):
        self.speaker = RecordingSpeaker()
        self.queue = SpeechQueue(self.speaker)

    def test_new_event_stops_previous_audio(self):
        self.queue.handle([EnterCell(0, 0, 'สระไอ')])
        self.queue.handle([EnterCell(0, 1, 'กอไก่')])
        self.assertEqual(self.speaker.calls, ['stop', ('speak', 'สระไอ'), 'stop', ('speak', 'กอไก่')])

    def test_only_newest_event_in_a_frame_is_spoken(self):
        with self.assertLogs('speech_queue', 'WARNING'):
            self.queue.handle([EnterCell(0, 2, 'ไม้เอก'), FinishWord(0, 'ไก่')])
        self.assertEqual(self.speaker.calls[-1], ('speak', 'สระไอ กอไก่ ไม้เอก'))
        self.assertEqual(self.speaker.calls.count('stop'), 1)

    def test_no_events_keeps_current_audio(self):
        self.queue.handle([])
        self.assertEqual(self.speaker.calls, [])

    def test_unknown_cell_says_mai_sap(self):
        self.queue.handle([EnterCell(0, 0, None)])
        self.assertEqual(self.speaker.calls[-1], ('speak', 'ไม่ทราบ'))

    def test_unprepared_word_is_spelled(self):
        with self.assertLogs('speech_queue', 'WARNING'):
            self.queue.handle([FinishWord(0, 'ไก่')])
        self.assertEqual(self.speaker.calls[-1], ('speak', 'สระไอ กอไก่ ไม้เอก'))

    def test_prepare_page_prepares_every_word(self):
        self.queue.prepare_page(page())
        self.assertEqual(self.speaker.prepared, ['ไก่', 'ม'])
        self.assertTrue(self.queue.ready('ไก่'))
        self.queue.handle([FinishWord(0, 'ไก่')])
        self.assertEqual(self.speaker.calls[-1], ('speak', 'ไก่'))

    def test_say_interrupts_and_speaks_text(self):
        self.queue.say('พร้อม 3 บรรทัด')
        self.assertEqual(self.speaker.calls, ['stop', ('speak', 'พร้อม 3 บรรทัด')])

    def test_failed_preparation_is_not_ready(self):
        def broken(text, lang):
            raise OSError('no model')
        self.speaker.prepare = broken
        with self.assertLogs('speech_queue', 'WARNING'):
            self.queue.prepare_page(page())
        self.assertFalse(self.queue.ready('ไก่'))


if __name__ == '__main__':
    unittest.main()
