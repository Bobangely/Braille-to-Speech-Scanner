"""Speech for finger events: newest event wins, unknown cells are never guessed."""
import unittest

from finger_state import EnterCell, FinishWord
from page_layout import CellSlot, LinePair, PageMap
from speech_queue import SpeechQueue


class RecordingSpeaker:
    def __init__(self):
        self.calls = []
        self.prepared = []
        self.fail = False

    def speak(self, text, lang):
        if self.fail:
            raise RuntimeError('audio device gone')
        self.calls.append(('speak', text))
        return True

    def stop(self):
        self.calls.append('stop')

    def prepare_many(self, texts, lang):
        self.prepared.extend(texts)
        return set() if self.fail else set(texts)


def page(words=('ไก่', 'ม')):
    cells = (CellSlot(0, 0.0, 6.0, frozenset({1}), 'สระไอ', 0),
             CellSlot(1, 6.0, 12.0, frozenset({2}), '๕', 0),
             CellSlot(2, 12.0, 18.0, frozenset({3}), None, 1))
    return PageMap((LinePair(0, 10.0, 20.0, cells),), words)


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

    def test_one_symbol_word_is_not_repeated_after_its_name(self):
        # 'ก ข ค': each letter is a word; its name was spoken on entry, so leaving it stays quiet.
        self.queue.prepare_page(page(words=('ก', 'ไก่')))
        self.queue.handle([EnterCell(0, 0, 'กอไก่')])
        self.queue.handle([FinishWord(0, 'ก')])
        self.assertEqual(self.speaker.calls, ['stop', ('speak', 'กอไก่')])

    def test_unprepared_word_is_spelled(self):
        with self.assertLogs('speech_queue', 'WARNING'):
            self.queue.handle([FinishWord(0, 'ไก่')])
        self.assertEqual(self.speaker.calls[-1], ('speak', 'สระไอ กอไก่ ไม้เอก'))

    def test_prepare_page_prepares_everything_the_page_can_say(self):
        self.queue.prepare_page(page())
        self.assertEqual(set(self.speaker.prepared),
                         {'ไก่', 'ม', 'สระไอ กอไก่ ไม้เอก', 'มอม้า', 'สระไอ', '๕', 'ไม่ทราบ'})
        self.assertTrue(self.queue.ready('ไก่'))
        self.queue.handle([FinishWord(0, 'ไก่')])
        self.assertEqual(self.speaker.calls[-1], ('speak', 'ไก่'))

    def test_failed_preparation_is_not_ready(self):
        self.speaker.fail = True
        with self.assertLogs('speech_queue', 'WARNING'):
            self.queue.prepare_page(page())
        self.assertFalse(self.queue.ready('ไก่'))

    def test_say_interrupts_and_speaks_text(self):
        self.queue.say('พร้อม 3 บรรทัด')
        self.assertEqual(self.speaker.calls, ['stop', ('speak', 'พร้อม 3 บรรทัด')])

    def test_playback_failure_is_logged_not_raised(self):
        self.speaker.fail = True
        with self.assertLogs('speech_queue', 'ERROR'):
            self.queue.handle([EnterCell(0, 0, 'สระไอ')])


if __name__ == '__main__':
    unittest.main()
