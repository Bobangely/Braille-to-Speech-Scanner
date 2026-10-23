"""FingerReader: page learning, reading and page changes, with every device faked."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from finger_reader import CachedWavSpeaker, FingerReader
from finger_state import EnterCell
from page_layout import CellSlot, LinePair, PageMap
from speech_queue import SpeechQueue

FRAME = np.zeros((10, 10, 3), np.uint8)


def page(lines=3):
    cells = (CellSlot(0, 10.0, 16.0, frozenset({1, 5, 6}), 'สระไอ', 0),)
    return PageMap(tuple(LinePair(i, 20.0 + 30 * i, 30.0 + 30 * i, cells) for i in range(lines)), ('ไ',))


class Calibration:
    """Identity px -> mm, so a fingertip pixel is its page millimetre."""

    def to_page(self, points):
        return np.asarray(points, dtype=float).reshape(-1, 2)


class FakeCalibrator:
    def __init__(self):
        self.result = (Calibration(), True)

    def update(self, frame, now_s):
        result, self.result = self.result, (self.result[0], False)
        return result


class FakeTracker:
    def __init__(self):
        self.tip, self.hand_present = None, False

    def update(self, frame, now_ms):
        return self.tip


class RecordingSpeaker:
    def __init__(self):
        self.spoken = []

    def speak(self, text, lang):
        self.spoken.append(text)

    def stop(self):
        pass

    def prepare_many(self, texts, lang):
        return set(texts)


class FingerReaderTests(unittest.TestCase):
    def setUp(self):
        self.calibrator, self.tracker, self.speaker = FakeCalibrator(), FakeTracker(), RecordingSpeaker()
        self.reads = []
        self.lines_on_page = 3

        def read_page(frame, calibration, fmt):
            self.reads.append(frame)
            return page(self.lines_on_page)

        self.reader = FingerReader(self.tracker, self.calibrator, SpeechQueue(self.speaker),
                                   read_page=read_page, required_maps=3)

    def learn(self, start=0.0):
        return [self.reader.step(FRAME, start + i * 0.1) for i in range(3)][-1]

    def test_no_markers_asks_to_place_page(self):
        self.calibrator.result = (None, False)
        self.assertEqual(self.reader.step(FRAME, 0.0), 'waiting_page')
        self.reader.step(FRAME, 0.1)
        self.assertEqual(self.speaker.spoken, ['วางหน้าให้เห็นมุมครบ'])

    def test_page_learning_waits_for_no_hand(self):
        self.tracker.hand_present = True
        self.assertEqual([self.reader.step(FRAME, i * 0.1) for i in range(5)], ['learning'] * 5)
        self.assertEqual(self.speaker.spoken, ['ยกมือออกก่อน'])
        self.assertEqual(self.reads, [])

    def test_learned_page_announces_lines(self):
        self.assertEqual(self.learn(), 'reading')
        self.assertEqual(self.speaker.spoken, ['พร้อม 3 บรรทัด'])

    def test_fingertip_mapped_to_mm_drives_events(self):
        self.learn()
        self.tracker.hand_present, self.tracker.tip = True, (13, 25)
        self.reader.step(FRAME, 1.0)
        self.reader.step(FRAME, 1.2)
        self.assertEqual(self.speaker.spoken[-1], 'สระไอ')
        self.assertEqual(self.reader.last_events, [EnterCell(0, 0, 'สระไอ')])
        self.assertAlmostEqual(self.reader.entered_s, 1.0)  # latency is measured from here

    def test_page_change_returns_to_learning(self):
        self.learn()
        self.calibrator.result = (Calibration(), True)
        self.assertEqual(self.reader.step(FRAME, 2.0), 'learning')
        self.tracker.tip = (13, 25)
        self.reader.step(FRAME, 2.1)
        self.assertEqual(self.reader.last_events, [])

    def test_page_flipped_in_place_is_learned_again(self):
        # Spiral book: the next sheet lands on the same markers, so calibration cannot tell.
        self.learn()
        self.lines_on_page = 2
        # Rechecked once a second after learning (t=0.2): three differing reads, then relearn.
        statuses = [self.reader.step(FRAME, 1.0 + i) for i in range(4)]
        self.assertEqual(statuses, ['reading', 'reading', 'reading', 'learning'])
        self.learn(start=6.0)
        self.assertEqual(self.speaker.spoken[-1], 'พร้อม 2 บรรทัด')

    def test_same_page_recheck_keeps_reading(self):
        self.learn()
        reads = len(self.reads)
        self.assertEqual([self.reader.step(FRAME, 1.0 + i * 0.1) for i in range(30)], ['reading'] * 30)
        self.assertLessEqual(len(self.reads) - reads, 4)  # about once a second, not every frame

    def test_hand_detection_dropouts_do_not_relearn_the_page(self):
        # MediaPipe misses the hand for a frame now and then while it still covers the dots.
        self.learn()
        self.lines_on_page = 2  # what a hand-covered page reads like
        statuses = []
        for i in range(60):  # 6 s at 10 frames/s, one missed detection every second
            self.tracker.hand_present = i % 10 != 9
            statuses.append(self.reader.step(FRAME, 1.0 + i * 0.1))
        self.assertEqual(set(statuses), {'reading'})

    def test_reading_can_start_on_any_line(self):
        # A reader may come back to line 2 without starting from the first letter.
        self.learn()
        self.tracker.hand_present, self.tracker.tip = True, (13, 55)  # line 2's raised band is 50-60 mm
        self.reader.step(FRAME, 1.0)
        self.reader.step(FRAME, 1.2)
        self.assertEqual(self.reader.last_events, [EnterCell(1, 0, 'สระไอ')])

    def test_relearning_says_why(self):
        self.learn()
        self.calibrator.result = (Calibration(), True)
        with self.assertLogs('finger_reader', 'INFO') as logs:
            self.reader.step(FRAME, 2.0)
        self.assertIn('markers moved', ' '.join(logs.output))

    def test_no_recheck_while_a_hand_is_reading(self):
        self.learn()
        reads = len(self.reads)
        self.tracker.hand_present = True
        self.lines_on_page = 2
        self.assertEqual([self.reader.step(FRAME, 1.0 + i) for i in range(5)], ['reading'] * 5)
        self.assertEqual(len(self.reads), reads)

    def test_lost_page_waits_again(self):
        self.learn()
        self.calibrator.result = (None, True)
        self.assertEqual(self.reader.step(FRAME, 2.0), 'waiting_page')

    def test_reader_exposes_what_the_preview_draws(self):
        self.learn()
        self.tracker.hand_present, self.tracker.tip = True, (13, 25)
        self.reader.step(FRAME, 1.0)
        self.reader.step(FRAME, 1.2)
        self.assertEqual(self.reader.tip, (13, 25))
        line, cell = self.reader.active_cell
        self.assertEqual((line.line_id, cell.index), (0, 0))
        self.assertIs(self.reader.page, self.reader._finger.page)


class OverlayTests(unittest.TestCase):
    """The live preview outlines every cell and word, and marks the finger's cell."""

    PX = 4  # fake calibration: 4 px per mm

    class Calibration:
        def to_image(self, points):
            return np.asarray(points, dtype=float).reshape(-1, 2) * OverlayTests.PX

    def setUp(self):
        cells = (CellSlot(0, 10.0, 16.0, frozenset({1}), 'สระไอ', 0), CellSlot(1, 16.0, 22.0, frozenset({1, 2}), 'กอไก่', 0),
                 CellSlot(2, 22.0, 28.0, frozenset(), None, None), CellSlot(3, 28.0, 34.0, frozenset({1, 3, 4}), 'มอม้า', 1))
        self.page = PageMap((LinePair(0, 40.0, 50.0, cells),), ('ไก่', 'ม'))
        self.frame = np.full((300, 200, 3), 255, np.uint8)

    def px(self, x_mm, y_mm):
        return self.frame[round(y_mm * self.PX), round(x_mm * self.PX)].tolist()

    def test_finger_cell_is_filled_green_on_the_raised_line(self):
        from finger_reader import ACTIVE_BGR, draw_overlay
        line = self.page.lines[0]
        draw_overlay(self.frame, self.page, self.Calibration(), (13 * self.PX, 45 * self.PX), (line, line.cells[1]))
        b, g, r = self.px(19.0, 45.0)  # middle of the raised cell under กอไก่
        self.assertGreater(g, 150)
        self.assertLess(r, 200)
        self.assertEqual(self.px(25.0, 45.0), [255, 255, 255])  # the empty cell is not outlined or filled

    def test_each_word_gets_its_own_outline_colour(self):
        from finger_reader import WORD_BGR, draw_overlay
        draw_overlay(self.frame, self.page, self.Calibration(), None, None)
        self.assertEqual(self.px(10.0, 45.0), list(WORD_BGR[0]))  # left edge of word 0
        self.assertEqual(self.px(34.0, 45.0), list(WORD_BGR[1]))  # right edge of word 1

    def test_status_bar_names_the_symbol_and_word(self):
        from finger_reader import draw_overlay
        line = self.page.lines[0]
        before = self.frame.copy()
        draw_overlay(self.frame, self.page, self.Calibration(), None, (line, line.cells[1]))
        self.assertFalse(np.array_equal(self.frame[-30:], before[-30:]))  # text drawn in the bottom bar


class CachedWavSpeakerTests(unittest.TestCase):
    def test_missing_tts_model_fails_at_startup(self):
        with tempfile.TemporaryDirectory() as cache:
            with self.assertRaises(FileNotFoundError):
                CachedWavSpeaker(Path(cache) / 'missing.onnx', Path(cache))

    def speaker(self, cache, batches, played, fail=()):
        model = Path(cache) / 'thai.onnx'
        model.write_bytes(b'')

        def synthesize(model, jobs):
            batches.append([text for text, _ in jobs])
            for text, output in jobs:
                if text not in fail:
                    Path(output).write_bytes(b'RIFF')

        return CachedWavSpeaker(model, Path(cache), synthesize=synthesize, play=played.append)

    def test_prepare_many_synthesizes_missing_texts_in_one_batch(self):
        with tempfile.TemporaryDirectory() as cache:
            batches, played = [], []
            speaker = self.speaker(cache, batches, played)
            self.assertEqual(speaker.prepare_many(['ไก่', 'มา'], 'thai'), {'ไก่', 'มา'})
            self.assertEqual(speaker.prepare_many(['ไก่', 'ดี'], 'thai'), {'ไก่', 'ดี'})
            self.assertEqual(batches, [['ไก่', 'มา'], ['ดี']])

    def test_prepared_text_is_played_from_cache(self):
        with tempfile.TemporaryDirectory() as cache:
            batches, played = [], []
            speaker = self.speaker(cache, batches, played)
            speaker.prepare_many(['ไก่'], 'thai')
            self.assertTrue(speaker.speak('ไก่', 'thai'))
            speaker.stop()
            self.assertTrue(str(played[0]).endswith('.wav'))
            self.assertIsNone(played[1])

    def test_speak_never_synthesizes_in_the_frame_loop(self):
        with tempfile.TemporaryDirectory() as cache:
            batches, played = [], []
            speaker = self.speaker(cache, batches, played)
            with self.assertLogs('finger_reader', 'WARNING'):
                self.assertFalse(speaker.speak('ไม่เคยเตรียม', 'thai'))
            self.assertEqual((batches, played), ([], []))

    def test_failed_texts_are_not_reported_ready(self):
        with tempfile.TemporaryDirectory() as cache:
            batches, played = [], []
            speaker = self.speaker(cache, batches, played, fail={'มา'})
            self.assertEqual(speaker.prepare_many(['ไก่', 'มา'], 'thai'), {'ไก่'})

    def test_blank_texts_are_never_sent_to_the_voice(self):
        # Piper writes no audio for '' and the whole batch used to fail (wave.Error).
        with tempfile.TemporaryDirectory() as cache:
            batches, played = [], []
            speaker = self.speaker(cache, batches, played)
            self.assertEqual(speaker.prepare_many(['', '  ', 'ไก่'], 'thai'), {'ไก่'})
            self.assertEqual(batches, [['ไก่']])

    def test_one_failing_text_does_not_stop_the_batch_in_the_voice_process(self):
        from finger_reader import BATCH_SCRIPT
        self.assertIn('except Exception', BATCH_SCRIPT)  # per text, inside the Piper/MMS process


if __name__ == '__main__':
    unittest.main()
