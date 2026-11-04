"""The tool that compares finger logs of reading and of lifting the hand."""
import unittest
from types import SimpleNamespace as P

from fingertip import hand_size_px
from tools.diagnostics.finger_lift import best_cut, events


def row(t, x, size=100.0, entered=''):
    return dict(t=t, x=x, y=10.0, size=size, entered=entered)


class EventTests(unittest.TestCase):
    def test_first_cell_after_the_finger_lands_is_not_counted(self):
        rows = [row(i * .05, 10 + i, entered='ก' if i in (5, 12) else '') for i in range(20)]
        self.assertEqual(len(events(rows)), 1)

    def test_a_gap_starts_a_new_landing(self):
        first = [row(i * .05, 10, entered='ก' if i == 2 else '') for i in range(4)]
        gone = [dict(t=.2 + i * .1, x=None, y=None, size=None, entered='') for i in range(5)]
        second = [row(1 + i * .05, 10, entered='ก' if i == 2 else '') for i in range(4)]
        self.assertEqual(events(first + gone + second), [])

    def test_speed_and_growth(self):
        # 20 mm/s แล้วมือใหญ่ขึ้น 1.5 เท่าที่เซลล์ที่สอง
        rows = [row(i * .05, i * 1.0, size=150.0 if i == 30 else 100.0, entered='ก' if i in (5, 30) else '')
                for i in range(31)]
        (speed, growth), = events(rows)
        self.assertAlmostEqual(speed, 20.0, places=3)
        self.assertAlmostEqual(growth, 1.5)


class CutTests(unittest.TestCase):
    def test_clean_split_blocks_every_lift_and_loses_no_reading(self):
        self.assertEqual(best_cut([10.0, 12.0, 14.0], [40.0, 50.0]), (40.0, 0.0, 1.0))

    def test_overlap_finds_no_cut_that_keeps_the_reading(self):
        self.assertEqual(best_cut([1.0, 2.0], [1.0, 2.0])[2], 0.0)


class HandSizeTests(unittest.TestCase):
    def test_measures_the_hand_nearest_the_reading_tip(self):
        def hand(x, length):
            points = [P(x=x, y=0.0) for _ in range(21)]
            points[9] = P(x=x + length, y=0.0)
            return points
        self.assertAlmostEqual(hand_size_px([hand(.1, .1), hand(.8, .2)], (800, 0), 1000, 1000), 200.0)

    def test_no_tip_or_hand(self):
        self.assertIsNone(hand_size_px([], (1, 1), 10, 10))
        self.assertIsNone(hand_size_px([[]], None, 10, 10))


if __name__ == '__main__':
    unittest.main()
