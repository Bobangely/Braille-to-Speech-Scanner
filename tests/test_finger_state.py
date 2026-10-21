import unittest

from finger_state import EnterCell, FingerState, FinishWord
from page_layout import CellSlot, LinePair, PageMap


def _line(line_id, y1, y2):
    spec = [("สระไอ", 0), ("กอไก่", 0), ("ไม้เอก", 0), (None, None), ("มอม้า", 1)]
    cells = tuple(
        CellSlot(i, 10 + 6 * i, 16 + 6 * i, frozenset({i + 1}), n, w)
        for i, (n, w) in enumerate(spec)
    )
    return LinePair(line_id, y1, y2, cells)


PAGE = PageMap((_line(0, 20, 30), _line(1, 50, 60)), ("ไก่", "ม"))


def run(fs, steps, y=25.0, hold=0.2, tick=0.05):
    """steps: x values (or None); each held for `hold` seconds. Returns events."""
    events, t = [], 0.0
    for x in steps:
        for k in range(round(hold / tick)):
            p = None if x is None else (x, y)
            events += fs.update(p, t + k * tick)
        t += hold
    return events


class FingerStateTest(unittest.TestCase):
    def setUp(self):
        self.fs = FingerState(PAGE)

    def test_reading_kai_then_space(self):
        self.assertEqual(
            run(self.fs, [13, 19, 25, 31]),
            [EnterCell(0, 0, "สระไอ"), EnterCell(0, 1, "กอไก่"),
             EnterCell(0, 2, "ไม้เอก"), FinishWord(0, "ไก่")],
        )

    def test_dwell_below_threshold_is_silent(self):
        ev = run(self.fs, [13], hold=0.1)
        ev += self.fs.update((13, 40), 0.1)
        self.assertEqual(ev, [])

    def test_backwards_speaks_symbol_without_word(self):
        self.assertEqual(
            run(self.fs, [13, 19, 13]),
            [EnterCell(0, 0, "สระไอ"), EnterCell(0, 1, "กอไก่"), EnterCell(0, 0, "สระไอ")],
        )

    def test_skipped_cells_speak_only_current(self):
        self.assertEqual(
            run(self.fs, [13, 25]), [EnterCell(0, 0, "สระไอ"), EnterCell(0, 2, "ไม้เอก")]
        )

    def test_jitter_on_boundary_speaks_once(self):
        ev = []
        for k in range(20):
            ev += self.fs.update((15.8 if k % 2 == 0 else 16.2, 25), k * 0.05)
        self.assertEqual(ev, [EnterCell(0, 0, "สระไอ")])

    def test_lost_finger_returns_outside(self):
        self.assertEqual(
            run(self.fs, [13, None, 13]),
            [EnterCell(0, 0, "สระไอ"), EnterCell(0, 0, "สระไอ")],
        )

    def test_leave_vertically_then_return_repeats_symbol_only(self):
        ev = run(self.fs, [13])
        ev += self.fs.update((13, 35), 0.3)
        for k in range(4):
            ev += self.fs.update((13, 25), 0.4 + k * 0.05)
        self.assertEqual(ev, [EnterCell(0, 0, "สระไอ"), EnterCell(0, 0, "สระไอ")])

    def test_jump_to_other_line_does_not_finish_word(self):
        ev = []
        for k in range(4):
            ev += self.fs.update((19, 25), k * 0.05)
        for k in range(4):
            ev += self.fs.update((19, 55), 0.3 + k * 0.05)
        self.assertEqual(ev, [EnterCell(0, 1, "กอไก่"), EnterCell(1, 1, "กอไก่")])

    def test_past_line_end_finishes_last_word(self):
        ev = run(self.fs, [37, 70])
        self.assertEqual(ev, [EnterCell(0, 4, "มอม้า"), FinishWord(1, "ม")])

    def test_unreadable_cell_reports_none_name(self):
        c = CellSlot(0, 0, 6, frozenset(), None, 0)
        fs = FingerState(PageMap((LinePair(0, 20, 30, (c,)),), ("?",)))
        self.assertEqual(run(fs, [3]), [EnterCell(0, 0, None)])

    def test_dropped_frame_entering_space_still_finishes_word(self):
        ev = run(self.fs, [13, 19, 25])
        ev += self.fs.update((31, 25), 0.60)
        ev += self.fs.update(None, 0.65)  # one frame without a hand
        for k in range(4):
            ev += self.fs.update((31, 25), 0.70 + k * 0.05)
        self.assertEqual(ev[-1], FinishWord(0, "ไก่"))

    def test_vertical_exit_then_land_on_space_finishes_word(self):
        ev = run(self.fs, [13, 19, 25])
        ev += self.fs.update((28, 35), 0.60)
        for k in range(4):
            ev += self.fs.update((31, 25), 0.65 + k * 0.05)
        self.assertEqual(ev[-1], FinishWord(0, "ไก่"))

    def test_jump_to_other_line_space_does_not_finish_old_word(self):
        ev = run(self.fs, [19])
        for k in range(4):
            ev += self.fs.update((31, 55), 0.3 + k * 0.05)
        self.assertEqual(ev, [EnterCell(0, 1, "กอไก่")])

    def test_past_edge_of_other_line_does_not_finish_old_word(self):
        ev = run(self.fs, [37])
        ev += self.fs.update((70, 55), 0.3)
        self.assertEqual(ev, [EnterCell(0, 4, "มอม้า")])

    def test_moving_left_onto_space_does_not_finish(self):
        self.assertEqual(run(self.fs, [37, 31]), [EnterCell(0, 4, "มอม้า")])

    def test_word_finishes_once(self):
        ev = run(self.fs, [13, 19, 25, 31, 70])
        self.assertEqual(ev.count(FinishWord(0, "ไก่")), 1)
        self.assertNotIn(FinishWord(1, "ม"), ev)  # ม was never entered


if __name__ == "__main__":
    unittest.main()
