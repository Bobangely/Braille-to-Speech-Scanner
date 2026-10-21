"""Reading the colored Braille lines of a calibrated page into a PageMap."""
import unittest

import numpy as np

from generate_test import text_to_braille_cells
from page_calibration import DEFAULT_FORMAT as FMT, calibrate, render_page
from page_map import PageLearner, read_page
from tests.test_page_calibration import camera_view, rotated_corners

PX = 8.0
TEXT = ('ไก่ มา', 'ไป ดู', 'นก บิน')


def page_lines(texts=TEXT):
    return [[p or set() for p in text_to_braille_cells(t, 'thai')] for t in texts]


def read(lines, angle=0):
    frame, _ = camera_view(render_page(FMT, px_per_mm=PX, lines=lines), rotated_corners(angle),
                           size=(1280, 960), background=200)
    calib = calibrate(frame, FMT)
    assert calib.ok, calib.reason
    return read_page(frame, calib, FMT)


class ReadPageTests(unittest.TestCase):
    def assert_page(self, page, lines, words):
        self.assertEqual(len(page.lines), len(lines))
        for line, expected in zip(page.lines, lines):
            self.assertEqual([c.dots for c in line.cells], [frozenset(d) for d in expected])
        self.assertEqual(page.words, words)

    def test_synthetic_page_reads_all_cells(self):
        page = read(page_lines())
        self.assert_page(page, page_lines(), ('ไก่', 'มา', 'ไป', 'ดู', 'นก', 'บิน'))
        first = page.lines[0].cells
        self.assertEqual([c.name for c in first[:3]], ['สระไอ', 'กอไก่', 'ไม้เอก'])
        self.assertEqual([c.word_index for c in first], [0, 0, 0, None, 1, 1])
        self.assertIsNone(first[3].name)

    def test_tilted_page_reads_all_cells(self):
        self.assert_page(read(page_lines(), angle=8), page_lines(), ('ไก่', 'มา', 'ไป', 'ดู', 'นก', 'บิน'))

    def test_cells_tile_the_line_left_to_right(self):
        cells = read(page_lines()).lines[0].cells
        for left, right in zip(cells, cells[1:]):
            self.assertAlmostEqual(left.x2_mm, right.x1_mm, places=6)
        origin_x = FMT.line_origins_mm[0][0]
        self.assertLess(abs((cells[0].x1_mm + cells[0].x2_mm) / 2 - (origin_x + FMT.dot_spacing_mm / 2)), 0.5)

    def test_missing_dot_marks_cell_unreadable(self):
        lines = page_lines()
        lines[0][1] = {4}  # ก lost dots 1, 2 and 5
        page = read(lines)
        cell = page.lines[0].cells[1]
        self.assertEqual((cell.word_index, cell.name), (0, None))
        self.assertEqual(page.words[0], 'ไม่ทราบ')

    def test_raised_band_sits_between_colored_lines(self):
        page = read(page_lines())
        ys = [y for _, y in FMT.line_origins_mm][:len(page.lines)]
        bottom = [y + 2 * FMT.dot_spacing_mm for y in ys]
        for line, colored_bottom, next_top in zip(page.lines, bottom, ys[1:] + [None]):
            self.assertGreater(line.raised_y1_mm, colored_bottom)
            if next_top is None:
                self.assertAlmostEqual(line.raised_y2_mm - line.raised_y1_mm, FMT.raised_band_mm, delta=0.5)
            else:
                self.assertLess(line.raised_y2_mm, next_top)

    def test_line_without_row_one_is_anchored_by_full_lines(self):
        # Short lines can lack any dot in row 1: their rows follow the page's line spacing.
        lines = page_lines()
        lines[1] = [{2, 5}, {3, 6}, {2, 3}]
        page = read(lines)
        self.assertEqual([c.dots for c in page.lines[1].cells],
                         [frozenset({2, 5}), frozenset({3, 6}), frozenset({2, 3})])

    def test_unanchored_partial_line_is_unreadable_not_guessed(self):
        page = read([[{2, 5}, {3, 6}, {2, 3}]])
        self.assertEqual([c.name for c in page.lines[0].cells], [None, None, None])
        self.assertEqual(page.words, ('ไม่ทราบ',))

    def test_no_dots_gives_empty_page(self):
        page = read([[]])
        self.assertEqual((page.lines, page.words), ((), ()))


class AccuracyToolTests(unittest.TestCase):
    def test_accuracy_counts_matching_cells(self):
        from tools.diagnostics.page_map_accuracy import cell_accuracy
        lines = page_lines()
        lines[0][1] = {4}
        correct, total = cell_accuracy(read(lines), TEXT)
        self.assertEqual((correct, total), (13, 14))  # spaces are not cells to read

    def test_missing_line_counts_as_wrong(self):
        from tools.diagnostics.page_map_accuracy import cell_accuracy
        self.assertEqual(cell_accuracy(read(page_lines(TEXT[:2])), TEXT), (9, 14))


class PageLearnerTests(unittest.TestCase):
    def test_learner_needs_three_equal_maps(self):
        a, b = read(page_lines()), read(page_lines(('ไก่', 'มา')))
        learner = PageLearner(required=3)
        results = [learner.update(m) for m in (a, a, b, a, a, a)]
        self.assertEqual(results[:5], [None] * 5)
        self.assertIs(results[5], a)

    def test_learner_ignores_sub_millimetre_jitter(self):
        a = read(page_lines())
        shifted = read(page_lines(), angle=0.3)
        learner = PageLearner(required=2)
        self.assertIsNone(learner.update(a))
        self.assertIs(learner.update(shifted), shifted)


if __name__ == '__main__':
    unittest.main()
