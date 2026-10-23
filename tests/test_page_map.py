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


def word_like_page(text, drift_mm=0.49):
    """A colored line laid out like Word: 'Expanded' only between characters.

    The cells inside a 2-cell letter sit drift_mm closer than the slate pitch,
    and every later cell inherits the shortfall.
    """
    import cv2
    page = render_page(FMT, px_per_mm=PX, lines=[[]])
    origin_x, origin_y = FMT.line_origins_mm[0]
    x, radius = origin_x, round(FMT.dot_diameter_mm / 2 * PX)
    for char in text:
        cells = text_to_braille_cells(char, 'thai') if char.strip() else [None]
        for index, cell in enumerate(cells):
            for dot in cell or ():
                cx = x + (dot > 3) * FMT.dot_spacing_mm
                cy = origin_y + (dot - 1) % 3 * FMT.dot_spacing_mm
                cv2.circle(page, (round(cx * PX), round(cy * PX)), radius, (255, 0, 0), -1, cv2.LINE_AA)
            x += FMT.cell_pitch_mm - (drift_mm if index < len(cells) - 1 else 0)
    frame, _ = camera_view(page, rotated_corners(0), size=(1280, 960), background=200)
    calib = calibrate(frame, FMT)
    assert calib.ok, calib.reason
    return read_page(frame, calib, FMT)


class ReadPageTests(unittest.TestCase):
    def test_two_cell_letter_names_both_of_its_cells(self):
        # The decoder adds a consumed '' token for the prefix cell; it must not replace the name.
        cells = read(page_lines(('ฃ ค',))).lines[0].cells
        self.assertEqual([c.name for c in cells], ['ขอขวด', 'ขอขวด', None, 'คอควาย'])

    def test_word_layout_drift_inside_a_line_is_followed(self):
        text = 'ก ข ฃ ค ฅ ฆ ง จ ฉ ช ซ'  # three 2-cell letters: 1.5 mm shortfall by the end
        page = word_like_page(text)
        expected = [frozenset(c or ()) for c in text_to_braille_cells(text, 'thai')]
        self.assertEqual([c.dots for c in page.lines[0].cells], expected)
        self.assertNotIn('ไม่ทราบ', page.words)

    def test_drift_beyond_half_a_cell_keeps_every_space(self):
        # Seven 2-cell letters each 0.5 mm short (a 30 pt page without Replace): 3.5 mm over the line.
        # 32.5 pt pages are ~0.04 mm short per letter; dots 2.52 mm apart fail beyond ~0.55 mm per letter.
        text = 'ฌ ญ ฎ ฏ ฐ ฑ ฒ ณ ด'
        page = word_like_page(text, drift_mm=0.5)
        self.assertEqual(page.words, tuple(text.split()))

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
