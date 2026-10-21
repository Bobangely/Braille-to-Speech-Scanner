"""Word template for two-line books: A4 portrait cut into two A5 landscape pages."""
import tempfile
import unittest
from pathlib import Path

from docx import Document
from docx.shared import Mm

from page_calibration import DEFAULT_FORMAT as FMT
from tools.make_book_template import (
    COLOR_STYLE, RAISED_STYLE, expanded_pt, line_spacing_pt, make_template, marker_positions_mm,
)

EMU_PER_MM = 36000


def anchors(part):
    """(x_mm, y_mm, size_mm) of every floating picture in a header part."""
    ns = {'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
    result = []
    for anchor in part._element.findall('.//wp:anchor', ns):
        x = int(anchor.find('wp:positionH/wp:posOffset', ns).text) / EMU_PER_MM
        y = int(anchor.find('wp:positionV/wp:posOffset', ns).text) / EMU_PER_MM
        size = int(anchor.find('wp:extent', ns).get('cx')) / EMU_PER_MM
        result.append((round(x, 2), round(y, 2), round(size, 2)))
    return sorted(result)


class BookTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = make_template(Path(cls.temp.name) / 'book.docx', cell_pitch_mm=FMT.cell_pitch_mm, line_pitch_mm=10)
        cls.doc = Document(str(cls.path))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_expanded_spacing_formula(self):
        self.assertAlmostEqual(expanded_pt(6.2), 1.37, delta=0.01)
        self.assertAlmostEqual(expanded_pt(165 / 27), 1.12, delta=0.01)

    def test_line_spacing_is_exact_slate_row_pitch(self):
        self.assertAlmostEqual(line_spacing_pt(10), 28.35, delta=0.01)

    def test_marker_positions_mm(self):
        top = {'TL': (10, 10), 'TR': (185, 10), 'BR': (185, 123), 'BL': (10, 123)}
        self.assertEqual(marker_positions_mm(FMT, 0.0), top)
        self.assertEqual(marker_positions_mm(FMT, 148.5)['BR'], (185, 271.5))

    def test_front_header_floats_eight_markers_at_their_positions(self):
        expected = sorted((float(x), float(y), 15.0)
                          for half in (0.0, 148.5) for x, y in marker_positions_mm(FMT, half).values())
        self.assertEqual(anchors(self.doc.sections[0].header), expected)

    def test_back_page_has_no_markers(self):
        back = self.doc.sections[1]
        self.assertFalse(back.header.is_linked_to_previous)
        self.assertEqual(anchors(back.header), [])

    def test_page_is_a4_portrait(self):
        for section in self.doc.sections:
            self.assertEqual((round(section.page_width.mm), round(section.page_height.mm)), (210, 297))

    def test_front_alternates_colored_and_raised_lines_on_both_halves(self):
        styles = [p.style.name for p in self.doc.sections[0].iter_inner_content() if hasattr(p, 'style')]
        self.assertEqual(styles, [COLOR_STYLE, RAISED_STYLE] * (2 * FMT.line_pairs))

    def test_colored_style_uses_braille_font_and_slate_spacing(self):
        style = self.doc.styles[COLOR_STYLE]
        self.assertEqual(style.font.name, 'SOV_braille_only')
        self.assertEqual(style.font.size.pt, 30)
        spacing = style.element.rPr.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}spacing')
        self.assertEqual(int(spacing.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')), 28)  # 1.4 pt
        self.assertAlmostEqual(style.paragraph_format.line_spacing.pt, 28.35, delta=0.05)

    def test_colored_style_is_lowered_into_the_exact_line(self):
        # 30 pt dots are taller than a 10 mm exact line; Word clips row 1 unless lowered.
        position = self.doc.styles[COLOR_STYLE].element.rPr.find(
            '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}position')
        self.assertEqual(int(position.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')), -4)

    def test_marker_size_is_configurable(self):
        from dataclasses import replace
        fmt = replace(FMT, marker_size_mm=12.0)
        with tempfile.TemporaryDirectory() as temp:
            doc = Document(str(make_template(Path(temp) / 'small.docx', 165 / 27, 10, fmt=fmt)))
        placed = anchors(doc.sections[0].header)
        self.assertEqual({size for _, _, size in placed}, {12.0})
        self.assertIn((188.0, 10.0, 12.0), placed)  # TR: 210 - 10 margin - 12

    def test_back_guides_are_mirrored(self):
        # The slate is written from the back, right to left: the guide sits on the mirrored side.
        guides = [p for p in self.doc.sections[1].iter_inner_content() if p.text.strip()]
        self.assertEqual(len(guides), 2 * FMT.line_pairs)
        front_left = self.doc.sections[0].left_margin
        for guide in guides:
            self.assertEqual(guide.alignment, 2)  # right
            self.assertAlmostEqual(guide.paragraph_format.right_indent.mm + self.doc.sections[1].right_margin.mm,
                                   front_left.mm, delta=0.1)


class PitchTestPageTests(unittest.TestCase):
    """Lay the slate over the page: the line whose 24 full cells sit in the windows wins."""

    def test_one_full_cell_line_per_expanded_value(self):
        from tools.make_book_template import FULL_CELL, make_pitch_test
        values = (0.6, 0.8, 1.0, 1.1, 1.2, 1.4)
        with tempfile.TemporaryDirectory() as temp:
            doc = Document(str(make_pitch_test(Path(temp) / 'pitch.docx', values)))
        w = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        lines = [p for p in doc.paragraphs if FULL_CELL in p.text]
        self.assertEqual(len(lines), len(values))
        for paragraph, value in zip(lines, values):
            cells = paragraph.runs[0]
            self.assertEqual(cells.text, FULL_CELL * 24)
            self.assertEqual(paragraph.style.name, COLOR_STYLE)
            self.assertEqual(int(cells._r.rPr.find(w + 'spacing').get(w + 'val')), round(value * 20))
            self.assertIn(f'{value:.1f} pt', paragraph.text)  # label after the cells

    def test_full_cell_is_the_six_dot_glyph(self):
        from tools.make_book_template import FULL_CELL
        self.assertEqual(FULL_CELL, 'ฮ')  # SOV_braille_only draws ฮ as all six dots, one cell wide


if __name__ == '__main__':
    unittest.main()
