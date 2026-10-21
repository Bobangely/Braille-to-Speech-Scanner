"""Write the Word template for two-line books (A4 portrait, cut into two A5 pages).

Front: 4 ArUco markers per A5 half, and alternating lines styled for the colored
Braille line (type here) and the raised line (left empty for the slate).
Back (print double-sided, flip on the long edge): a grey guide mark where each
raised line starts, mirrored, because the slate is written from the back.

Usage: make_book_template.py --cell-pitch-mm 6.11 --line-pitch-mm 10 --out book.docx
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Mm, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from page_calibration import CORNERS, DEFAULT_FORMAT, PageFormat, page_format  # noqa: E402
from tools.export_markers import export_markers  # noqa: E402

FONT = 'SOV_braille_only'
FONT_PT = 30
LOWERED_PT = 2  # the exact 10 mm line is shorter than 30 pt: lower the dots so row 1 is not clipped
PT_MM = 25.4 / 72
COLOR_STYLE = 'Braille สี'
RAISED_STYLE = 'บรรทัดนูน'
HALF_MM = 148.5  # half of a portrait A4: where the sheet is cut
TOP_MM = 28.0  # first colored line of each half
LEFT_MM, RIGHT_MM = 30.0, 20.0
DOT_BLUE = RGBColor(0x00, 0x50, 0xDC)  # hue inside the reader's blue range
GUIDE = '◄'


def expanded_pt(cell_pitch_mm: float, font_pt: float = FONT_PT) -> float:
    """Word 'Expanded by' spacing that makes the font's cells match the slate pitch."""
    return cell_pitch_mm / PT_MM - 0.54 * font_pt  # 0.54 em: SOV_braille_only advance width


def line_spacing_pt(line_pitch_mm: float) -> float:
    return line_pitch_mm / PT_MM


def marker_positions_mm(fmt: PageFormat, top_mm: float) -> dict[str, tuple[float, float]]:
    """Top-left corner of each marker on the A4 sheet, for the A5 half starting at top_mm."""
    corners = fmt.marker_corners_mm()
    return {name: (float(corners[mid][0][0]), float(corners[mid][0][1]) + top_mm)
            for name, mid in zip(CORNERS, fmt.marker_ids)}


def make_template(out: Path, cell_pitch_mm: float, line_pitch_mm: float,
                  fmt: PageFormat = DEFAULT_FORMAT) -> Path:
    doc = Document()
    front = doc.sections[0]
    _set_page(front, top_mm=TOP_MM)
    _add_styles(doc, cell_pitch_mm, line_pitch_mm)
    rows = 2 * fmt.line_pairs  # per A5 half: colored, raised, colored, raised, ...
    gap_mm = HALF_MM - rows * line_pitch_mm  # the second half starts one A5 page lower
    front_lines = [doc.add_paragraph() for _ in range(2 * rows - 1)]
    doc.add_section()
    front_lines.append(doc.paragraphs[-1])  # add_section left the front's break in a new paragraph
    front, back = doc.sections  # the old section object now points at the last section
    for index, paragraph in enumerate(front_lines):
        paragraph.style = doc.styles[COLOR_STYLE if index % 2 == 0 else RAISED_STYLE]
    front_lines[rows].paragraph_format.space_before = Mm(gap_mm)
    _set_page(back, top_mm=TOP_MM)
    back.header.is_linked_to_previous = False
    for index in range(2 * rows):
        paragraph = doc.add_paragraph(style=doc.styles[RAISED_STYLE])
        if index == rows:
            paragraph.paragraph_format.space_before = Mm(gap_mm)
        if index % 2 == 1:  # a raised line: mark where the slate's first cell goes
            paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            paragraph.paragraph_format.right_indent = Mm(LEFT_MM - RIGHT_MM)
            run = paragraph.add_run(GUIDE)
            run.font.size, run.font.color.rgb = Pt(10), RGBColor(0xB0, 0xB0, 0xB0)
    with tempfile.TemporaryDirectory() as temp:
        paths = dict(zip(CORNERS, export_markers(Path(temp), px=600, fmt=fmt)))
        header = front.header.paragraphs[0]
        for top in (0.0, HALF_MM):
            for name, (x, y) in marker_positions_mm(fmt, top).items():
                _float_picture(header.add_run(), paths[name], x, y, fmt.marker_size_mm)
        out.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out))
    return out


def _set_page(section, top_mm: float) -> None:
    section.page_width, section.page_height = Mm(210), Mm(297)
    section.top_margin, section.bottom_margin = Mm(top_mm), Mm(10)
    section.left_margin, section.right_margin = Mm(LEFT_MM), Mm(RIGHT_MM)
    section.header_distance = Mm(5)


def _add_styles(doc, cell_pitch_mm: float, line_pitch_mm: float) -> None:
    for name in (COLOR_STYLE, RAISED_STYLE):
        style = doc.styles.add_style(name, 1)  # paragraph style
        style.base_style = doc.styles['Normal']
        layout = style.paragraph_format
        layout.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        layout.line_spacing = Pt(line_spacing_pt(line_pitch_mm))
        layout.space_before = layout.space_after = Pt(0)
    style = doc.styles[COLOR_STYLE]
    style.font.name, style.font.size, style.font.color.rgb = FONT, Pt(FONT_PT), DOT_BLUE
    rpr = style.element.get_or_add_rPr()
    rpr.rFonts.set(qn('w:cs'), FONT)  # Thai text is complex script in Word
    rpr.append(parse_xml(f'<w:szCs {nsdecls("w")} w:val="{FONT_PT * 2}"/>'))
    rpr.append(parse_xml(f'<w:spacing {nsdecls("w")} w:val="{round(expanded_pt(cell_pitch_mm) * 20)}"/>'))
    rpr.append(parse_xml(f'<w:position {nsdecls("w")} w:val="{-LOWERED_PT * 2}"/>'))  # half-points


def _float_picture(run, path: Path, x_mm: float, y_mm: float, size_mm: float) -> None:
    """Insert a picture at a fixed page position (Word: In Front of Text, locked)."""
    run.add_picture(str(path), width=Mm(size_mm), height=Mm(size_mm))
    inline = run._r.xpath('.//wp:inline')[-1]
    emu = lambda mm: round(mm * 36000)  # noqa: E731
    anchor = parse_xml(
        f'<wp:anchor {nsdecls("wp", "a", "pic", "r")} distT="0" distB="0" distL="0" distR="0" '
        f'simplePos="0" relativeHeight="251659264" behindDoc="0" locked="1" layoutInCell="1" '
        f'allowOverlap="1"><wp:simplePos x="0" y="0"/>'
        f'<wp:positionH relativeFrom="page"><wp:posOffset>{emu(x_mm)}</wp:posOffset></wp:positionH>'
        f'<wp:positionV relativeFrom="page"><wp:posOffset>{emu(y_mm)}</wp:posOffset></wp:positionV>'
        f'</wp:anchor>')
    anchor.append(copy.deepcopy(inline.find(qn('wp:extent'))))
    anchor.append(parse_xml(f'<wp:effectExtent {nsdecls("wp")} l="0" t="0" r="0" b="0"/>'))
    anchor.append(parse_xml(f'<wp:wrapNone {nsdecls("wp")}/>'))
    for tag in ('wp:docPr', 'wp:cNvGraphicFramePr', 'a:graphic'):
        anchor.append(copy.deepcopy(inline.find(qn(tag))))
    inline.getparent().replace(inline, anchor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cell-pitch-mm', type=float, default=165 / 27, help='slate cell pitch')
    parser.add_argument('--line-pitch-mm', type=float, default=10.0, help='slate row pitch')
    parser.add_argument('--marker-size-mm', type=float,
                        help=f'printed marker side (default {DEFAULT_FORMAT.marker_size_mm:g}); '
                             'pass the same value to finger_reader.py')
    parser.add_argument('--out', type=Path, default=Path('output/book_template.docx'))
    args = parser.parse_args()
    path = make_template(args.out, args.cell_pitch_mm, args.line_pitch_mm, fmt=page_format(args.marker_size_mm))
    print(f'{path}  Expanded {expanded_pt(args.cell_pitch_mm):.2f} pt, '
          f'line spacing exactly {line_spacing_pt(args.line_pitch_mm):.2f} pt')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
