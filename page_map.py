"""Read the colored Braille lines of a calibrated page into a PageMap.

The page is first warped to a fixed scale, so dot spacing and cell pitch are
known in millimetres (from ``PageFormat``). Only line positions come from the
image: Word lays the lines out, and the team writes the raised line by hand.
"""

from __future__ import annotations

import numpy as np

from colored_braille import ColoredCellStream
from decoder import decode_cells, decode_cells_verbose
from page_calibration import DEFAULT_FORMAT, PageCalibration, PageFormat, warp_page
from page_layout import CellSlot, LinePair, PageMap
from thai_speech import symbol_name

PX_PER_MM = 8.0
UNKNOWN_WORD = 'ไม่ทราบ'
_BAND_GAP_MM = 1.0  # clearance between a colored line and the raised band


def read_page(image, calibration: PageCalibration, fmt: PageFormat = DEFAULT_FORMAT,
              color: str = 'blue') -> PageMap:
    flat = warp_page(image, calibration, fmt, px_per_mm=PX_PER_MM)
    dots, _ = ColoredCellStream(color).find_dots(flat)
    points = np.array([d['center'] for d in dots], dtype=float).reshape(-1, 2) / PX_PER_MM
    rows = _line_rows(points, fmt.dot_spacing_mm)
    lines, words = [], []
    for line_id, (top, members) in enumerate(rows):
        next_top = rows[line_id + 1][0] if line_id + 1 < len(rows) else None
        bottom = top + 2 * fmt.dot_spacing_mm + _BAND_GAP_MM
        band_end = next_top - _BAND_GAP_MM if next_top is not None else bottom + fmt.raised_band_mm
        cells = _line_cells(points[members], top, fmt, words)
        lines.append(LinePair(line_id, bottom, band_end, cells))
    return PageMap(tuple(lines), tuple(words))


def _line_rows(points: np.ndarray, spacing: float) -> list[tuple[float, np.ndarray]]:
    """Group dots into colored lines: rows closer than 1.5 dot spacings join one line.

    shortcut: a line whose dots use only two adjacent rows is assumed to start at
    its upper row; revisit if real pages show lines without row-1 dots.
    """
    if not len(points):
        return []
    order = np.argsort(points[:, 1])
    lines, current = [], [order[0]]
    for index in order[1:]:
        # A colored line spans at most 2 dot spacings (rows 1-3).
        if points[index, 1] - points[current[0], 1] < 2.5 * spacing:
            current.append(index)
        else:
            lines.append(current)
            current = [index]
    lines.append(current)
    return [(float(points[members, 1].min()), np.array(members)) for members in lines]


def _line_cells(points: np.ndarray, top: float, fmt: PageFormat, words: list[str]) -> tuple[CellSlot, ...]:
    pitch, spacing = fmt.cell_pitch_mm, fmt.dot_spacing_mm
    origin = _column_origin(points[:, 0], pitch, spacing)
    offset = points[:, 0] - origin
    index = np.round((offset - np.where(_is_right(offset, pitch, spacing), spacing, 0)) / pitch).astype(int)
    right = _is_right(offset, pitch, spacing)
    row = np.clip(np.round((points[:, 1] - top) / spacing), 0, 2).astype(int)
    patterns: dict[int, set[int]] = {}
    for k, r, c in zip(index, right, row):
        patterns.setdefault(int(k), set()).add(int(c) + 1 + 3 * int(r))
    first = min(patterns)
    count = max(patterns) - first + 1
    pad = (pitch - spacing) / 2
    slots = []
    for k in range(count):
        x1 = origin + (first + k) * pitch - pad
        slots.append([k, x1, x1 + pitch, frozenset(patterns.get(first + k, ())), None, None])
    _decode_words(slots, words)
    return tuple(CellSlot(*s) for s in slots)


def _column_origin(xs: np.ndarray, pitch: float, spacing: float) -> float:
    """Left-column x of cell 0: the phase that puts every dot nearest a column."""
    def error(origin):
        phase = np.mod(xs - origin, pitch)
        return np.minimum.reduce([phase, np.abs(phase - spacing), pitch - phase]).sum()
    candidates = [x - shift for x in xs for shift in (0.0, spacing)]
    origin = min(candidates, key=error)
    # Cell 0 is the leftmost cell that holds a dot.
    leftmost = xs.min() - origin
    return origin + np.floor((leftmost + spacing / 2) / pitch) * pitch


def _is_right(offset: np.ndarray, pitch: float, spacing: float) -> np.ndarray:
    phase = np.mod(offset, pitch)
    return np.abs(phase - spacing) < np.minimum(phase, pitch - phase)


def _decode_words(slots: list[list], words: list[str]) -> None:
    """Fill name and word_index in place; words split at empty cells."""
    groups, current = [], []
    for slot in slots:
        if slot[3]:
            current.append(slot)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    for group in groups:
        cells = [dict(dots=s[3], line_id=0, x=float(i), y=0.0, center=(float(i), 0.0))
                 for i, s in enumerate(group)]
        unreadable = False
        for token in decode_cells_verbose(cells, 'thai'):
            name = None if token.get('warning') else symbol_name(token['char'])
            unreadable |= name is None
            for s in group[token['cell_start']:token['cell_end'] + 1]:
                s[4] = name
        for s in group:
            s[5] = len(words)
        words.append(UNKNOWN_WORD if unreadable else decode_cells(cells, 'thai'))


class PageLearner:
    """Accept a page map once the same cells were read ``required`` times in a row."""

    def __init__(self, required: int = 3):
        self.required = required
        self._signature = None
        self._count = 0

    def update(self, page_map: PageMap) -> PageMap | None:
        signature = tuple(tuple(c.dots for c in line.cells) for line in page_map.lines)
        self._count = self._count + 1 if signature == self._signature else 1
        self._signature = signature
        return page_map if self._count >= self.required else None
