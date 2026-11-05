"""อ่านบรรทัดเบรลล์สีของหน้าที่ปรับเทียบแล้วเป็น PageMap

หน้าถูก warp เป็นมาตราส่วนคงที่ก่อน ระยะจุดและระยะเซลล์จึงรู้เป็นมิลลิเมตร (จาก ``PageFormat``)
มีแค่ตำแหน่งบรรทัดที่มาจากภาพ เพราะ Word จัดบรรทัดสี และทีมเขียนบรรทัดนูนด้วยมือ
"""

from __future__ import annotations

import numpy as np

from colored_braille import ColoredCellStream
from decoder import decode_cells, decode_cells_verbose
from page_calibration import DEFAULT_FORMAT, PageCalibration, PageFormat, warp_page
from page_layout import CellSlot, LinePair, PageMap
from thai_lexicon import word_from_cells
from thai_speech import symbol_name

PX_PER_MM = 8.0
UNKNOWN_WORD = 'ไม่ทราบ'
_BAND_GAP_MM = 1.0  # ระยะเว้นระหว่างบรรทัดสีกับแถบนูน


def read_page(image, calibration: PageCalibration, fmt: PageFormat = DEFAULT_FORMAT) -> PageMap:
    flat = warp_page(image, calibration, fmt, px_per_mm=PX_PER_MM)
    dots, _ = ColoredCellStream().find_dots(flat)
    points = np.array([d['center'] for d in dots], dtype=float).reshape(-1, 2) / PX_PER_MM
    rows = _line_rows(points, fmt.dot_spacing_mm)
    tops = _line_tops(points, rows, fmt.dot_spacing_mm)
    lines, words = [], []
    for line_id, ((first_y, members), top) in enumerate(zip(rows, tops)):
        next_top = rows[line_id + 1][0] if line_id + 1 < len(rows) else None
        bottom = (top if top is not None else first_y) + 2 * fmt.dot_spacing_mm + _BAND_GAP_MM
        band_end = next_top - _BAND_GAP_MM if next_top is not None else bottom + fmt.raised_band_mm
        cells = _line_cells(points[members], first_y if top is None else top, fmt, words,
                            trusted=top is not None)
        lines.append(LinePair(line_id, bottom, band_end, cells))
    return PageMap(tuple(lines), tuple(words), tuple(map(tuple, points.tolist())))


def _line_tops(points: np.ndarray, rows, spacing: float) -> list[float | None]:
    """y ของแถว 1 ในแต่ละบรรทัดสี หรือ None ถ้ารู้ไม่ได้

    บรรทัดที่มีจุดครบสามแถวกำหนดขอบบนของตัวเองได้ บรรทัดที่ใช้แค่สองแถว
    (ไม่มีจุดแถว 1 หรือแถว 3) ใช้ขอบบนจากระยะของบรรทัดที่ยึดได้ครบ
    เพราะ Word พิมพ์บรรทัดด้วยระยะบรรทัดคงที่
    """
    spans = [np.ptp(points[members, 1]) for _, members in rows]
    full = [i for i, span in enumerate(spans) if span > 1.5 * spacing]
    tops: list[float | None] = [rows[i][0] if i in full else None for i in range(len(rows))]
    if len(full) < 2:
        return tops
    line_pitch = (rows[full[-1]][0] - rows[full[0]][0]) / (full[-1] - full[0])
    for i, top in enumerate(tops):
        if top is None:
            ref = min(full, key=lambda f: abs(f - i))
            expected = rows[ref][0] + (i - ref) * line_pitch
            first_y = rows[i][0]
            candidate = min((first_y, first_y - spacing), key=lambda t: abs(t - expected))
            if abs(candidate - expected) < .5 * spacing:
                tops[i] = candidate
    return tops


def _line_rows(points: np.ndarray, spacing: float) -> list[tuple[float, np.ndarray]]:
    """จัดกลุ่มจุดเป็นบรรทัดสี: (y ของจุดบนสุด, index ของจุด)"""
    if not len(points):
        return []
    order = np.argsort(points[:, 1])
    lines, current = [], [order[0]]
    for index in order[1:]:
        # บรรทัดสีหนึ่งกว้างไม่เกิน 2 ระยะจุด (แถว 1-3)
        if points[index, 1] - points[current[0], 1] < 2.5 * spacing:
            current.append(index)
        else:
            lines.append(current)
            current = [index]
    lines.append(current)
    return [(float(points[members, 1].min()), np.array(members)) for members in lines]


def _line_cells(points: np.ndarray, top: float, fmt: PageFormat, words: list[str],
                trusted: bool = True) -> tuple[CellSlot, ...]:
    """เซลล์ของบรรทัดหนึ่ง ถ้า ``trusted`` เป็น False แปลว่าแถวไม่แน่นอน จึงไม่ตั้งชื่อ"""
    pitch, spacing = fmt.cell_pitch_mm, fmt.dot_spacing_mm
    origin = _column_origin(points[:, 0], pitch, spacing)
    offset = points[:, 0] - _local_origins(points[:, 0], origin, pitch, spacing)
    index = np.round((offset - np.where(_is_right(offset, pitch, spacing), spacing, 0)) / pitch).astype(int)
    right = _is_right(offset, pitch, spacing)
    row = np.clip(np.round((points[:, 1] - top) / spacing), 0, 2).astype(int)
    patterns: dict[int, set[int]] = {}
    lefts: dict[int, list[float]] = {}
    for x, k, r, c in zip(points[:, 0], index, right, row):
        patterns.setdefault(int(k), set()).add(int(c) + 1 + 3 * int(r))
        lefts.setdefault(int(k), []).append(x - r * spacing)
    first = min(patterns)
    count = max(patterns) - first + 1
    # กรอบตามจุดที่วัดได้จริง เพราะระยะเซลล์บนกระดาษเพี้ยนสะสมตามบรรทัด เซลล์ว่างเฉลี่ยจากเซลล์ข้างเคียง
    known = sorted(lefts)
    centers = np.interp(range(first, first + count), known, [np.median(lefts[k]) for k in known]) + spacing / 2
    edges = [centers[0] - pitch / 2, *((centers[:-1] + centers[1:]) / 2), centers[-1] + pitch / 2]
    slots = []
    for k in range(count):
        slots.append(dict(index=k, x1_mm=float(edges[k]), x2_mm=float(edges[k + 1]),
                          dots=frozenset(patterns.get(first + k, ())), name=None, word_index=None))
    start = len(words)
    _decode_words(slots, words)
    if not trusted:  # ห้ามเดา: แถวเลื่อนเพียงแถวเดียวทำให้ทุกสัญลักษณ์กลายเป็นตัวอื่น
        words[start:] = [UNKNOWN_WORD] * (len(words) - start)
        for slot in slots:
            slot['name'] = None
    return tuple(CellSlot(**slot) for slot in slots)


def _column_origin(xs: np.ndarray, pitch: float, spacing: float) -> float:
    """x ของคอลัมน์ซ้ายในเซลล์ 0: ค่าที่ทำให้ทุกจุดใกล้คอลัมน์ที่สุด"""
    def error(origin):
        phase = np.mod(xs - origin, pitch)
        return np.minimum.reduce([phase, np.abs(phase - spacing), pitch - phase]).sum()
    candidates = [x - shift for x in xs for shift in (0.0, spacing)]
    origin = min(candidates, key=error)
    # เซลล์ 0 คือเซลล์ซ้ายสุดที่มีจุด
    leftmost = xs.min() - origin
    return origin + np.floor((leftmost + spacing / 2) / pitch) * pitch


def _local_origins(xs: np.ndarray, origin: float, pitch: float, spacing: float) -> np.ndarray:
    """จุดเริ่มเซลล์ 0 เมื่อมองจากจุดข้างเคียงของแต่ละจุด (ข้างละราว 2.5 เซลล์)

    Word เพิ่ม 'Expanded' เฉพาะระหว่างตัวอักษร เซลล์ในตัวอักษร 2 เซลล์จึงชิดกว่า
    และบรรทัดที่พิมพ์อาจเพี้ยนเกินครึ่งเซลล์ เมื่อไล่จากซ้ายไปขวา จุดเริ่มแต่ละจุด
    จะอยู่ไม่เกินครึ่งเซลล์จากจุดก่อนหน้า เลขเซลล์จึงตามการเพี้ยนแทนการกระโดดข้ามเซลล์
    ถ้าเสมอกัน (คอลัมน์เดี่ยว) ใช้ค่าที่ใกล้จุดก่อนหน้าที่สุด
    """
    def error(near, candidate):
        phase = np.mod(near - candidate, pitch)
        return np.minimum.reduce([phase, np.abs(phase - spacing), pitch - phase]).sum()
    result = np.empty(len(xs))
    reference = origin
    for i in np.argsort(xs):
        near = xs[np.abs(xs - xs[i]) <= 2.5 * pitch]
        candidates = [c - pitch * np.round((c - reference) / pitch) for n in near for c in (n, n - spacing)]
        reference = result[i] = min(candidates, key=lambda c: error(near, c) + 1e-3 * abs(c - reference))
    return result


def _is_right(offset: np.ndarray, pitch: float, spacing: float) -> np.ndarray:
    phase = np.mod(offset, pitch)
    return np.abs(phase - spacing) < np.minimum(phase, pitch - phase)


def _decode_words(slots: list[dict], words: list[str]) -> None:
    """เติม name และ word_index ลงใน slot โดยตัดคำที่เซลล์ว่าง"""
    groups, current = [], []
    for slot in slots:
        if slot['dots']:
            current.append(slot)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    for group in groups:
        cells = [dict(dots=slot['dots'], line_id=0, x=float(i), y=0.0, center=(float(i), 0.0))
                 for i, slot in enumerate(group)]
        known = word_from_cells(slot['dots'] for slot in group)
        unreadable = False
        for token in decode_cells_verbose(cells, 'thai'):
            if token.get('consumed'):
                continue  # เซลล์ตัวนำที่ได้ชื่อจาก token ของตัวอักษรแล้ว
            # พจนานุกรมรู้จักทั้งคำแล้ว คำเตือนว่ากฎไม่แน่ใจจึงไม่เกี่ยวกับชื่อเซลล์
            name = None if token.get('warning') and known is None else symbol_name(token['char'])
            unreadable |= name is None
            for slot in group[token['cell_start']:token['cell_end'] + 1]:
                slot['name'] = name
        for slot in group:
            slot['word_index'] = len(words)
        words.append(known or (UNKNOWN_WORD if unreadable else decode_cells(cells, 'thai')))


def new_ink_dots(read: PageMap, learned: PageMap, fmt: PageFormat = DEFAULT_FORMAT, tolerance_mm: float = 1.2) -> int:
    """จุดของ ``read`` ที่ไม่มีจุดของ ``learned`` อยู่ใกล้

    มือหรือเงาทำได้แค่บังจุด หน้าที่ถูกบังจึงไม่มีจุดแบบนี้ ส่วนหน้าอื่น
    จะมีหมึกตรงที่หน้าเดิมเป็นกระดาษเปล่า
    """
    known = _dot_points_mm(learned, fmt)
    if not len(known):
        return len(_dot_points_mm(read, fmt))
    return sum(np.min(np.hypot(*(known - point).T)) > tolerance_mm for point in _dot_points_mm(read, fmt))


def _dot_points_mm(page: PageMap, fmt: PageFormat) -> np.ndarray:
    """จุดกึ่งกลางที่วัดได้ (mm) ถ้าแผนที่ไม่มีข้อมูลนี้จะสร้างจากเซลล์"""
    if page.dots_mm:
        return np.array(page.dots_mm, dtype=float)
    spacing, pad = fmt.dot_spacing_mm, (fmt.cell_pitch_mm - fmt.dot_spacing_mm) / 2
    points = [(cell.x1_mm + pad + (dot > 3) * spacing,
               line.raised_y1_mm - _BAND_GAP_MM - 2 * spacing + (dot - 1) % 3 * spacing)
              for line in page.lines for cell in line.cells for dot in cell.dots]
    return np.array(points, dtype=float).reshape(-1, 2)


def page_signature(page_map: PageMap):
    """เนื้อหาของหน้า โดยไม่สนตำแหน่งต่ำกว่ามิลลิเมตรที่สั่นระหว่างเฟรม"""
    return tuple(tuple(c.dots for c in line.cells) for line in page_map.lines)


class PageLearner:
    """รับแผนที่หน้าเมื่ออ่านเซลล์ได้เหมือนเดิมติดกันครบ ``required`` ครั้ง"""

    def __init__(self, required: int = 3):
        self.required = required
        self._signature = None
        self._count = 0

    def update(self, page_map: PageMap) -> PageMap | None:
        signature = page_signature(page_map)
        self._count = self._count + 1 if signature == self._signature else 1
        self._signature = signature
        # ภาพเบลอหรือมีของบังอ่านได้ 0 บรรทัดซ้ำกันได้ ต้องไม่นับเป็นหน้าที่เรียนสำเร็จ
        return page_map if page_map.lines and self._count >= self.required else None
