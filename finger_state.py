from dataclasses import dataclass

from page_layout import CellSlot, LinePair, PageMap

_EPS = 1e-9
LINE_DWELL_S = 0.4  # มือที่เฉียดบรรทัดถัดไปต้องไม่เริ่มอ่านที่บรรทัดนั้น
LIFT_S = 1.0  # นิ้วหายหรือออกนอกแถบบรรทัดนานเท่านี้ถือว่ายกออก กลับมาเซลล์เดิมจะพูดอีกครั้ง


@dataclass(frozen=True)
class EnterCell:
    line_id: int
    cell_index: int
    name: str | None


@dataclass(frozen=True)
class FinishWord:
    word_index: int
    text: str


class FingerState:
    """แปลงตำแหน่งปลายนิ้ว (mm บนหน้า) เป็นเหตุการณ์เสียง เป็นตรรกะล้วนไม่แตะกล้องหรือเสียง"""

    def __init__(self, page: PageMap, dwell_s: float = 0.3, hysteresis_mm: float = 0.5):
        self.page = page
        self.dwell_s = dwell_s
        self.hysteresis_mm = hysteresis_mm
        self._word_cells: dict[tuple[int, int], set[int]] = {}
        for line in page.lines:
            for c in line.cells:
                if c.word_index is not None:
                    self._word_cells.setdefault((line.line_id, c.word_index), set()).add(c.index)
        self.reset()

    def reset(self) -> None:
        self._active: tuple[LinePair, CellSlot] | None = None
        self._candidate: tuple[LinePair, CellSlot] | None = None
        self._since = 0.0
        self._held: tuple[LinePair, CellSlot] | None = None  # เซลล์ที่พูดล่าสุด คงไว้ตอนจุดหลุดสั้นๆ กันพูดซ้ำ
        self._away_since: float | None = None  # เวลาที่นิ้วเริ่มไม่อยู่บนเซลล์ใด
        self.entered_s: float | None = None  # เวลาที่นิ้วถึงเซลล์ที่พูดล่าสุด
        self._pending: dict[tuple[int, int], set[int]] = {}  # (line_id, word) ที่เข้าแล้ว -> เซลล์ที่อ่านไปแล้ว
        self._line_id: int | None = None  # บรรทัดของเซลล์ล่าสุดที่อ่าน เก็บไว้ตอนยกนิ้ว

    @property
    def active_cell(self) -> tuple[LinePair, CellSlot] | None:
        """เซลล์ที่พูดชื่อล่าสุด ตราบที่นิ้วยังอยู่บนเซลล์นั้น"""
        return self._active

    def _locate(self, x, y):
        """คืน ('cell', line, cell), ('past', line) หรือ None"""
        band = next((l for l in self.page.lines if l.raised_y1_mm <= y <= l.raised_y2_mm), None)
        if band is None:
            return None
        for held in (self._held, self._candidate):  # hysteresis: อยู่เซลล์เดิมไว้ก่อน กันเสียงซ้ำเมื่อนิ้วสั่นตรงขอบ
            if held and held[0] is band:
                c = held[1]
                if c.x1_mm - self.hysteresis_mm <= x < c.x2_mm + self.hysteresis_mm:
                    return ("cell", band, c)
        for c in band.cells:
            if c.x1_mm <= x < c.x2_mm:
                return ("cell", band, c)
        if band.cells and x >= band.cells[-1].x2_mm:
            return ("past", band)
        return None

    def update(self, point_mm, now_s):
        if point_mm is None:
            self._leave(now_s)
            return []
        loc = self._locate(*point_mm)
        events: list[EnterCell | FinishWord] = []
        if loc is None:
            self._leave(now_s)
        elif loc[0] == "past":
            line = loc[1]
            events += self._finish(line, self._word_left_of(line, len(line.cells)))
            self._active = self._candidate = self._held = None
        else:
            _, line, cell = loc
            self._away_since = None
            if self._held == (line, cell):
                self._active, self._candidate = self._held, None
            elif self._candidate != (line, cell):
                self._candidate, self._since = (line, cell), now_s
            dwell = self.dwell_s if self._line_id in (None, line.line_id) else LINE_DWELL_S
            if self._candidate and now_s - self._since >= dwell - _EPS:
                events += self._promote(line, cell)
        return events

    def _leave(self, now_s):
        """นิ้วไม่อยู่บนเซลล์ใด: จำเซลล์ที่พูดไว้จนกว่าจะหายนานถึง ``LIFT_S``"""
        self._active = self._candidate = None
        if self._away_since is None:
            self._away_since = now_s
        if now_s - self._away_since >= LIFT_S - _EPS:
            self._held = None

    @staticmethod
    def _word_left_of(line, index):
        """คำของเซลล์ที่มีจุดซึ่งอยู่ใกล้ที่สุดทางซ้ายของ ``index`` ในบรรทัด"""
        return next((c.word_index for c in reversed(line.cells[:index]) if c.word_index is not None), None)

    def _finish(self, line, word):
        # คำที่ค้างอยู่ยังเก็บไว้แม้นิ้วหายไปหนึ่งเฟรม
        # เฟรมที่หายหรือการยกนิ้วลงบนช่องว่างจึงยังพูดคำได้
        # พูดเฉพาะคำที่อ่านจนจบ การยกหรือกระโดดกลางคำต้องไม่เฉลยคำตอบ
        key = (line.line_id, word)
        if word is not None and key in self._pending:
            read = self._pending.pop(key)
            if read >= self._word_cells[key]:
                return [FinishWord(word, self.page.words[word])]
        return []

    def _promote(self, line, cell):
        self._active, self._candidate = (line, cell), None
        self._held = self._active
        self._line_id = line.line_id
        # ย้ายไปบรรทัดอื่นจะทิ้งคำที่เริ่มอ่านค้างไว้
        self._pending = {key: read for key, read in self._pending.items() if key[0] == line.line_id}
        if cell.word_index is None:
            return self._finish(line, self._word_left_of(line, cell.index))
        events = []
        # เลื่อนผ่านช่องว่างเร็วเกินกว่าจะหยุด: คำก่อนหน้าถือว่าจบ (ถ้าอ่านครบ)
        for key in [k for k in self._pending if k[1] != cell.word_index]:
            events += self._finish(line, key[1])
        self._pending.setdefault((line.line_id, cell.word_index), set()).add(cell.index)
        self.entered_s = self._since
        return events + [EnterCell(line.line_id, cell.index, cell.name)]
