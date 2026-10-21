from dataclasses import dataclass

from page_layout import CellSlot, LinePair, PageMap

_EPS = 1e-9


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
    """Turns fingertip positions (page mm) into speech events. Pure logic."""

    def __init__(self, page: PageMap, dwell_s: float = 0.15, hysteresis_mm: float = 0.5):
        self.page = page
        self.dwell_s = dwell_s
        self.hysteresis_mm = hysteresis_mm
        self.reset()

    def reset(self) -> None:
        self._active: tuple[LinePair, CellSlot] | None = None
        self._candidate: tuple[LinePair, CellSlot] | None = None
        self._since = 0.0
        self._pending: set[int] = set()  # words entered but not yet finished

    def _locate(self, x, y):
        """Return ('cell', line, cell), ('past', line) or None."""
        band = next((l for l in self.page.lines if l.raised_y1_mm <= y <= l.raised_y2_mm), None)
        if band is None:
            return None
        for held in (self._active, self._candidate):  # hysteresis: keep current cell
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
            self._active = self._candidate = None
            return []
        loc = self._locate(*point_mm)
        events: list[EnterCell | FinishWord] = []
        if loc is None:
            self._active = self._candidate = None
        elif loc[0] == "past":
            if self._active and self._active[0] is loc[1]:
                events += self._finish(self._active[1].word_index)
            self._active = self._candidate = None
        else:
            _, line, cell = loc
            if self._active == (line, cell):
                self._candidate = None
            elif self._candidate != (line, cell):
                self._candidate, self._since = (line, cell), now_s
            if self._candidate and now_s - self._since >= self.dwell_s - _EPS:
                events += self._promote(line, cell)
        return events

    def _finish(self, word):
        if word is not None and word in self._pending:
            self._pending.discard(word)
            return [FinishWord(word, self.page.words[word])]
        return []

    def _promote(self, line, cell):
        prev, self._active, self._candidate = self._active, (line, cell), None
        if cell.word_index is None:
            if prev and prev[0] is line and cell.index > prev[1].index:
                return self._finish(prev[1].word_index)
            return []
        self._pending.add(cell.word_index)
        return [EnterCell(line.line_id, cell.index, cell.name)]
