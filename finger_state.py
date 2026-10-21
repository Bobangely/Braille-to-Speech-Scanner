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
        self.entered_s: float | None = None  # when the finger reached the last spoken cell
        self._pending: set[tuple[int, int]] = set()  # (line_id, word) entered, not yet finished

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
            line = loc[1]
            events += self._finish(line, self._word_left_of(line, len(line.cells)))
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

    @staticmethod
    def _word_left_of(line, index):
        """Word of the nearest non-empty cell left of ``index`` on the line."""
        return next((c.word_index for c in reversed(line.cells[:index]) if c.word_index is not None), None)

    def _finish(self, line, word):
        # Pending words are kept even when the finger drops out for a frame,
        # so a lost frame or a lift onto the space still speaks the word.
        key = (line.line_id, word)
        if word is not None and key in self._pending:
            self._pending.discard(key)
            return [FinishWord(word, self.page.words[word])]
        return []

    def _promote(self, line, cell):
        self._active, self._candidate = (line, cell), None
        # Moving to another line pair abandons the words started there.
        self._pending = {key for key in self._pending if key[0] == line.line_id}
        if cell.word_index is None:
            return self._finish(line, self._word_left_of(line, cell.index))
        self._pending.add((line.line_id, cell.word_index))
        self.entered_s = self._since
        return [EnterCell(line.line_id, cell.index, cell.name)]
