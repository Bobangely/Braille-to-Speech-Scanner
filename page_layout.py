from dataclasses import dataclass


@dataclass(frozen=True)
class CellSlot:
    index: int
    x1_mm: float
    x2_mm: float
    dots: frozenset[int]
    name: str | None  # None = unreadable
    word_index: int | None  # None = empty cell


@dataclass(frozen=True)
class LinePair:
    line_id: int
    raised_y1_mm: float
    raised_y2_mm: float
    cells: tuple[CellSlot, ...]


@dataclass(frozen=True)
class PageMap:
    lines: tuple[LinePair, ...]
    words: tuple[str, ...]  # words[word_index] is the word text
    dots_mm: tuple[tuple[float, float], ...] = ()  # measured colored dots, to tell a covered page from a new one
