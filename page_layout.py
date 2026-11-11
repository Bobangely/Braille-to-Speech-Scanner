from dataclasses import dataclass


@dataclass(frozen=True)
class CellSlot:
    index: int
    x1_mm: float
    x2_mm: float
    dots: frozenset[int]
    name: str | None  # None = อ่านไม่ได้
    word_index: int | None  # None = เซลล์ว่าง


@dataclass(frozen=True)
class LinePair:
    line_id: int
    raised_y1_mm: float
    raised_y2_mm: float
    cells: tuple[CellSlot, ...]


@dataclass(frozen=True)
class PageMap:
    lines: tuple[LinePair, ...]
    words: tuple[str, ...]  # words[word_index] คือข้อความของคำ
    dots_mm: tuple[tuple[float, float], ...] = ()  # จุดสีที่วัดได้ ใช้แยกหน้าที่ถูกบังออกจากหน้าใหม่
