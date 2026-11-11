"""อ่านคำจากพจนานุกรมโดยเทียบเซลล์เบรลล์ทั้งคำ"""
from functools import lru_cache
import logging
import re
import unicodedata

from text_to_braille import text_to_braille_cells

logger = logging.getLogger(__name__)
_PLAIN_WORD = re.compile('[ก-ฮะ-ูเ-๎]+')
_LONGEST_PIECE = 30  # เซลล์ของคำย่อยยาวสุดที่ลองตัด คำยาวกว่านี้ค้นตรงได้อยู่แล้ว


@lru_cache(maxsize=1)
def default_lexicon():
    """โหลดรายการคำที่มากับแพ็กเกจครั้งเดียว และไม่ดึงข้อมูลตอนรันจริง"""
    try:
        from pythainlp.corpus import thai_words
        return frozenset(unicodedata.normalize('NFC', word) for word in thai_words() if word)
    except (ImportError, OSError) as exc:
        logger.warning('Thai lexicon unavailable; using the parser rules only: %s', exc)
        return frozenset()


def _is_plain_word(word):
    """ตัด ฯ จุด ตัวเลข และพินทุ เพราะไม่มีเซลล์ของตัวเอง ทำให้ชนกับคำอื่น"""
    return bool(_PLAIN_WORD.fullmatch(word))


@lru_cache(maxsize=1)
def _cell_table():
    """เซลล์เบรลล์ของทุกคำในพจนานุกรม -> คำนั้น"""
    return {tuple(frozenset(cell) for cell in text_to_braille_cells(word, 'thai')): word
            for word in default_lexicon() if _is_plain_word(word)}


def word_from_cells(cells):
    """คำจากพจนานุกรมที่มีเซลล์ตรงกันทั้งหมด หรือ None ถ้าไม่รู้จัก

    กฎอย่างเดียวรู้ขอบพยางค์ไม่แน่ชัด เช่น อ่าน ะ ที่มีพยัญชนะตามเป็น ั (พระองค์ เป็น พรัองค์)
    คำประสมที่ไม่อยู่ในพจนานุกรมจะตัดเป็นคำย่อยให้ครบทุกเซลล์ โดยใช้จำนวนคำน้อยที่สุด
    """
    table, cells = _cell_table(), tuple(cells)
    if cells in table:
        return table[cells]
    best = [None] * len(cells) + [(0, '')]  # ตำแหน่งนี้ถึงท้ายคำ: (จำนวนคำ, ข้อความ)
    for start in range(len(cells) - 1, -1, -1):
        for end in range(start + 1, min(len(cells), start + _LONGEST_PIECE) + 1):
            word = table.get(cells[start:end])
            if word and best[end] and (best[start] is None or best[end][0] + 1 < best[start][0]):
                best[start] = (best[end][0] + 1, word + best[end][1])
    return best[0][1] if best[0] else None
