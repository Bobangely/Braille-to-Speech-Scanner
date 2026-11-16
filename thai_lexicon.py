"""อ่านคำจากพจนานุกรมโดยเทียบเซลล์เบรลล์ทั้งคำ"""
from functools import lru_cache
import logging
import re
import unicodedata

from text_to_braille import text_to_braille_cells

logger = logging.getLogger(__name__)
_PLAIN_WORD = re.compile('[ก-ฮะ-ูเ-๎]+')
_LONGEST_PIECE = 30  #คำย่อยยาวไม่เกิน 30 


@lru_cache(maxsize=1)
def default_lexicon():
    try:
        from pythainlp.corpus import thai_words
        return frozenset(unicodedata.normalize('NFC', word) for word in thai_words() if word)
    except (ImportError, OSError) as exc:
        logger.warning('Thai lexicon unavailable; using the parser rules only: %s', exc)
        return frozenset()


def _is_plain_word(word): #ตัด ฯ จุด ตัวเลข และพินทุ
    return bool(_PLAIN_WORD.fullmatch(word))


@lru_cache(maxsize=1)
def _cell_table():#ตารางที่ใช้เซลล์เบรลล์ของคำ key
    return {tuple(frozenset(cell) for cell in text_to_braille_cells(word, 'thai')): word
            for word in default_lexicon() if _is_plain_word(word)}


def word_from_cells(cells):
    """คำจากพจนานุกรมที่มีเซลล์ตรงกันทั้งหมด หรือ None ถ้าไม่รู้จัก

    กฎอย่างเดียวรู้ขอบพยางค์ไม่แน่ชัด เช่น อ่าน ะ ที่มีพยัญชนะตามเป็น  ั (พระองค์ เป็น พรัองค์)
    คำประสมที่ไม่อยู่ในพจนานุกรมจะตัดเป็นคำย่อยให้ครบทุกเซลล์ """
    table, cells = _cell_table(), tuple(cells)
    if cells in table:
        return table[cells]
    best = [None] * len(cells) + [(0, '')]  # (จำนวนคำ, ข้อความ)
    for start in range(len(cells) - 1, -1, -1): # ไล่จากท้ายมาหน้า
        for end in range(start + 1, min(len(cells), start + _LONGEST_PIECE) + 1):
            word = table.get(cells[start:end])
            if word and best[end] and (best[start] is None or best[end][0] + 1 < best[start][0]):
                best[start] = (best[end][0] + 1, word + best[end][1])
    return best[0][1] if best[0] else None
