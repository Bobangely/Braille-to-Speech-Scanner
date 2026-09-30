"""Standard Thai Braille: adjacent-cell tokenization, then orthographic assembly."""

import re
import unicodedata
from thai_composer import compose_tokens

from config_thai import (THAI_BRAILLE_TO_CHAR, THAI_READING_MULTI_CELL,
                        THAI_DIGIT_MAP, NUMBER_INDICATOR)


def _spacing(cell):
    grid = cell.get('grid') or {}
    cols = grid.get('expected_cols')
    return abs(cols[1] - cols[0]) if cols else 20.0


def is_unread_cell(cell):
    """A proposed cell with a failed crop read is unknown, not a blank space."""
    return not cell['dots'] and cell.get('crop_status') == 'empty'


def cell_read_warning(cell):
    if is_unread_cell(cell):
        return 'empty_crop'
    if cell.get('row_ambiguous'):
        return 'ambiguous_row_grid'
    return None


def _break_before(cells, index, adjacent=False):
    if index == 0:
        return False
    left, right = cells[index - 1:index + 1]
    if any(not cell['dots'] and not is_unread_cell(cell) for cell in (left, right)):
        return True
    if 'line_id' in left and 'line_id' in right and left['line_id'] != right['line_id']:
        return True
    if all('reading_x' in cell and 'cell_pitch' in cell for cell in (left, right)):
        dx = right['reading_x'] - left['reading_x']
        if dx > min(left.get('adjacency_limit', float('inf')), right.get('adjacency_limit', float('inf'))):
            return True
        return dx <= 0 or dx > 1.5 * max(left['cell_pitch'], right['cell_pitch'])
    spacing = max(1.0, _spacing(right))
    if abs(right.get('y', 0) - left.get('y', 0)) > spacing * 1.5:
        return True
    if 'x' not in left or 'x' not in right:
        return False  # Ordered logical cells need no image coordinates.
    dx = right.get('x', 0) - left.get('x', 0)
    return dx < 0 or dx > spacing * (3.4 if adjacent else 6.5)


def tokenize_thai(cells):
    """One record per physical cell, with shared spans for multi-cell symbols.

    Prefixes cannot consume a cell across a line, blank cell, or large gap.
    Complete adjacent multi-cell symbols take precedence over single-cell signs,
    including prefix 356 versus karan. Without a word lexicon, an unseparated
    karan followed by ท/ข cannot be distinguished from ธ/ฃ by dots alone.
    Continuations have an empty display label and consumed=True, preserving the
    existing per-cell overlay API while showing a two-cell letter only once.
    """
    result, index, numeric = [], 0, False
    while index < len(cells):
        cell = cells[index]
        pattern = frozenset(cell['dots'])
        boundary = _break_before(cells, index)
        line_break = (index > 0 and 'line_id' in cell
                      and 'line_id' in cells[index - 1]
                      and cell['line_id'] != cells[index - 1]['line_id'])
        if boundary or not pattern:
            numeric = False
        span, char, warning = 1, '', None
        if cell_read_warning(cell):
            char, warning, numeric = '�', cell_read_warning(cell), False
        elif not pattern:
            char = ' '
        elif pattern == NUMBER_INDICATOR:
            numeric = True
        elif numeric and pattern in THAI_DIGIT_MAP:
            char = THAI_DIGIT_MAP[pattern]
        else:
            numeric = False
            if (index + 1 < len(cells) and not cell_read_warning(cells[index + 1])
                    and not _break_before(cells, index + 1, adjacent=True)):
                char = THAI_READING_MULTI_CELL.get((pattern, frozenset(cells[index + 1]['dots'])), '')
                if char:
                    span = 2
                    # Match pair_markers' complete-pair precedence. Being inside
                    # a word alone is not evidence of a failed cell recognition.
            if not char:
                char = THAI_BRAILLE_TO_CHAR.get(pattern, '�')
                if char == '�':
                    warning = 'unknown_or_incomplete_symbol'
        for offset in range(span):
            source = cells[index + offset]
            codepoint = chr(0x2800 + sum(1 << (d - 1) for d in source['dots'] if 1 <= d <= 6))
            result.append(dict(dots=sorted(source['dots']), char=char if offset == 0 else '',
                               braille_unicode=codepoint, unicode=codepoint,
                               center=source.get('center', (0, 0)),
                               cell_start=index, cell_end=index + span - 1,
                               consumed=offset > 0, break_before=boundary if offset == 0 else False,
                               line_break_before=line_break if offset == 0 else False,
                               line_id=source.get('line_id'),
                               warning=warning))
        index += span
    return result


def normalize_thai(text):
    """Reorder marks without changing a recognized vowel into a guessed tone."""
    for _ in range(2):
        text = re.sub(r'([่้๊๋์])([ัิีึืุู็ํ])', r'\2\1', text)
        text = re.sub(r'([าำๅ])([่้๊๋])', r'\2\1', text)
    return unicodedata.normalize('NFC', text)


def decode_thai_details(cells):
    """Keep physical-cell tokens and syllable decisions alongside final text."""
    return compose_tokens(tokenize_thai(cells))


def decode_thai(cells):
    return decode_thai_details(cells)['text']


def thai_verbose(cells):
    """Preserve overlay labels/spans; attach composition decisions for diagnostics."""
    result = decode_thai_details(cells)
    tokens = result['raw_tokens']
    for syllable in result['syllables']:
        token = tokens[syllable['indices'][0]]
        token['composition'] = syllable
        if syllable['needs_review'] and not token.get('warning'):
            token['warning'] = syllable['status']
    return tokens
