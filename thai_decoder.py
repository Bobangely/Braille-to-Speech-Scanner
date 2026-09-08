"""Standard Thai Braille: adjacent-cell tokenization, then orthographic assembly."""

import re
import unicodedata

from config_thai import (THAI_BRAILLE_TO_CHAR, THAI_MULTI_CELL, THAI_DIGIT_MAP,
                        THAI_CONSONANTS, THAI_TONE_MARKS, NUMBER_INDICATOR,
                        COMPOUND_VOWELS, LEADING_VOWELS)

CONSONANTS = set(THAI_CONSONANTS.values()) | {
    value for value in THAI_MULTI_CELL.values() if len(value) == 1 and 'ก' <= value <= 'ฮ'
}
TONES = set(THAI_TONE_MARKS.values())
MARKS = 'ัิีึืุู็ํ'


def _spacing(cell):
    grid = cell.get('grid') or {}
    cols = grid.get('expected_cols')
    return abs(cols[1] - cols[0]) if cols else 20.0


def _break_before(cells, index, adjacent=False):
    if index == 0:
        return False
    left, right = cells[index - 1:index + 1]
    if not left['dots'] or not right['dots']:
        return True
    if 'line_id' in left and 'line_id' in right and left['line_id'] != right['line_id']:
        return True
    if all('reading_x' in cell and 'cell_pitch' in cell for cell in (left, right)):
        dx = right['reading_x'] - left['reading_x']
        return dx <= 0 or dx > 1.5 * max(left['cell_pitch'], right['cell_pitch'])
    spacing = max(1.0, _spacing(right))
    if abs(right.get('y', 0) - left.get('y', 0)) > spacing * 1.5:
        return True
    dx = right.get('x', 0) - left.get('x', 0)
    return dx < 0 or dx > spacing * (3.4 if adjacent else 6.5)


def tokenize_thai(cells):
    """One record per physical cell, with shared spans for multi-cell symbols.

    Prefixes cannot consume a cell across a line, blank cell, or large gap.
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
        if not pattern:
            char = ' '
        elif pattern == NUMBER_INDICATOR:
            numeric = True
        elif numeric and pattern in THAI_DIGIT_MAP:
            char = THAI_DIGIT_MAP[pattern]
        else:
            numeric = False
            if index + 1 < len(cells) and not _break_before(cells, index + 1, adjacent=True):
                char = THAI_MULTI_CELL.get((pattern, frozenset(cells[index + 1]['dots'])), '')
                if char:
                    span = 2
                    if pattern == frozenset({3, 5, 6}) and index > 0 and not boundary:
                        warning = 'ambiguous_prefix_or_karan'
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


def _onset_index(output):
    index = len(output) - 1
    while index >= 0 and output[index] in TONES:
        index -= 1
    if index < 0 or output[index] not in CONSONANTS:
        return None
    if index > 0:
        previous, current = output[index - 1:index + 1]
        if ((previous == 'ห' and current in 'งญนมยรลว') or (previous == 'อ' and current == 'ย')
                or (previous in 'กขคตปผพทศส' and current in 'รลว')):
            index -= 1
    return index


def decode_thai(cells):
    output = []
    for token in tokenize_thai(cells):
        if token['consumed']:
            continue
        if token['line_break_before'] and output:
            while output and output[-1] == ' ':
                output.pop()
            output.append('\n')
        elif token['break_before'] and output and output[-1] not in ' \n':
            output.append(' ')
        char = token['char']
        if not char:
            continue
        if char in COMPOUND_VOWELS:
            onset = _onset_index(output)
            if onset is None:
                # Preserve uncertainty instead of inventing a missing consonant.
                output.append('�')
                continue
            if char.startswith('เ') and (onset == 0 or output[onset - 1] != 'เ'):
                output.insert(onset, 'เ')
            output.extend({'เ◌า': 'า', 'เ◌อ': 'อ', 'เ◌ีย': 'ีย',
                           'เ◌ือ': 'ือ', '◌ัว': 'ัว'}[char])
        elif char in TONES:
            trailing = bool(output) and output[-1] in 'าำะ'
            if len(output) > 1 and output[-1] in 'อยว':
                trailing = output[-2] in MARKS or (
                    output[-1] == 'อ' and len(output) >= 3 and output[-3] == 'เ')
            if trailing:
                output.insert(len(output) - 1, char)
            else:
                output.append(char)
        else:
            output.extend(char)
    return normalize_thai(''.join(output).strip())
