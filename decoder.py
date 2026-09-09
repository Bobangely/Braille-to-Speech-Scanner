"""Decode ordered Thai/English Braille cells while preserving reading failures."""

from config import BRAILLE_TO_CHAR
from thai_decoder import _break_before, cell_read_warning, decode_thai, normalize_thai, tokenize_thai


_CAPITAL = frozenset({6})
_NUMBER = frozenset({3, 4, 5, 6})
_LETTER_TO_DIGIT = dict(zip('abcdefghij', '1234567890'))


def dots_to_braille_unicode(dots):
    return chr(0x2800 + sum(1 << (dot - 1) for dot in set(dots) if 1 <= dot <= 6))


def normalize_thai_text(text):
    return normalize_thai(text)


def decode_cells_thai(cells):
    return decode_thai(cells)


def _english_tokens(cells):
    capitalize_next = number_mode = False
    tokens = []
    for index, cell in enumerate(cells):
        dots = frozenset(cell['dots'])
        boundary = _break_before(cells, index)
        previous = cells[index - 1] if index else {}
        line_break = (index > 0 and 'line_id' in previous and 'line_id' in cell
                      and previous['line_id'] != cell['line_id'])
        if boundary:
            capitalize_next = number_mode = False
        char, warning = '', None
        if cell_read_warning(cell):
            char, warning = '�', cell_read_warning(cell)
            capitalize_next = number_mode = False
        elif not dots:
            char = ' '
            capitalize_next = number_mode = False
        elif dots == _CAPITAL:
            capitalize_next = True
        elif dots == _NUMBER:
            number_mode = True
        else:
            char = BRAILLE_TO_CHAR.get(dots)
            if char is None:
                char = '[' + ','.join(map(str, sorted(dots))) + ']'
                warning = 'unknown_pattern'
                number_mode = False
            elif number_mode and char in _LETTER_TO_DIGIT:
                char = _LETTER_TO_DIGIT[char]
            elif number_mode and char in (',', '.'):
                pass
            else:
                number_mode = False
                if capitalize_next:
                    char = char.upper()
            capitalize_next = False
        tokens.append(dict(dots=sorted(dots), char=char,
                           braille_unicode=dots_to_braille_unicode(dots),
                           center=cell.get('center', (0, 0)), warning=warning,
                           break_before=boundary, line_break_before=line_break))
    return tokens


def decode_cells_english(cells):
    output = []
    for token in _english_tokens(cells):
        if token['line_break_before'] and output:
            while output and output[-1] == ' ':
                output.pop()
            output.append('\n')
        elif token['break_before'] and output and output[-1] not in (' ', '\n'):
            output.append(' ')
        char = token['char']
        if char and not (char == ' ' and (not output or output[-1] in (' ', '\n'))):
            output.append(char)
    return ''.join(output).strip()


def decode_cells(cells, lang='english'):
    language = lang.lower()
    if language in ('thai', 'th'):
        return decode_thai(cells)
    if language == 'thai-legacy':
        # Historical image fixtures have different mappings. Load that decoder
        # only for an explicit legacy request, never in the live Thai pipeline.
        from archive.legacy_decoder import decode_cells_thai_legacy
        return decode_cells_thai_legacy(cells)
    if language in ('english', 'en'):
        return decode_cells_english(cells)
    raise ValueError(f'Unsupported Braille language: {lang}')


def decode_cells_verbose(cells, lang='english'):
    language = lang.lower()
    if language in ('thai', 'th'):
        return tokenize_thai(cells)
    if language == 'thai-legacy':
        from archive.legacy_decoder import decode_cells_verbose as legacy_verbose
        return legacy_verbose(cells, lang)
    if language in ('english', 'en'):
        return _english_tokens(cells)
    raise ValueError(f'Unsupported Braille language: {lang}')
