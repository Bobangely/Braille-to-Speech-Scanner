"""Raw Thai Grade-1 cell snapshots and longest lexical/contextual rules.

Scanner subset of THAI_BRAILLE_G1_PARSER_RULES.md. Forward Liblouis tables
are references, not a claim that every reverse reading is unambiguous.
"""
from copy import deepcopy
import logging
import math
from numbers import Integral

from config_thai import (THAI_BRAILLE_TO_CHAR, THAI_READING_MULTI_CELL,
                        THAI_MULTI_CELL_SYMBOLS, THAI_DIGIT_MAP, NUMBER_INDICATOR)
from thai_renderer import CONSONANTS, TONES, LEADING, normalize_vowel

BRAILLE_STANDARD = 'TH_G1_SCANNER_V1'
RULES_SOURCE = 'THAI_BRAILLE_G1_PARSER_RULES.md'
logger = logging.getLogger(__name__)

POST_ONSET_VOWELS = {
    ('124', '1'): 'เอะ', ('126', '1'): 'แอะ', ('24', '1'): 'โอะ',
    ('135', '1'): 'เอาะ', ('146', '1'): 'เออะ',
    ('12356', '1'): 'เอียะ', ('12345', '1'): 'เอือะ', ('15', '1'): 'อัวะ',
    ('26', '135'): 'อือ',
}


def _descriptor(value):
    canonical = normalize_vowel(value)
    if value in CONSONANTS:
        kind = 'consonant'
    elif value in LEADING:
        kind = 'leading'
    elif value in TONES:
        kind = 'tone'
    elif canonical is not None:
        kind = 'vowel'
    else:
        kind = 'literal'
    return dict(kind=kind, value=value, vowel=canonical,
                semantic=kind == 'vowel' and len(value) > 1)


PATTERN_CANDIDATES = {
    ''.join(map(str, sorted(pattern))): [_descriptor(value)]
    for pattern, value in THAI_BRAILLE_TO_CHAR.items()
}
PATTERN_CANDIDATES['135'] += [
    dict(kind='vowel', value='ออ', vowel='ออ', context='AFTER_ONSET'),
    dict(kind='compound_prefix', value='เอาะ', next_pattern='1', context='AFTER_ONSET'),
]
PATTERN_CANDIDATES['3'] += [dict(kind='mark', value='ฺ', context='NEEDS_LANGUAGE_CONTEXT')]
PATTERN_CANDIDATES['1'] += [dict(kind='short_vowel_suffix', value='ะ', context='AFTER_VOWEL')]
PATTERN_CANDIDATES['156'] += [dict(kind='leading_prefix', value='ใ', next_pattern='2')]
PATTERN_CANDIDATES['236'] += [dict(kind='punctuation', value='?', prefix_pattern='456')]

LEXICAL_SEQUENCES = {
    tuple(''.join(map(str, sorted(p))) for p in sequence): _descriptor(value)
    for sequence, value in THAI_READING_MULTI_CELL.items()
}
for sequence, symbol in THAI_MULTI_CELL_SYMBOLS.items():
    key = tuple(''.join(map(str, sorted(p))) for p in sequence)
    LEXICAL_SEQUENCES[key].update(symbol_id=symbol['id'],
                                  tts=symbol['tts'], symbol_type=symbol['type'])
SYMBOL_PREFIXES = {''.join(map(str, sorted(sequence[0])))
                   for sequence in THAI_MULTI_CELL_SYMBOLS}


def match_post_onset_vowel(atoms, index):
    """Only called while an onset exists and its vowel has not been selected."""
    for sequence in sorted(POST_ONSET_VOWELS, key=len, reverse=True):
        following = atoms[index:index+len(sequence)]
        if (len(following) == len(sequence)
                and all(a.get('patterns') == [p] for a, p in zip(following, sequence))
                and all(a.get('adjacent_before', True) for a in following[1:])):
            return POST_ONSET_VOWELS[sequence], len(sequence)
    return None


def _spacing(cell):
    grid = cell.get('grid') or {}
    cols = grid.get('expected_cols')
    return abs(cols[1] - cols[0]) if cols else 20.0


def is_unread_cell(cell):
    """A proposed cell with a failed crop read is unknown, not a blank space."""
    return not cell.get('dots') and cell.get('crop_status') in ('empty', 'invalid_grid')


def cell_read_warning(cell):
    if cell.get('geometry_status') == 'invalid' or cell.get('crop_status') == 'invalid_grid':
        return 'invalid_cell_grid'
    if is_unread_cell(cell):
        return 'empty_crop'
    if cell.get('row_ambiguous'):
        return 'ambiguous_row_grid'
    return None


def _break_before(cells, index, adjacent=False):
    if index == 0:
        return False
    left, right = cells[index - 1:index + 1]
    if right.get('break_before') or right.get('line_break_before'):
        return True
    if any(not cell.get('dots') and not is_unread_cell(cell) for cell in (left, right)):
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


def snapshot_cells(cells):
    """Preserve dots and measured confidence before any interpretation."""
    prepared = []
    for cell in cells:
        if isinstance(cell, str):
            cell = dict(dots=[int(d) if d.isdigit() else d for d in cell])
        item = dict(cell)
        if 'line_id' not in item and 'line' in item:
            item['line_id'] = item['line']
        prepared.append(item)
    raw = []
    for index, cell in enumerate(prepared):
        values = list(cell.get('dots', ()))
        valid = 'dots' in cell and all(isinstance(d, Integral) and 1 <= d <= 6 for d in values)
        valid = valid and len(values) == len(set(values))
        dots = sorted(values) if valid else values
        confidence = cell.get('confidence')
        if confidence is None:
            measured = [d['confidence'] for d in cell.get('read_dots', ()) if d.get('confidence') is not None]
            confidence = min(measured) if measured else None
        if confidence is not None:
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                confidence = None
        pattern = ''.join(map(str, dots))
        raw.append(dict(cell_id=cell.get('cell_id', f"C{cell.get('track_id', index+1)}"),
                        dots=deepcopy(dots), raw_dots=deepcopy(values), pattern=pattern, confidence=confidence,
                        index=index, line_id=cell.get('line_id'), center=cell.get('center', (0, 0)),
                        break_before=_break_before(prepared, index),
                        adjacent_before=not _break_before(prepared, index, adjacent=True),
                        line_break_before=(bool(cell.get('line_break_before')) or
                                           (index > 0 and prepared[index-1].get('line_id') is not None
                                           and cell.get('line_id') is not None
                                           and prepared[index-1]['line_id'] != cell['line_id'])),
                        warning=cell_read_warning(cell) if valid else 'invalid_raw_pattern'))
    return raw


def lex_braille(cells):
    """Keep overlay labels separate from candidate-bearing parser atoms."""
    raw = snapshot_cells(cells)
    tokens, atoms = [], []
    index, numeric = 0, False
    sequences = sorted(LEXICAL_SEQUENCES, key=len, reverse=True)
    while index < len(raw):
        cell = raw[index]
        pattern = cell['pattern']
        if cell['break_before'] or not pattern:
            numeric = False
        span, char, warning = 1, '', cell['warning']
        descriptor = None
        candidates = deepcopy(PATTERN_CANDIDATES.get(pattern, []))
        rule = 'SINGLE_PATTERN'
        if warning:
            char, numeric = '�', False
        elif not pattern:
            char = ' '
        elif frozenset(cell['dots']) == NUMBER_INDICATOR:
            numeric = True
        elif numeric and frozenset(cell['dots']) in THAI_DIGIT_MAP:
            char = THAI_DIGIT_MAP[frozenset(cell['dots'])]
            rule = 'NUMBER'
        else:
            numeric = False
            for sequence in sequences:
                following = raw[index:index+len(sequence)]
                if (len(following) == len(sequence)
                        and tuple(c['pattern'] for c in following) == sequence
                        and not any(c['warning'] for c in following)
                        and all(c['adjacent_before'] for c in following[1:])):
                    descriptor = dict(LEXICAL_SEQUENCES[sequence])
                    char, span, rule = descriptor['value'], len(sequence), 'LONGEST_LEXICAL_SEQUENCE'
                    candidates = [dict(descriptor)]
                    break
            if descriptor is None:
                if candidates:
                    descriptor = dict(candidates[0])
                    char = descriptor['value']
                    if pattern == '135':
                        descriptor['kind'] = 'ambiguous_o'
                    elif pattern == '3':
                        descriptor['kind'] = 'ambiguous_mark'
                else:
                    char, warning = '�', 'unknown_or_incomplete_symbol'
        if logger.isEnabledFor(logging.DEBUG) and (
                warning in ('unknown_or_incomplete_symbol', 'invalid_raw_pattern')
                or (span == 1 and pattern in SYMBOL_PREFIXES)):
            logger.debug(
                '[UNKNOWN_BRAILLE_SEQUENCE] line=%r cell_index=%s raw_dots=%r '
                'normalized_dots=%r neighbor_cells=%r confidence=%r',
                cell['line_id'], index, cell['raw_dots'], cell['dots'],
                [dict(line=c['line_id'], pattern=c['pattern'], dots=c['dots'])
                 for c in raw[max(0, index-1):index+3]], cell['confidence'])
        if warning:
            descriptor = dict(kind='error', value='�', vowel=None, semantic=False)
        elif descriptor is None:
            descriptor = dict(kind='literal', value=char, vowel=None, semantic=False)
        source_indices = list(range(index, index+span))
        atom = dict(descriptor, index=index, source_indices=source_indices,
                    source_cells=[raw[i]['cell_id'] for i in source_indices],
                    patterns=[raw[i]['pattern'] for i in source_indices],
                    adjacent_before=cell['adjacent_before'], candidates=candidates, rule=rule)
        symbol_metadata = {key: descriptor[key] for key in ('symbol_id', 'tts', 'symbol_type')
                           if key in descriptor}
        for offset in range(span):
            origin = raw[index+offset]
            codepoint = (chr(0x2800 + sum(1 << (d-1) for d in origin['dots']))
                         if origin['warning'] != 'invalid_raw_pattern' else '')
            tokens.append(dict(dots=deepcopy(origin['dots']), pattern=origin['pattern'],
                               cell_id=origin['cell_id'], confidence=origin['confidence'],
                               char=char if offset == 0 else '', braille_unicode=codepoint, unicode=codepoint,
                               center=origin['center'], cell_start=index, cell_end=index+span-1,
                               consumed=offset > 0, break_before=cell['break_before'] if offset == 0 else False,
                               line_break_before=cell['line_break_before'] if offset == 0 else False,
                               line_id=origin['line_id'], warning=warning, **symbol_metadata))
            atoms.append(atom if offset == 0 else None)
        index += span
    return raw, tokens, atoms
