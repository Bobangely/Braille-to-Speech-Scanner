"""Public Thai decoding API: raw Braille parser followed by Thai rendering."""

import re
import unicodedata

from thai_braille import _break_before, cell_read_warning, is_unread_cell, lex_braille
from thai_parser import parse_braille
from thai_renderer import render_thai


def tokenize_thai(cells):
    """Physical-cell labels for overlays/debug; not the parser's semantic input."""
    return lex_braille(cells)[1]


def normalize_thai(text):
    """Reorder marks without changing a recognized vowel into a guessed tone."""
    for _ in range(2):
        text = re.sub(r'([่้๊๋์])([ัิีึืุู็ํ])', r'\2\1', text)
        text = re.sub(r'([าำๅ])([่้๊๋])', r'\2\1', text)
    return unicodedata.normalize('NFC', text)


def decode_thai_details(cells, *, lexicon=None, confidence_threshold=None):
    return render_thai(parse_braille(cells), lexicon=lexicon,
                       confidence_threshold=confidence_threshold)


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
