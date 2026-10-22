"""Public Thai decoding API: raw Braille parser followed by Thai rendering."""

import re
import unicodedata
import logging

from thai_braille import _break_before, cell_read_warning, is_unread_cell, lex_braille
from thai_parser import parse_braille
from thai_renderer import render_thai

logger = logging.getLogger(__name__)


def tokenize_thai(cells):
    """Physical-cell labels for overlays/debug; not the parser's semantic input."""
    return lex_braille(cells)[1]


def normalize_thai(text):
    """Reorder marks without changing a recognized vowel into a guessed tone."""
    for _ in range(2):
        text = re.sub(r'([่้๊๋์])([ัิีึืุู็ํ])', r'\2\1', text)
        text = re.sub(r'([าำๅ])([่้๊๋])', r'\2\1', text)
    return unicodedata.normalize('NFC', text)


def decode_thai_details(cells, *, lexicon=None, confidence_threshold=None, compatibility=False):
    result = render_thai(parse_braille(cells, compatibility=compatibility), lexicon=lexicon,
                         confidence_threshold=confidence_threshold)
    if result['needs_review'] and logger.isEnabledFor(logging.DEBUG):
        logger.debug('[THAI_COMPOSITION_REVIEW] raw_cells=%r decoded_cells=%r '
                     'syllables=%r composed_output=%r', result['raw_cells'],
                     result['raw_tokens'], result['syllables'], result['text'])
    return result


def decode_thai(cells, *, compatibility=False):
    return decode_thai_details(cells, compatibility=compatibility)['text']


def thai_verbose(cells, *, compatibility=False):
    """Preserve overlay labels/spans; attach composition decisions for diagnostics."""
    result = decode_thai_details(cells, compatibility=compatibility)
    tokens = result['raw_tokens']
    for syllable in result['syllables']:
        token = tokens[syllable['indices'][0]]
        token['composition'] = syllable
        if syllable['needs_review']:
            # Text segmentation cannot invalidate an already known cell label.
            # Keep its review channel separate from detection/mapping errors.
            if (syllable['status'] == syllable.get('boundary_status')
                    and syllable.get('boundary_indices')):
                token['composition_warning'] = syllable['status']
                tokens[syllable['boundary_indices'][0]]['segmentation'] = syllable
            elif not token.get('warning'):
                token['warning'] = syllable['status']
    return tokens
