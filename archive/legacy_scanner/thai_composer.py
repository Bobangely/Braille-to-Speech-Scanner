"""Compatibility entry points for token parsing and Thai rendering."""

from thai_parser import parse_tokens
from thai_renderer import CompositionError, compose_syllable, normalize_vowel, render_thai


def compose_tokens(tokens, *, lexicon=None, confidence_threshold=None):
    return render_thai(parse_tokens(tokens), lexicon=lexicon,
                       confidence_threshold=confidence_threshold)
