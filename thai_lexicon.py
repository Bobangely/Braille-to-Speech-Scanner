"""Positive-only lexical evidence for structurally validated Thai paths."""
from copy import deepcopy
from functools import lru_cache
import logging
import unicodedata

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def default_lexicon():
    """Load the packaged general word list once; never fetch data at runtime."""
    try:
        from pythainlp.corpus import thai_words
        return frozenset(unicodedata.normalize('NFC', word) for word in thai_words() if word)
    except (ImportError, OSError) as exc:
        logger.warning('Thai lexicon unavailable; keeping structural candidates: %s', exc)
        return frozenset()


def _coverage(parts, words):
    """Best known-character coverage at syllable boundaries, allowing unknowns.

    No substring match inside a syllable and no negative score for a new name.
    Whole words may span several syllables, but never a parser run boundary.
    """
    best = [(0, []) for _ in range(len(parts) + 1)]
    for start in range(len(parts)-1, -1, -1):
        best[start] = best[start+1]
        word = ''
        for end in range(start+1, len(parts)+1):
            word += parts[end-1]
            if word in words:
                score = len(word) + best[end][0]
                if score > best[start][0]:
                    best[start] = (score, [dict(word=word, start=start, end=end), *best[end][1]])
    return best[0]


def rank_segmentations(parsed, lexicon, render_unit):
    """Rank complete valid paths; a lexical winner is not proof of validity.

    All structurally possible rivals remain plausible, including unseen names.
    A changed display path still needs review whenever a rival survives.
    """
    runs = parsed.get('segmentation_runs', [])
    if not runs:
        return parsed['syllables'], [], dict(source='unused', available=False)
    words = default_lexicon() if lexicon is None else frozenset(
        unicodedata.normalize('NFC', word) for word in lexicon if isinstance(word, str) and word)
    info = dict(source='pythainlp.corpus.thai_words' if lexicon is None else 'supplied',
                available=bool(words), word_count=len(words))
    replacements, rankings = {}, []
    for run in runs:
        original = parsed['syllables'][run['unit_start']:run['unit_start']+run['unit_count']]
        candidates = []
        if run['generation_complete']:
            for path in run['candidates']:
                if not path['valid']:
                    continue
                try:
                    parts = [render_unit(unit) for unit in path['syllables']]
                except ValueError:
                    # A renderer-level structural failure cannot gain validity
                    # from lexical support either.
                    continue
                score, matches = _coverage(parts, words)
                text = ''.join(parts)
                candidates.append(dict(text=text, valid=True, plausible=True,
                                       known_characters=score, matches=matches,
                                       coverage=score / max(1, len(text)),
                                       syllables=deepcopy(path['syllables'])))
        candidates.sort(key=lambda candidate: -candidate['coverage'])
        selected = None
        original_text = None
        reason = 'INCOMPLETE_CANDIDATE_SEARCH' if not run['generation_complete'] else 'NO_VALID_CANDIDATE'
        if candidates:
            highest = candidates[0]['coverage']
            best = [c for c in candidates if c['coverage'] == highest]
            reason = ('UNIQUE_STRUCTURAL_PATH' if len(candidates) == 1 else
                      'NO_LEXICAL_EVIDENCE' if highest == 0 else
                      'LEXICON_TIE' if len(best) > 1 else 'LEXICON_SUPPORTED_DISPLAY')
            # With no evidence, preserve the structural display preference.
            # On a tie keep it when it is among the strongest candidates.
            try:
                original_text = ''.join(render_unit(unit) for unit in original)
            except ValueError:
                pass
            if highest:
                selected = next((c for c in best if c['text'] == original_text), best[0])
        # A lexical annotation must not erase existing parser evidence when
        # it agrees with the structural display path.
        units = deepcopy(selected['syllables'] if selected and selected['text'] != original_text else original)
        review = (not run['generation_complete'] or len(candidates) > 1
                  or any(u.get('boundary_review') for u in original))
        if review and units:
            # Preserve the original evidence; rank scores never clear a warning.
            origin = next((u for u in original if u.get('boundary_candidates')), original[0])
            owner = next((u for u in units if origin['indices'][0] in u['indices']), units[0])
            for key in ('boundary_indices', 'boundary_patterns', 'boundary_source_cells',
                        'boundary_candidates', 'boundary_evidence'):
                if key in origin:
                    owner[key] = deepcopy(origin[key])
            owner['boundary_review'] = True
            owner['boundary_status'] = (origin.get('boundary_status') or
                                           ('AMBIGUOUS_SEGMENTATION' if len(candidates) > 1
                                            else 'UNRESOLVED_SEGMENTATION'))
        replacements[run['unit_start']] = (run['unit_count'], units)
        rankings.append(dict(indices=run['indices'], generation_complete=run['generation_complete'],
                             reason=reason, candidates=candidates, needs_review=review,
                             selected_text=selected['text'] if selected else None))
    result = []
    index = 0
    while index < len(parsed['syllables']):
        if index in replacements:
            count, units = replacements[index]
            result.extend(units)
            index += count
        else:
            result.append(parsed['syllables'][index])
            index += 1
    return result, rankings, info
