"""Token-aware Thai syllable composition, independent of Braille/vision.

Semantic vowels undergo spelling changes. Already-written vowel fragments keep
their identity (e.g. ื is not silently expanded to ือ). No dictionary guessing
or whole-sentence replacement is performed. Ambiguous forms retain a candidate
in the trace but produce a review marker in the readable text.
"""

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import unicodedata


CONSONANTS = set('กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ')
TONES = set('่้๊๋')
LEADING = {'เ': 'เอ', 'แ': 'แอ', 'โ': 'โอ', 'ไ': 'ไอ', 'ใ': 'ใอ'}
# Every rule places the tone explicitly, after combining vowels and before
# trailing vowels/finals. O includes the complete onset; its last letter anchors T.
RULES = {
    'อะ': ('{O}{T}ะ', '{O}ั{T}{F}'),
    'อา': ('{O}{T}า', '{O}{T}า{F}'),
    'อิ': ('{O}ิ{T}', '{O}ิ{T}{F}'),
    'อี': ('{O}ี{T}', '{O}ี{T}{F}'),
    'อึ': ('{O}ึ{T}', '{O}ึ{T}{F}'),
    'อือ': ('{O}ื{T}อ', '{O}ื{T}{F}'),
    'อุ': ('{O}ุ{T}', '{O}ุ{T}{F}'),
    'อู': ('{O}ู{T}', '{O}ู{T}{F}'),
    'เอะ': ('เ{O}{T}ะ', 'เ{O}็{T}{F}'),
    'เอ': ('เ{O}{T}', 'เ{O}{T}{F}'),
    'แอะ': ('แ{O}{T}ะ', 'แ{O}็{T}{F}'),
    'แอ': ('แ{O}{T}', 'แ{O}{T}{F}'),
    'โอะ': ('โ{O}{T}ะ', '{O}{T}{F}'),
    'โอ': ('โ{O}{T}', 'โ{O}{T}{F}'),
    'เอาะ': ('เ{O}{T}าะ', '{O}็{T}อ{F}'),
    'ออ': ('{O}{T}อ', '{O}{T}อ{F}'),
    'เออ': ('เ{O}{T}อ', 'เ{O}ิ{T}{F}'),
    'เออะ': ('เ{O}{T}อะ', 'เ{O}ิ{T}{F}'),
    'เอีย': ('เ{O}ี{T}ย', 'เ{O}ี{T}ย{F}'),
    'เอือ': ('เ{O}ื{T}อ', 'เ{O}ื{T}อ{F}'),
    'อัว': ('{O}ั{T}ว', '{O}{T}ว{F}'),
    'เอียะ': ('เ{O}ี{T}ยะ', None),
    'เอือะ': ('เ{O}ื{T}อะ', None),
    'อัวะ': ('{O}ั{T}วะ', None),
    'อำ': ('{O}{T}ำ', None),
    'ไอ': ('ไ{O}{T}', None),
    'ใอ': ('ใ{O}{T}', None),
    'เอา': ('เ{O}{T}า', None),
    # Orthographic tokens already read from the page, not semantic vowel names.
    'ั': ('{O}ั{T}', '{O}ั{T}{F}'),
    'ื': ('{O}ื{T}', '{O}ื{T}{F}'),
    '็': ('{O}็{T}', '{O}็{T}{F}'),
    'ํ': ('{O}ํ{T}', '{O}ํ{T}{F}'),
}
ALIASES = {name: name for name in RULES}
ALIASES.update({
    'ะ': 'อะ', '-ะ': 'อะ', 'า': 'อา', '-า': 'อา',
    'ิ': 'อิ', '-ิ': 'อิ', 'ี': 'อี', '-ี': 'อี',
    'ึ': 'อึ', '-ึ': 'อึ', '-ือ': 'อือ', 'ุ': 'อุ', '-ุ': 'อุ',
    'ู': 'อู', '-ู': 'อู', 'เ-': 'เอ', 'แ-': 'แอ', 'โ-': 'โอ',
    'เ-ะ': 'เอะ', 'แ-ะ': 'แอะ', 'โ-ะ': 'โอะ', 'เ-าะ': 'เอาะ',
    '-อ': 'ออ', 'เ-อ': 'เออ', 'เ-อะ': 'เออะ',
    'เ-ีย': 'เอีย', 'เ-ือ': 'เอือ', '-ัว': 'อัว',
    'เ-ียะ': 'เอียะ', 'เ-ือะ': 'เอือะ', '-ัวะ': 'อัวะ',
    'ำ': 'อำ', '-ำ': 'อำ', 'ไ-': 'ไอ', 'ใ-': 'ใอ', 'เ-า': 'เอา',
})
SHORT = {'เอ': 'เอะ', 'แอ': 'แอะ', 'โอ': 'โอะ', 'เออ': 'เออะ',
         'เอา': 'เอาะ', 'เอีย': 'เอียะ', 'เอือ': 'เอือะ', 'อัว': 'อัวะ'}
INHERENT_FINAL = {'อำ', 'ไอ', 'ใอ', 'เอา'}
SHORT_COMPOUNDS = {'เอียะ', 'เอือะ', 'อัวะ'}


class CompositionError(ValueError):
    def __init__(self, status):
        self.status = status
        super().__init__(status)


def normalize_vowel(value):
    """Normalize names/placeholder notation, never rewrite the detected text."""
    return ALIASES.get(value.removeprefix('สระ').replace('◌', '-'))


@lru_cache(maxsize=1)
def _exceptions():
    path = Path(__file__).with_name('thai_word_exceptions.json')
    with path.open(encoding='utf-8') as source:
        return {(r['onset'], r['vowel'], r.get('final'), r.get('tone')): r['text']
                for r in json.load(source)}


def compose_syllable(onset, vowel, final=None, tone=None):
    """Render an explicitly parsed syllable; invalid combinations raise a status."""
    vowel = normalize_vowel(vowel)
    if vowel is None:
        raise CompositionError('UNKNOWN_VOWEL')
    if not onset or any(c not in CONSONANTS for c in onset):
        raise CompositionError('AMBIGUOUS_SYLLABLE')
    if final and (len(final) != 1 or final not in CONSONANTS):
        raise CompositionError('INVALID_FINAL')
    if tone and tone not in TONES:
        raise CompositionError('UNSUPPORTED_COMBINATION')
    exception = _exceptions().get((onset, vowel, final or None, tone or None))
    if exception is not None:
        return exception
    if vowel == 'เออ' and final == 'ย':
        template = 'เ{O}{T}ย'
    else:
        template = RULES[vowel][bool(final)]
    if template is None:
        raise CompositionError('NEEDS_LEXICON' if vowel in SHORT_COMPOUNDS else 'INVALID_FINAL')
    # Mai taikhu plus a tone needs lexical evidence; do not emit two competing marks.
    if tone and '็' in template:
        raise CompositionError('UNSUPPORTED_COMBINATION')
    return unicodedata.normalize('NFC', template.format(O=onset, T=tone or '', F=final or ''))


def _cluster(left, right):
    return ((left == 'ห' and right in 'งญนมยรลว') or (left == 'อ' and right == 'ย')
            or (left in 'กขคตปผพทศส' and right in 'รลว'))


def _atom(record, index):
    value = record.get('char', record.get('token', ''))
    canonical = normalize_vowel(value)
    if record.get('warning') or value == '�':
        kind = 'error'
    elif canonical is not None and len(value) > 1:
        kind = 'vowel'
    elif value in CONSONANTS or (len(value) > 1 and all(c in CONSONANTS for c in value)):
        kind = 'consonant'
    elif value in LEADING:
        kind = 'leading'
    elif value in TONES:
        kind = 'tone'
    elif canonical is not None:
        kind = 'vowel'
    else:
        kind = 'literal'
    return dict(value=value, vowel=canonical, kind=kind, index=index,
                semantic=kind == 'vowel' and len(value) > 1)


def _is_written_o(atoms, index):
    """Resolve อ as a vowel only when it has no own following vowel."""
    if atoms[index]['kind'] != 'consonant' or atoms[index]['value'] != 'อ':
        return False
    end = index + 1
    while end < len(atoms) and atoms[end]['kind'] == 'tone':
        end += 1
    if end == len(atoms):
        return True
    if atoms[end]['kind'] == 'vowel' or atoms[end]['value'] == 'อ':
        return False  # อิ/ออ/... start their own syllable, as in เต็มอิ่ม.
    if atoms[end]['kind'] == 'leading':
        return end+1 < len(atoms) and atoms[end+1]['kind'] == 'consonant'
    return True


def _is_short_o_pair(atoms, index):
    """Recognize adjacent written อ + ะ; the caller must supply onset context."""
    return (index+1 < len(atoms) and atoms[index]['kind'] == 'consonant'
            and atoms[index]['value'] == 'อ' and atoms[index+1]['kind'] == 'vowel'
            and atoms[index+1]['value'] == 'ะ')


def _starts_syllable(atoms, index, *, onset_context=False):
    """A consonant with its own following vowel belongs to the next syllable."""
    end = index + 1
    if end < len(atoms) and atoms[end]['kind'] == 'consonant' and _cluster(atoms[index]['value'], atoms[end]['value']):
        end += 1
    while end < len(atoms) and atoms[end]['kind'] == 'tone':
        end += 1
    if end >= len(atoms):
        return False
    if atoms[end]['kind'] == 'leading':
        # Written แ + ม starts the next syllable; it is not a vowel of the
        # preceding ณ in คุณแม่. Post-onset เ + ะ remains a short-vowel form.
        return end+1 >= len(atoms) or atoms[end+1]['kind'] != 'consonant'
    # อ can be a written vowel after an onset (กล + ่ + อ + ง).
    # Its lexical token remains a consonant because it can also be an onset;
    # identify its vowel role here before borrowing that onset as a coda.
    # In coda lookahead อ + ะ can be an independent อ onset (กินอะ).
    # Compound-vowel evidence is safe while resolving the current onset cluster.
    return (atoms[end]['kind'] == 'vowel' or _is_written_o(atoms, end)
            or (onset_context and _is_short_o_pair(atoms, end)))


def _parse_run(atoms):
    """Parse a boundary-free run; preserve orthographic fragments as explicit units."""
    index = 0
    while index < len(atoms):
        start = index
        prefix = ''
        if atoms[index]['kind'] == 'leading' and index+1 < len(atoms) and atoms[index+1]['kind'] == 'consonant':
            prefix = atoms[index]['value']
            index += 1
        if atoms[index]['kind'] != 'consonant':
            atom = atoms[index]
            status = 'AMBIGUOUS_SYLLABLE' if atom['semantic'] else None
            yield dict(indices=[atom['index']], literal=atom['value'], status=status)
            index += 1
            continue
        onset = atoms[index]['value']
        index += 1
        if (index < len(atoms) and atoms[index]['kind'] == 'consonant'
                and _cluster(onset, atoms[index]['value'])
                and (prefix or _starts_syllable(atoms, index, onset_context=True))):
            onset += atoms[index]['value']
            index += 1
        vowel, tone, semantic, status = None, None, False, None
        while index < len(atoms):
            atom = atoms[index]
            if atom['kind'] == 'tone':
                if tone:
                    status = 'UNSUPPORTED_COMBINATION'
                tone = atom['value']
            elif atom['kind'] == 'leading' and not prefix and vowel is None:
                if index+1 < len(atoms) and atoms[index+1]['kind'] == 'consonant':
                    break
                prefix = atom['value']
            elif vowel is None and not prefix and _is_short_o_pair(atoms, index):
                # Consume both physical tokens as one semantic vowel, retaining
                # their original indices/raw tokens for diagnostics.
                vowel, semantic = 'เอาะ', True
                index += 1
            elif vowel is None and _is_written_o(atoms, index):
                # Contextual written อ, rather than a semantic compound vowel.
                vowel = 'ออ'
            elif atom['kind'] == 'vowel':
                if vowel is None:
                    vowel, semantic = atom['vowel'], atom['semantic']
                elif atom['vowel'] == 'อะ' and vowel in SHORT:
                    vowel, semantic = SHORT[vowel], True
                else:
                    break
            else:
                break
            index += 1
        # A written leading vowel plus short-a denotes a short vowel. Other
        # fragments (เ + ก + ิ) remain already-written spelling, not semantic เออ.
        orthographic_prefix = ''
        if prefix:
            if vowel is None:
                vowel = LEADING[prefix]
            elif vowel == 'อะ' and prefix in 'เแโ':
                vowel, semantic = SHORT[LEADING[prefix]], True
            elif prefix == 'เ' and vowel == 'อา' and index < len(atoms) and atoms[index]['vowel'] == 'อะ':
                vowel, semantic = 'เอาะ', True
                index += 1
            else:
                orthographic_prefix = prefix
        final = None
        if vowel and index < len(atoms) and atoms[index]['kind'] == 'consonant':
            if not _starts_syllable(atoms, index) and (semantic or vowel not in INHERENT_FINAL):
                tail = index
                while tail < len(atoms) and atoms[tail]['kind'] == 'consonant':
                    tail += 1
                if (semantic and tail-index > 1 and tail == len(atoms)
                        and not _starts_syllable(atoms, index+1)):
                    # Several bare consonants could be a coda, silent ending,
                    # or another implicit-vowel syllable. No lexicon to decide.
                    status = status or 'AMBIGUOUS_SYLLABLE'
                final = atoms[index]['value']
                index += 1
                if index < len(atoms) and atoms[index]['kind'] == 'tone':
                    if tone:
                        status = 'UNSUPPORTED_COMBINATION'
                    tone = atoms[index]['value']
                    index += 1
        yield dict(onset=onset, vowel=vowel, tone=tone, final=final,
                   tone_anchor=len(onset)-1, prefix=orthographic_prefix,
                   indices=[a['index'] for a in atoms[start:index]], status=status)


def compose_tokens(tokens, *, lexicon=None, confidence_threshold=None):
    """Return composed text, immutable-source debug copies and review statuses.

    Accepts semantic strings or decoder records. Whitespace, punctuation, line
    changes and failed detections are hard parsing boundaries. A supplied lexicon
    validates uncertain candidates; it never replaces tokens with a guessed word.
    """
    raw = deepcopy(list(tokens))
    records = [dict(token=t) if isinstance(t, str) else t for t in raw]
    normalized, pieces, traces, run = [], [], [], []

    def flush():
        for syllable in _parse_run(run):
            status = syllable.pop('status')
            try:
                if 'literal' in syllable:
                    candidate = syllable['literal']
                elif syllable['vowel']:
                    candidate = syllable['prefix'] + compose_syllable(
                        syllable['onset'], syllable['vowel'], syllable['final'], syllable['tone'])
                    if syllable['vowel'] == 'เออะ' and syllable['final'] and (lexicon is None or candidate not in lexicon):
                        status = status or 'NEEDS_LEXICON'
                else:
                    candidate = syllable['onset'] + (syllable['tone'] or '')
            except CompositionError as exc:
                candidate, status = None, status or exc.status
            confidence = [records[i]['confidence'] for i in syllable['indices']
                          if records[i].get('confidence') is not None]
            minimum = min(confidence) if confidence else None
            if confidence_threshold is not None and minimum is not None and minimum < confidence_threshold:
                status = status or 'LOW_CONFIDENCE'
            text = '�' if status else candidate
            pieces.append(text)
            traces.append(dict(syllable, candidate=candidate, text=text, status=status or 'OK',
                               needs_review=bool(status), min_confidence=minimum))
        run.clear()

    last_line = None
    for index, record in enumerate(records):
        if record.get('consumed'):
            continue
        atom = _atom(record, index)
        line = record.get('line_id')
        line_break = record.get('line_break_before') or (line is not None and last_line is not None and line != last_line)
        if line is not None:
            last_line = line
        if line_break or record.get('break_before'):
            flush()
            if pieces:
                if line_break:
                    while pieces and pieces[-1] == ' ':
                        pieces.pop()
                    pieces.append('\n')
                elif pieces[-1] not in (' ', '\n'):
                    pieces.append(' ')
        normalized.append(dict(atom))
        if atom['kind'] in ('literal', 'error'):
            flush()
            value = atom['value']
            if atom['kind'] == 'error':
                pieces.append('�')
                traces.append(dict(indices=[index], status=record.get('warning') or 'UNKNOWN_TOKEN',
                                   needs_review=True, text='�', candidate=None))
            elif value and not (value == ' ' and (not pieces or pieces[-1] in (' ', '\n'))):
                pieces.append(value)
        else:
            run.append(atom)
    flush()
    return dict(text=unicodedata.normalize('NFC', ''.join(pieces).strip()),
                raw_tokens=raw, normalized_tokens=normalized, syllables=traces,
                needs_review=any(t['needs_review'] for t in traces))
