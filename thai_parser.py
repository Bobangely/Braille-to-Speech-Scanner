"""Parse token sequences into Thai syllables before orthographic rendering."""

from copy import deepcopy

from thai_renderer import CONSONANTS, TONES, LEADING, SHORT, INHERENT_FINAL, normalize_vowel


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


def parse_tokens(tokens, *, atoms=None):
    """Keep source tokens and return ordered syllables/literals without rendering."""
    raw = deepcopy(list(tokens))
    records = [dict(token=t) if isinstance(t, str) else t for t in raw]
    normalized, units, run = [], [], []

    def flush():
        units.extend(_parse_run(run))
        run.clear()

    def literal(value, indices, status=None):
        if value:
            units.append(dict(literal=value, indices=indices, status=status, trace=bool(status)))

    last_line = None
    for index, record in enumerate(records):
        if record.get('consumed'):
            continue
        atom = deepcopy(atoms[index]) if atoms is not None else _atom(record, index)
        line = record.get('line_id')
        line_break = record.get('line_break_before') or (line is not None and last_line is not None and line != last_line)
        if line is not None:
            last_line = line
        if line_break or record.get('break_before'):
            flush()
            if units:
                if line_break:
                    while units and units[-1].get('literal') == ' ':
                        units.pop()
                    literal('\n', [])
                elif units[-1].get('literal') not in (' ', '\n'):
                    literal(' ', [])
        normalized.append(dict(atom))
        if atom['kind'] in ('literal', 'error'):
            flush()
            value = atom['value']
            if atom['kind'] == 'error':
                literal('�', [index], record.get('warning') or 'UNKNOWN_TOKEN')
            elif value and not (value == ' ' and (not units or units[-1].get('literal') in (' ', '\n'))):
                literal(value, [index])
        else:
            run.append(atom)
    flush()
    return dict(raw_tokens=raw, normalized_tokens=normalized, syllables=units)
