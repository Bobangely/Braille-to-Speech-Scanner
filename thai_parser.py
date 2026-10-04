"""Parse token sequences into Thai syllables before orthographic rendering."""

from copy import deepcopy

from thai_renderer import (CONSONANTS, TONES, LEADING, SHORT, INHERENT_FINAL,
                           CODA_CHANGES_VOWEL, normalize_vowel)
from thai_braille import BRAILLE_STANDARD, RULES_SOURCE, lex_braille, match_post_onset_vowel


def _is_consonant(atom):
    return atom['kind'] in ('consonant', 'ambiguous_o')


def _component(kind, value, atoms, rule):
    indices = [i for atom in atoms for i in atom.get('source_indices', [atom['index']])]
    return dict(type=kind, value=value, indices=indices, rule=rule,
                patterns=[p for atom in atoms for p in atom.get('patterns', [])],
                source_cells=[c for atom in atoms for c in atom.get('source_cells', [])])


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
    if not _is_consonant(atoms[index]) or atoms[index]['value'] != 'อ':
        return False
    end = index + 1
    while end < len(atoms) and atoms[end]['kind'] == 'tone':
        end += 1
    if end == len(atoms):
        return True
    if atoms[end]['kind'] == 'vowel' or atoms[end]['value'] == 'อ':
        return False  # อิ/ออ/... start their own syllable, as in เต็มอิ่ม.
    if atoms[end]['kind'] == 'leading':
        return end+1 < len(atoms) and _is_consonant(atoms[end+1])
    return True


def _is_short_o_pair(atoms, index):
    """Recognize adjacent written อ + ะ; the caller must supply onset context."""
    # Physical Braille uses the raw sequence table and its adjacency checks.
    return (not atoms[index].get('patterns') and index+1 < len(atoms) and _is_consonant(atoms[index])
            and atoms[index]['value'] == 'อ' and atoms[index+1]['kind'] == 'vowel'
            and atoms[index+1]['value'] == 'ะ')


def _starts_syllable(atoms, index, *, onset_context=False):
    """A consonant with its own following vowel belongs to the next syllable."""
    end = index + 1
    if end < len(atoms) and _is_consonant(atoms[end]) and _cluster(atoms[index]['value'], atoms[end]['value']):
        end += 1
    while end < len(atoms) and atoms[end]['kind'] == 'tone':
        end += 1
    if end >= len(atoms):
        return False
    if atoms[end]['kind'] == 'leading':
        # Written แ + ม starts the next syllable; it is not a vowel of the
        # preceding ณ in คุณแม่. Post-onset เ + ะ remains a short-vowel form.
        return end+1 >= len(atoms) or not _is_consonant(atoms[end+1])
    # อ can be a written vowel after an onset (กล + ่ + อ + ง).
    # Its lexical token remains a consonant because it can also be an onset;
    # identify its vowel role here before borrowing that onset as a coda.
    # In coda lookahead อ + ะ can be an independent อ onset (กินอะ).
    # Compound-vowel evidence is safe while resolving the current onset cluster.
    return (atoms[end]['kind'] in ('vowel', 'ambiguous_mark') or _is_written_o(atoms, end)
            or (onset_context and (_is_short_o_pair(atoms, end)
                                   or match_post_onset_vowel(atoms, end) is not None)))


def _implicit_closed_span(atoms, index):
    """Recognize an onset + omitted โอะ + final with a supported right boundary.

    An explicit vowel owns its onset first. A residual bare consonant is not
    boundary evidence: the following run must end or start another complete
    syllable. This prevents turning every adjacent consonant into a coda.
    """
    if (index >= len(atoms) or not _is_consonant(atoms[index])
            or _starts_syllable(atoms, index, onset_context=True)):
        return None
    onset_end = index + 1
    if (onset_end+1 < len(atoms) and _is_consonant(atoms[onset_end])
            and _is_consonant(atoms[onset_end+1])
            and _cluster(atoms[index]['value'], atoms[onset_end]['value'])
            and not _starts_syllable(atoms, onset_end, onset_context=True)):
        onset_end += 1
    final_index = onset_end
    while final_index < len(atoms) and atoms[final_index]['kind'] == 'tone':
        final_index += 1
    if (final_index >= len(atoms) or not _is_consonant(atoms[final_index])
            or atoms[final_index]['value'] == 'อ'
            or _starts_syllable(atoms, final_index, onset_context=True)):
        return None
    end = final_index + 1
    while end < len(atoms) and atoms[end]['kind'] == 'tone':
        end += 1
    if end < len(atoms):
        preposed = (atoms[end]['kind'] == 'leading' and end+1 < len(atoms)
                    and _is_consonant(atoms[end+1]))
        explicit = _is_consonant(atoms[end]) and _starts_syllable(atoms, end, onset_context=True)
        if not (preposed or explicit or _implicit_closed_span(atoms, end) is not None):
            return None
    return onset_end, final_index, end


def _final_role(atoms, index, vowel):
    """Resolve the syllable boundary before a renderer can reduce its vowel."""
    if _starts_syllable(atoms, index):
        return 'NEXT_EXPLICIT_ONSET'
    if _implicit_closed_span(atoms, index) is not None:
        # A short-a ending supports the open syllable. Where the vowel's
        # spelling is unchanged, prefer the complete following syllable too.
        # Otherwise a boundary could change the recognized vowel spelling.
        if vowel == 'อะ' or vowel in SHORT.values() or vowel not in CODA_CHANGES_VOWEL:
            return 'NEXT_IMPLICIT_ONSET'
        return 'AMBIGUOUS_BOUNDARY'
    return 'CODA_CONTEXT'


def _parse_run(atoms):
    """Parse a boundary-free run; preserve orthographic fragments as explicit units."""
    index = 0
    while index < len(atoms):
        start = index
        prefix = ''
        components, prefix_atoms, vowel_atoms = [], [], []
        vowel_rule, ambiguous_mark = 'VOWEL', False
        if atoms[index]['kind'] == 'leading' and index+1 < len(atoms) and _is_consonant(atoms[index+1]):
            prefix = atoms[index]['value']
            prefix_atoms.append(atoms[index])
            index += 1
        if not _is_consonant(atoms[index]):
            atom = atoms[index]
            status = ('AMBIGUOUS_SEQUENCE' if atom['kind'] == 'ambiguous_mark' else
                      'AMBIGUOUS_SYLLABLE' if atom['semantic'] else None)
            yield dict(indices=atom.get('source_indices', [atom['index']]), literal=atom['value'],
                       status=status, candidates=atom.get('candidates', []))
            index += 1
            continue
        onset_start = index
        implicit = _implicit_closed_span(atoms, index) if not prefix else None
        onset = atoms[index]['value']
        index += 1
        if (index < len(atoms) and _is_consonant(atoms[index])
                and _cluster(onset, atoms[index]['value'])
                and (prefix or _starts_syllable(atoms, index, onset_context=True)
                     or (implicit is not None and implicit[0] > index))):
            onset += atoms[index]['value']
            index += 1
        components.append(_component('ONSET', onset, atoms[onset_start:index], 'ONSET_CONTEXT'))
        vowel, tone, semantic, status = None, None, False, None
        while index < len(atoms):
            atom = atoms[index]
            matched = match_post_onset_vowel(atoms, index) if vowel is None and not prefix else None
            if atom['kind'] == 'tone':
                if tone:
                    status = 'UNSUPPORTED_COMBINATION'
                tone = atom['value']
                components.append(_component('TONE', tone, [atom], 'SYLLABLE_TONE'))
            elif matched:
                vowel, span = matched
                semantic, vowel_rule = True, 'POST_ONSET_SEQUENCE'
                vowel_atoms.extend(atoms[index:index+span])
                index += span
                continue
            elif atom['kind'] == 'leading' and not prefix and vowel is None:
                if index+1 < len(atoms) and _is_consonant(atoms[index+1]):
                    break
                prefix = atom['value']
                prefix_atoms.append(atom)
            elif vowel is None and not prefix and _is_short_o_pair(atoms, index):
                # Consume both physical tokens as one semantic vowel, retaining
                # their original indices/raw tokens for diagnostics.
                vowel, semantic = 'เอาะ', True
                vowel_atoms.extend(atoms[index:index+2])
                index += 1
            elif vowel is None and _is_written_o(atoms, index):
                # Contextual written อ, rather than a semantic compound vowel.
                vowel = 'ออ'
                vowel_atoms.append(atom)
                vowel_rule = 'WRITTEN_O_AFTER_ONSET'
            elif atom['kind'] == 'ambiguous_mark' and vowel is None:
                vowel, ambiguous_mark = '็', True
                vowel_atoms.append(atom)
                vowel_rule = 'EXPLICIT_MAITAIKHU'
            elif atom['kind'] == 'vowel':
                if vowel is None:
                    vowel, semantic = atom['vowel'], atom['semantic']
                elif atom['vowel'] == 'อะ' and vowel in SHORT:
                    vowel, semantic = SHORT[vowel], True
                else:
                    break
                vowel_atoms.append(atom)
            else:
                break
            index += 1
        # A written leading vowel plus short-a denotes a short vowel. Other
        # fragments (เ + ก + ิ) remain already-written spelling, not semantic เออ.
        orthographic_prefix = ''
        if prefix:
            if vowel is None:
                vowel = LEADING[prefix]
                vowel_atoms = prefix_atoms
            elif vowel == 'อะ' and prefix in 'เแโ':
                vowel, semantic = SHORT[LEADING[prefix]], True
                vowel_atoms = prefix_atoms + vowel_atoms
            elif prefix == 'เ' and vowel == 'อา' and index < len(atoms) and atoms[index]['vowel'] == 'อะ':
                vowel, semantic = 'เอาะ', True
                vowel_atoms = prefix_atoms + vowel_atoms + [atoms[index]]
                index += 1
            else:
                orthographic_prefix = prefix
                components.append(_component('PRE_VOWEL', prefix, prefix_atoms, 'WRITTEN_PREFIX'))
        if vowel is None and not prefix and implicit is not None and index == implicit[1]:
            vowel, semantic, vowel_rule = 'โอะ', True, 'IMPLICIT_O_CLOSED'
        if vowel:
            components.append(_component('VOWEL', vowel, vowel_atoms, vowel_rule))
        final = None
        boundary_rule = 'RUN_END'
        if vowel and index < len(atoms) and _is_consonant(atoms[index]):
            boundary_rule = ('CODA_IMPLICIT_O' if vowel_rule == 'IMPLICIT_O_CLOSED'
                             else _final_role(atoms, index, vowel))
            if (not boundary_rule.startswith('NEXT_') and
                    (semantic or vowel not in INHERENT_FINAL)):
                if boundary_rule == 'AMBIGUOUS_BOUNDARY':
                    status = status or 'AMBIGUOUS_SYLLABLE'
                tail = index
                while tail < len(atoms) and _is_consonant(atoms[tail]):
                    tail += 1
                if (semantic and vowel_rule != 'IMPLICIT_O_CLOSED'
                        and tail-index > 1 and tail == len(atoms)
                        and not _starts_syllable(atoms, index+1)
                        and _implicit_closed_span(atoms, index+1) is None):
                    # Several bare consonants could be a coda, silent ending,
                    # or another implicit-vowel syllable. No lexicon to decide.
                    status = status or 'AMBIGUOUS_SYLLABLE'
                final = atoms[index]['value']
                components.append(_component('FINAL', final, [atoms[index]], 'CODA_CONTEXT'))
                index += 1
                if index < len(atoms) and atoms[index]['kind'] == 'tone':
                    if tone:
                        status = 'UNSUPPORTED_COMBINATION'
                    tone = atoms[index]['value']
                    components.append(_component('TONE', tone, [atoms[index]], 'SYLLABLE_TONE'))
                    index += 1
        if ambiguous_mark:
            # A supplied mark in an open written Thai syllable (e.g. ก็) is
            # not a generated short-vowel mark. Keep that existing reading;
            # an incompatible closed form still needs additional context.
            explicit_open = not orthographic_prefix and final is None and tone is None
            explicit_closed = orthographic_prefix in ('เ', 'แ') and final is not None
            if not (explicit_open or explicit_closed):
                status = status or 'AMBIGUOUS_SEQUENCE'
        yield dict(onset=onset, vowel=vowel, tone=tone, final=final,
                   tone_anchor=len(onset)-1, prefix=orthographic_prefix,
                   indices=[i for a in atoms[start:index] for i in a.get('source_indices', [a['index']])],
                   boundary_rule=boundary_rule,
                   components=sorted(components, key=lambda c: c['indices'][0] if c['indices']
                                     else atoms[onset_start]['index']+.5), status=status)


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
        # Raw Braille must use the same adjacency boundary for lexical and
        # contextual matches; the text-token compatibility path has no geometry.
        sequence_break = atoms is not None and not atom.get('adjacent_before', True)
        if line_break or record.get('break_before') or sequence_break:
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
            indices = atom.get('source_indices', [index])
            if atom['kind'] == 'error':
                literal('�', indices, record.get('warning') or 'UNKNOWN_TOKEN')
            elif value and not (value == ' ' and (not units or units[-1].get('literal') in (' ', '\n'))):
                literal(value, indices)
        else:
            run.append(atom)
    flush()
    return dict(raw_tokens=raw, normalized_tokens=normalized, syllables=units)


def parse_braille(cells):
    """Raw patterns -> contextual semantic syllables, without producing Thai text."""
    raw, tokens, atoms = lex_braille(cells)
    parsed = parse_tokens(tokens, atoms=atoms)
    semantic, trace = [], []
    states = {'ONSET': 'ONSET', 'VOWEL': 'POST_VOWEL', 'PRE_VOWEL': 'PRE_VOWEL',
              'TONE': 'TONE', 'FINAL': 'FINAL'}
    for syllable in parsed['syllables']:
        indices = syllable['indices']
        syllable['raw_patterns'] = [raw[i]['pattern'] for i in indices]
        syllable['source_cells'] = [raw[i]['cell_id'] for i in indices]
        state = 'START'
        for component in syllable.get('components', []):
            semantic.append(deepcopy(component))
            following = states[component['type']]
            trace.append(dict(component, state_before=state, state_after=following))
            state = following
        if indices:
            trace.append(dict(state_before=state, state_after='INVALID' if syllable.get('status') else 'COMPLETE',
                              rule=syllable.get('status') or 'SYLLABLE_END',
                              patterns=syllable['raw_patterns'], source_cells=syllable['source_cells']))
    return dict(parsed, standard=BRAILLE_STANDARD, rules_source=RULES_SOURCE, raw_cells=raw,
                raw_patterns=[cell['pattern'] for cell in raw], semantic_tokens=semantic, parser_trace=trace)
