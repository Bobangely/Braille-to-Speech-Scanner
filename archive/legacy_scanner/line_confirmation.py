"""Confirm independent readable lines without promoting reviewed text to final."""
from collections import deque


def reading_lines(cells, tokens, text):
    """Scope existing decoder output to contiguous detected lines; never re-decode.

    If the text/token boundaries cannot be aligned exactly, no partial reading
    is offered. The existing whole-frame confirmation still handles that case.
    """
    if not cells or len(tokens) != len(cells):
        return []
    groups = []
    seen = set()
    for index, cell in enumerate(cells):
        line_id = cell.get('line_id', 0)
        if not groups or groups[-1][0] != line_id:
            if line_id in seen:
                return []
            seen.add(line_id)
            groups.append((line_id, []))
        groups[-1][1].append(index)
    texts = text.split('\n')
    if len(texts) != len(groups):
        return []
    lines = []
    for (line_id, indices), candidate in zip(groups, texts):
        warnings = [str(tokens[i][key]) for i in indices
                    for key in ('warning', 'composition_warning') if tokens[i].get(key)]
        status = 'uncertain' if warnings or '�' in candidate else 'ok' if candidate.strip() else 'empty'
        lines.append(dict(line_id=line_id, text=candidate.strip(), status=status,
                          reason=', '.join(dict.fromkeys(warnings)) or ('incomplete_symbol' if status == 'uncertain' else None),
                          indices=tuple(indices),
                          identity=tuple(cells[i].get('track_id', cells[i].get('cell_id', i)) for i in indices)))
    return lines


class LineConfirmation:
    """Keep only current-line evidence, counted once per completed inference."""
    def __init__(self, required=6):
        self.required = max(1, int(required))
        self.reset()

    def reset(self):
        self._context = None
        self._topology = None
        self._last_result_id = None
        self._history = {}
        self.lines = []

    def update(self, result):
        lines = result.get('line_results', [])
        result_id = result.get('result_id', 0)
        if (result_id <= 0 or not lines or result.get('error')
                or result.get('status') not in ('ok', 'uncertain')
                or result.get('debug_info', {}).get('grid_pending')):
            self.reset()
            return
        context = (result.get('context'), result.get('lang'), result.get('source_shape'), result.get('dot_color'))
        topology = tuple(line['line_id'] for line in lines)
        if context != self._context or topology != self._topology:
            self.reset()
            self._context, self._topology = context, topology
        if self._last_result_id is not None and result_id < self._last_result_id:
            self.reset()
            return
        if result_id == self._last_result_id:
            return
        # Tracker IDs restart when motion/context is reacquired. Equal text
        # and IDs are not evidence from the same page lineage. Process this
        # only once per inference, not once per display tick.
        if result.get('debug_info', {}).get('grid_tracking') == 'acquired':
            self._history.clear()
            self.lines = []
        self._last_result_id = result_id
        current = []
        for line in lines:
            identity = (line['identity'], line['text'])
            previous, history = self._history.get(line['line_id'], (None, deque(maxlen=self.required)))
            if identity != previous or line['status'] != 'ok':
                history.clear()
            if line['status'] == 'ok':
                history.append(line['text'])
            self._history[line['line_id']] = (identity, history)
            current.append(dict(line, samples=len(history), confirmed=len(history) >= self.required))
        self.lines = current

    def preview_tokens(self, tokens):
        """Publish labels only for confirmed lines, preserving the physical index."""
        allowed = {i for line in self.lines if line['confirmed'] for i in line['indices']}
        if not allowed:
            return []
        return [dict(token) if i in allowed else dict(token, char='…', consumed=False)
                for i, token in enumerate(tokens)]
