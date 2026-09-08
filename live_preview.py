"""Bounded camera preview with cached text/grid drawing; inference keeps source pixels."""

import cv2
import numpy as np


def scaled_geometry(dots, cells, sx, sy):
    def point(p):
        return (p[0] * sx, p[1] * sy)

    def box(b):
        return tuple(int(round(v * (sx if i % 2 == 0 else sy))) for i, v in enumerate(b))

    small_dots, small_cells = [], []
    for dot in dots:
        small = dict(dot, center=point(dot['center']), area=dot['area'] * sx * sy)
        if dot.get('bbox'):
            small['bbox'] = box(dot['bbox'])
        small_dots.append(small)
    for cell in cells:
        small = dict(cell, center=point(cell['center']), x=cell['x'] * sx, y=cell['y'] * sy)
        if cell.get('grid'):
            grid = cell['grid']
            small['grid'] = dict(grid, bbox=box(grid['bbox']),
                                 expected_cols=[x * sx for x in grid['expected_cols']],
                                 expected_rows=[y * sy for y in grid['expected_rows']],
                                 slots={k: point(p) for k, p in grid['slots'].items()})
        small_cells.append(small)
    return small_dots, small_cells


class LivePreview:
    def __init__(self, max_width=1280):
        self.max_width = max_width
        self._key = None

    def render(self, frame, detector, result, lang):
        height, width = frame.shape[:2]
        scale = min(1.0, self.max_width / width)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        preview = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        key = (result['result_id'], frame.shape, size, lang, detector.mode)
        if key != self._key:
            dots, cells = scaled_geometry(result['dots'], result['cells'], size[0]/width, size[1]/height)
            self._overlay = detector.annotate_with_text(np.zeros_like(preview), dots, cells,
                                decoded_text=result['decoded_text'],
                                verbose_results=result['verbose_results'], lang=lang)
            self._mask = np.any(self._overlay != 0, axis=2).astype(np.uint8) * 255
            self._key = key
        canvas = np.zeros_like(self._overlay)
        canvas[:size[1]] = preview
        cv2.copyTo(self._overlay, self._mask, canvas)
        return canvas
