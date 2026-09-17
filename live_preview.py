"""Bounded camera preview with cached text/grid drawing; inference keeps source pixels."""

import cv2
import numpy as np
from frame_motion import cell_motion_mask, estimate_motion, full_size_motion, tracking_gray
from grid_stability import GridStabilizer


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
    def __init__(self, max_width=1280, dashboard=False):
        self.max_width = max_width
        self.dashboard = dashboard
        self.details = False
        self._key = None
        self.motion_valid = True
        self._reference = None
        self._motion_mask = None
        self._grids = GridStabilizer()
        self._waiting_key = None
        self._grid_key = None

    def render(self, frame, detector, result, lang):
        height, width = frame.shape[:2]
        scale = min(1.0, self.max_width / width)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        preview = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        grid_key = (result['result_id'], frame.shape, lang, result.get('reader_name'), result.get('context'))
        key = (grid_key, size, detector.mode, result['decoded_text'], bool(result['verbose_results']),
               self.dashboard, self.details)
        style = dict(details=self.details, footer=False, cell_hud=True) if self.dashboard else {}
        if key != self._key:
            reference = result.get('camera_roi')
            if grid_key != self._grid_key:
                if result['cells'] and all(c.get('grid_tracked') for c in result['cells']):
                    # Recognition already used this geometry. A second smoother
                    # here would display different slots from those actually read.
                    self._stable_cells = result['cells']
                    self._grids.reset()
                else:
                    self._stable_cells = self._grids.update(result['cells'], reference,
                        (result.get('context'), frame.shape, lang, result.get('reader_name')))
                self._grid_key = grid_key
            dots, cells = scaled_geometry(result['dots'], self._stable_cells, size[0]/width, size[1]/height)
            self._overlay = detector.annotate_with_text(np.zeros_like(preview), dots, cells,
                decoded_text=result['decoded_text'], verbose_results=result['verbose_results'],
                lang=lang, reader_name=result.get('reader_name'), **style)
            self._mask = np.any(self._overlay > 0, axis=2).astype(np.uint8) * 255
            self._reference = (tracking_gray(reference) if result['cells'] and
                isinstance(reference, np.ndarray) and reference.shape == frame.shape else None)
            self._motion_mask = (cell_motion_mask(self._stable_cells, frame.shape, self._reference.shape)
                                 if self._reference is not None else None)
            waiting_key = (size, lang, detector.mode, result.get('reader_name'), self.dashboard, self.details)
            if self._reference is not None and waiting_key != self._waiting_key:
                self._waiting = detector.annotate_with_text(np.zeros_like(preview), [], [],
                    decoded_text='', lang=lang, reader_name=result.get('reader_name'), **style)
                self._waiting_mask = np.any(self._waiting > 0, axis=2).astype(np.uint8) * 255
                self._waiting_key = waiting_key
            self._key = key
        self.motion_valid = True
        overlay, mask = self._overlay, self._mask
        if self._reference is not None:
            motion = estimate_motion(self._reference, tracking_gray(frame), self._motion_mask)
            self.motion_valid = motion is not None
            if motion is None:
                overlay, mask = self._waiting, self._waiting_mask
            else:
                transform = full_size_motion(motion, preview.shape, self._reference.shape)[:2]
                # Move only spatial annotations; keep the text footer fixed on screen.
                overlay, mask = self._overlay.copy(), self._mask.copy()
                overlay[:size[1]] = cv2.warpAffine(self._overlay[:size[1]], transform, size)
                mask[:size[1]] = cv2.warpAffine(self._mask[:size[1]], transform, size, flags=cv2.INTER_NEAREST)
        # Waiting text can have a shorter footer than a multiline result.
        canvas = np.zeros_like(overlay)
        canvas[:size[1]] = preview
        cv2.copyTo(overlay, mask, canvas)
        return canvas
