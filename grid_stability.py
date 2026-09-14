"""Associate display grids across measured camera motion; preserve raw readings."""

import time

import cv2
import numpy as np

from frame_motion import estimate_motion, full_size_motion, tracking_gray


class GridStabilizer:
    """One bounded state per preview. Never retain absent cells or change dot bits."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._cells = []
        self._reference = None
        self._context = None
        self._updated_at = 0.
        self._next_id = 1

    def update(self, cells, image, context):
        if not cells or not isinstance(image, np.ndarray):
            self.reset()
            return cells
        # Keep support for existing callers which have no full spatial grid.
        if any(not c.get('grid') or not c.get('crop_quad') or
               len(c['grid'].get('slots', {})) != 6 for c in cells):
            self.reset()
            return cells
        now = time.monotonic()
        if context != self._context or now - self._updated_at > .75:
            self.reset()
        gray = tracking_gray(image)
        motion = estimate_motion(self._reference, gray) if self._reference is not None else None
        previous = self._cells
        if motion is None:
            previous = []
            self._next_id = 1
        else:
            motion = full_size_motion(motion, image.shape, gray.shape)
        current_slots = np.array([[c['grid']['slots'][i] for i in range(1, 7)] for c in cells])
        matches = {}
        predicted = None
        if previous:
            old_slots = np.float32([[c['grid']['slots'][i] for i in range(1, 7)] for c in previous])
            predicted = cv2.perspectiveTransform(old_slots.reshape(1, -1, 2), motion).reshape(-1, 6, 2)
            distances = np.linalg.norm(current_slots.mean(axis=1)[:, None] -
                                       predicted.mean(axis=1)[None, :], axis=2)
            nearest_old, nearest_new = distances.argmin(axis=1), distances.argmin(axis=0)
            for index, old_index in enumerate(nearest_old):
                spacing = np.linalg.norm(current_slots[index, 3] - current_slots[index, 0])
                # Mutual nearest-neighbour and all six slots must agree. A
                # changed row phase/marker interpretation starts a new track.
                residual = np.linalg.norm(current_slots[index] - predicted[old_index], axis=1)
                if nearest_new[old_index] == index and residual.max() < .35 * spacing:
                    matches[index] = int(old_index)
        stable = []
        for index, cell in enumerate(cells):
            result = dict(cell)
            if index in matches:
                old_index = matches[index]
                old = previous[old_index]
                slots = .75 * predicted[old_index] + .25 * current_slots[index]
                old_quad = cv2.perspectiveTransform(np.float32([old['crop_quad']]), motion)[0]
                quad = .75 * old_quad + .25 * np.array(cell['crop_quad'])
                center = slots.mean(axis=0)
                h, w = image.shape[:2]
                bbox = (max(0, int(np.floor(quad[:, 0].min()))), max(0, int(np.floor(quad[:, 1].min()))),
                        min(w, int(np.ceil(quad[:, 0].max()))), min(h, int(np.ceil(quad[:, 1].max()))))
                result.update(center=tuple(center), x=float(center[0]), y=float(center[1]),
                              crop_quad=quad.tolist(), track_id=old['track_id'],
                              grid=dict(cell['grid'], bbox=bbox,
                                  slots={i+1: tuple(p) for i, p in enumerate(slots)},
                                  expected_cols=[float(slots[:3, 0].mean()), float(slots[3:, 0].mean())],
                                  expected_rows=[float(slots[[i, i+3], 1].mean()) for i in range(3)]))
            else:
                result['track_id'] = self._next_id
                self._next_id += 1
            stable.append(result)
        # Preserve current spatial order and evidence. Track IDs, not list
        # positions, identify surviving cells after another cell disappears.
        self._cells, self._reference = stable, gray
        self._context, self._updated_at = context, now
        return stable
