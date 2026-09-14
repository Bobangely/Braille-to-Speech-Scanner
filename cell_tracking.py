"""Camera-only cell geometry tracking BEFORE colored-dot recognition.

Geometry may survive a bad proposal. Dot bits never survive a frame: every
returned cell is populated from this frame's measured colored components.
"""

import time

import cv2
import numpy as np

from frame_motion import cell_motion_mask, estimate_motion, full_size_motion, tracking_gray


def transform_grid(cell, matrix, shape):
    """Transform all spatial fields together, including the decoder's spacing."""
    slots = cv2.perspectiveTransform(np.float32([[cell['grid']['slots'][i]
                                               for i in range(1, 7)]]), matrix)[0]
    quad = cv2.perspectiveTransform(np.float32([cell['crop_quad']]), matrix)[0]
    return _geometry(cell, slots, quad, shape)


def _geometry(cell, slots, quad, shape):
    center = slots.mean(axis=0)
    axis = slots[3]-slots[0]
    spacing = float(np.linalg.norm(axis))
    scale = spacing / cell['dot_spacing']
    bbox = (max(0, int(np.floor(quad[:, 0].min()))), max(0, int(np.floor(quad[:, 1].min()))),
            min(shape[1], int(np.ceil(quad[:, 0].max()))), min(shape[0], int(np.ceil(quad[:, 1].max()))))
    return dict(cell, center=tuple(center), x=float(center[0]), y=float(center[1]),
        reading_x=float(np.dot(center, axis)/spacing), dot_spacing=spacing,
        cell_pitch=cell['cell_pitch']*scale, crop_quad=quad.tolist(),
        grid=dict(cell['grid'], slots={i+1: tuple(p) for i, p in enumerate(slots)}, bbox=bbox,
            expected_cols=[float(slots[:3, 0].mean()), float(slots[3:, 0].mean())],
            expected_rows=[float(slots[[i, i+3], 1].mean()) for i in range(3)]))


def split_joined_dots(dots, mask, cells):
    """Split a two-lobed color component only with current pixel evidence.

    Two supported grid slots AND a low-ink valley are required. A wide solid
    smudge must not become two dots simply because it covers two grid slots.
    """
    if mask is None or not cells or len(dots) < 4:
        return dots
    typical_area = float(np.median([dot['area'] for dot in dots]))
    slots = np.asarray([cell['grid']['slots'][i] for cell in cells for i in range(1, 7)])
    spacing = float(np.median([cell['dot_spacing'] for cell in cells]))
    radius = max(2., .18*spacing)

    def occupancy(point):
        x, y = point
        x1, x2 = max(0, int(x-radius)), min(mask.shape[1], int(x+radius)+1)
        y1, y2 = max(0, int(y-radius)), min(mask.shape[0], int(y+radius)+1)
        yy, xx = np.ogrid[y1:y2, x1:x2]
        disk = (xx-x)**2+(yy-y)**2 <= radius**2
        return float(np.mean(mask[y1:y2, x1:x2][disk] > 0)) if disk.any() else 0.

    refined = []
    for dot in dots:
        bbox = dot.get('bbox')
        if bbox is None or dot['area'] < 1.5*typical_area:
            refined.append(dot)
            continue
        x1, y1, x2, y2 = bbox
        inside = slots[(slots[:, 0] >= x1) & (slots[:, 0] < x2) &
                       (slots[:, 1] >= y1) & (slots[:, 1] < y2)]
        peaks = [point for point in inside if occupancy(point) >= .6]
        if (len(peaks) != 2 or not .7*spacing <= np.linalg.norm(peaks[1]-peaks[0]) <= 1.5*spacing or
                occupancy(np.mean(peaks, axis=0)) >= .5*min(map(occupancy, peaks))):
            refined.append(dot)
            continue
        yy, xx = np.nonzero(mask[y1:y2, x1:x2])
        pixels = np.column_stack((xx+x1, yy+y1))
        distance = np.linalg.norm(pixels[:, None]-np.asarray(peaks)[None], axis=2)
        parts = []
        nearest = distance.argmin(axis=1)
        for index, peak in enumerate(peaks):
            part = pixels[(nearest == index) & (distance[:, index] <= .45*spacing)]
            if len(part) < max(6, .25*typical_area) or np.linalg.norm(part.mean(axis=0)-peak) > .3*spacing:
                break
            low, high = part.min(axis=0), part.max(axis=0)+1
            parts.append(dict(dot, center=tuple(part.mean(axis=0)), area=float(len(part)),
                              bbox=tuple(map(int, (*low, *high))), color_pixels=len(part),
                              split_from_bbox=bbox))
        refined.extend(parts if len(parts) == 2 else [dot])
    return refined


class CellGridTracker:
    """Bounded geometry state; three fresh proposals confirm a topology change."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cells = []
        self.reference = self.mask = None
        self.context = None
        self.updated_at = 0.
        self.pending = []
        self.pending_count = 0
        self.next_id = 1
        self.current_dots = []

    @staticmethod
    def _associate(predicted, measured):
        if not predicted or not measured:
            return {}
        before = np.asarray([[c['grid']['slots'][i] for i in range(1, 7)] for c in predicted])
        after = np.asarray([[c['grid']['slots'][i] for i in range(1, 7)] for c in measured])
        distance = np.linalg.norm(before.mean(axis=1)[:, None]-after.mean(axis=1)[None], axis=2)
        nearest_new, nearest_old = distance.argmin(axis=1), distance.argmin(axis=0)
        matches = {}
        for old, new in enumerate(nearest_new):
            spacing = predicted[old]['dot_spacing']
            if (nearest_old[new] == old and
                    np.linalg.norm(before[old]-after[new], axis=1).max() < .35*spacing and
                    abs(measured[new]['cell_pitch']/predicted[old]['cell_pitch']-1) < .2):
                matches[old] = int(new)
        return matches

    @staticmethod
    def _assign(cells, dots):
        """One measured dot belongs to at most one cell/slot; no stored bits."""
        evidence = [[] for _ in cells]
        if not cells or not dots:
            return evidence, 0
        slots = np.asarray([[c['grid']['slots'][i] for i in range(1, 7)] for c in cells])
        widths = np.asarray([c['dot_spacing'] for c in cells])
        points = np.asarray([d['center'] for d in dots])
        assigned = 0
        for start in range(0, len(points), 128):
            distance = np.linalg.norm(points[start:start+128, None, None]-slots[None], axis=3)
            distance /= widths[None, :, None]
            nearest = distance.reshape(len(distance), -1).argmin(axis=1)
            for offset, slot in enumerate(nearest):
                cell, position = divmod(int(slot), 6)
                if distance[offset, cell, position] <= .38:
                    evidence[cell].append(start+offset)
                    assigned += 1
        return evidence, assigned

    def update(self, measured, dots, image, context, color_mask=None):
        now = time.monotonic()
        context = (context, image.shape)
        if context != self.context or now-self.updated_at > .75 or not dots:
            self.reset()
        if not dots:
            return [], 'empty'
        gray = tracking_gray(image)
        motion = (estimate_motion(self.reference, gray, self.mask)
                  if self.reference is not None else None)
        if motion is None:
            self.reset()
            chosen, status = measured, 'acquired'
        else:
            motion = full_size_motion(motion, image.shape, gray.shape)
            predicted = [transform_grid(c, motion, image.shape) for c in self.cells]
            dots = split_joined_dots(dots, color_mask, predicted)
            evidence, assigned = self._assign(predicted, dots)
            # New/missing cells need confirmation. Pure rephasing of a proposal
            # cannot override a tracked grid still supported by current dots.
            supported = bool(predicted) and all(evidence) and assigned == len(dots)
            if supported:
                self.pending, self.pending_count = [], 0
                chosen, status = predicted, 'tracked'
                for old, new in self._associate(predicted, measured).items():
                    current = measured[new]
                    if current.get('row_ambiguous') or current.get('column_ambiguous'):
                        continue
                    # Correct residual detector jitter without slowing real motion.
                    source = np.float32([predicted[old]['grid']['slots'][i] for i in range(1, 7)])
                    target = np.float32([current['grid']['slots'][i] for i in range(1, 7)])
                    quad = .75*np.asarray(predicted[old]['crop_quad'])+.25*np.asarray(current['crop_quad'])
                    chosen[old] = _geometry(predicted[old], .75*source+.25*target, quad, image.shape)
                    for key in ('row_ambiguous', 'column_ambiguous', 'pitch_ambiguous'):
                        chosen[old][key] = current[key]
            else:
                pending = [transform_grid(c, motion, image.shape) for c in self.pending]
                consistent = (bool(measured) and len(pending) == len(measured) and
                              len(self._associate(pending, measured)) == len(measured))
                self.pending_count = self.pending_count+1 if consistent else 1
                self.pending = measured
                if self.pending_count >= 3:
                    # Carry IDs for surviving cells; accept new layout only now.
                    measured = [dict(c) for c in measured]
                    for old, new in self._associate(predicted, measured).items():
                        measured[new]['track_id'] = predicted[old]['track_id']
                    chosen, status = measured, 'reacquired'
                    self.pending, self.pending_count = [], 0
                else:
                    chosen, status = predicted, 'pending'
        chosen = [dict(c) for c in chosen]
        for cell in chosen:
            if 'track_id' not in cell:
                cell['track_id'] = self.next_id
                self.next_id += 1
            cell['grid_tracked'] = True
        evidence, assigned = self._assign(chosen, dots)
        if assigned != len(dots) or not all(evidence):
            status = 'pending'  # A partial fit must never confirm a partial word.
        output = []
        for cell, indices in zip(chosen, evidence):
            if not indices:
                continue  # Never decode/draw an absent or unpainted cell.
            slots = np.asarray([cell['grid']['slots'][i] for i in range(1, 7)])
            pattern = frozenset(int(np.linalg.norm(slots-dots[i]['center'], axis=1).argmin())+1
                                for i in indices)
            output.append(dict(cell, source_dot_ids=indices, dots=pattern))
        self.cells, self.reference = chosen, gray
        self.current_dots = dots
        self.mask = cell_motion_mask(chosen, image.shape, gray.shape)
        self.context, self.updated_at = context, now
        return output, status
