"""Read only chromatic Braille dots; unpainted relief never proposes a cell."""

import cv2
import numpy as np

from cell_tracking import CellGridTracker
from yolo_cell_stream import UnreadableBrailleFrame, plan_cells, read_cell, pair_markers, _spacing


DOT_COLORS = ('blue', 'red')
# OpenCV HSV hue is 0..179. Neutral paper/shadows are excluded by saturation.
HUE_RANGES = {'blue': ((100, 145),), 'red': ((0, 15), (165, 179))}


def split_color_components(dots, mask, *, max_dots=1536):
    """Separate two ink lobes joined by a thin neck before grid acquisition.

    Both cores must exist in the current mask, at a measured dot spacing.
    A solid wide mark has one core and must never invent a second dot.
    """
    if len(dots) > max_dots:
        return dots  # Preserve the detector's frame budget before pairwise fitting.
    reference = [d for d in dots if
        max(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1]) <=
        3*min(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1])]
    if len(reference) < 4:
        return dots
    typical = float(np.median([d['area'] for d in reference]))
    if not any(dot['area'] >= 1.5*typical for dot in dots):
        return dots
    points = np.asarray([d['center'] for d in reference])
    areas = np.asarray([d['area'] for d in reference])
    spacing = _spacing(points, np.sqrt(areas))
    if spacing is None or spacing < 2:
        return dots
    refined = []
    for dot in dots:
        if dot['area'] < 1.5*typical:
            refined.append(dot)
            continue
        x1, y1, x2, y2 = dot['bbox']
        patch = np.pad(mask[y1:y2, x1:x2], 1)
        # Do not include a different component enclosed by an irregular bbox.
        if cv2.connectedComponents(patch)[0] != 2:
            refined.append(dot)
            continue
        distance = cv2.distanceTransform(patch, cv2.DIST_L2, 5)
        cores = (distance >= .55*float(distance.max())).astype(np.uint8)
        count, _, stats, centers = cv2.connectedComponentsWithStats(cores)
        if (count != 3 or min(stats[1:, cv2.CC_STAT_AREA]) < .1*typical or
                not .7*spacing <= np.linalg.norm(centers[1]-centers[2]) <= 1.5*spacing):
            refined.append(dot)
            continue
        yy, xx = np.nonzero(patch)
        pixels = np.column_stack((xx, yy))
        nearest = np.linalg.norm(pixels[:, None]-centers[None, 1:], axis=2).argmin(axis=1)
        # Paint size can differ between lines. Judge the two pieces against
        # neighbouring marks, not only the whole page's median ink area.
        nearby = np.linalg.norm(points-dot['center'], axis=1)
        neighbours = areas[(nearby > 0) & (nearby < 4*spacing)]
        local_area = float(np.median(neighbours)) if len(neighbours) >= 3 else typical
        parts = []
        for index in range(2):
            part = pixels[nearest == index] + (x1-1, y1-1)
            if not .35*local_area <= len(part) <= 1.6*local_area:
                break
            low, high = part.min(axis=0), part.max(axis=0)+1
            width, height = high-low
            if max(width, height) > 3*min(width, height) or len(part)/(width*height) < .2:
                break
            parts.append(dict(dot, center=tuple(part.mean(axis=0)), area=float(len(part)),
                bbox=tuple(map(int, (*low, *high))), color_pixels=len(part),
                split_from_bbox=dot['bbox']))
        refined.extend(parts if len(parts) == 2 else [dot])
    return refined


class ColoredCellStream:
    def __init__(self, color='blue', max_cells=256, min_area=6, max_area=8000):
        if color not in DOT_COLORS:
            raise ValueError(f'Unsupported dot color: {color}')
        if max_cells < 1 or min_area < 1 or max_area < min_area:
            raise ValueError('Invalid colored-dot or cell limits')
        self.color, self.max_cells = color, max_cells
        self.min_area, self.max_area = min_area, max_area
        self._grid_tracker = CellGridTracker()

    def find_dots(self, image):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
                image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
            raise UnreadableBrailleFrame('Expected a nonempty BGR uint8 image')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 60, 15), (179, 255, 255))
        hue_mask = np.zeros_like(mask)
        for low, high in HUE_RANGES[self.color]:
            hue_mask |= cv2.inRange(hsv[:, :, 0], low, high)
        mask &= hue_mask
        # Closing joins small breaks in painted edges; no opening that erases small dots.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        dots = []
        for index in range(1, count):
            x, y, width, height, area = map(int, stats[index])
            if not self.min_area <= area <= self.max_area or area/(width*height) < .2:
                continue
            dots.append(dict(center=tuple(map(float, centers[index])), area=float(area),
                             bbox=(x, y, x+width, y+height), confidence=1.0,
                             source='color', color_pixels=area))
        # Reject small speckles relative to actual ink marks while retaining partial dots.
        normal_areas = [d['area'] for d in dots if
            max(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1]) <=
            3*min(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1])]
        if len(normal_areas) >= 4:
            floor = max(self.min_area, .12*float(np.median(normal_areas)))
            dots = [dot for dot in dots if dot['area'] >= floor]
        # A joined pair can be wider than one dot. Apply the original aspect
        # limit AFTER attempting a pixel-supported split; unsplit lines stay out.
        dots = split_color_components(dots, mask, max_dots=6*self.max_cells)
        dots = [d for d in dots if max(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1]) <=
                3*min(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1])]
        split_count = sum('split_from_bbox' in d for d in dots)//2
        accepted_components = len(dots)-split_count
        return dots, dict(mask=mask, color=self.color, color_components=count-1,
                          colored_dots=len(dots), rejected_color_components=count-1-accepted_components,
                          split_color_components=split_count,
                          color_config=dict(hue_ranges=HUE_RANGES.get(self.color, ((0, 179),)),
                              saturation_min=60, value_min=15, min_area=self.min_area,
                              max_area=self.max_area))

    def detect(self, image, lang='thai', *, context=None, roi=None):
        # ROI changes never change the tracker's coordinate system. Measure
        # color in the crop, then plan/track/read in full source coordinates.
        if roi is None:
            dots, diagnostics = self.find_dots(image)
        else:
            x1, y1, x2, y2 = roi
            dots, diagnostics = self.find_dots(image[y1:y2, x1:x2])
            dots = [dict(dot, center=(dot['center'][0]+x1, dot['center'][1]+y1),
                         bbox=tuple(v+(x1 if i % 2 == 0 else y1) for i, v in enumerate(dot['bbox'])))
                    for dot in dots]
            mask = np.zeros(image.shape[:2], np.uint8)
            mask[y1:y2, x1:x2] = diagnostics['mask']
            diagnostics['mask'] = mask
        if len(dots) > 6 * self.max_cells:
            raise UnreadableBrailleFrame('Too many colored dots; narrow the camera ROI')
        # FILTER precedes geometry: gray cells cannot create rows, candidates, or '?'.
        planning_error = None
        try:
            planned = plan_cells(dots, image.shape)
        except UnreadableBrailleFrame as exc:
            if context is None:
                raise
            planned, planning_error = [], exc
        if len(planned) > self.max_cells:
            raise UnreadableBrailleFrame(f'{len(planned)} colored cells exceed limit {self.max_cells}')
        if context is not None:
            candidate_count = len(planned)
            planned, tracking = self._grid_tracker.update(planned, dots, image, (context, lang, self.color),
                                                         diagnostics.get('mask'))
            dots = self._grid_tracker.current_dots
            if planning_error is not None and not planned:
                raise planning_error
            diagnostics.update(grid_tracking=tracking, grid_pending=tracking == 'pending',
                               refined_colored_dots=len(dots),
                               grid_candidate_count=candidate_count,
                               grid_planning_error=str(planning_error) if planning_error else None)
        else:
            self._grid_tracker.reset()  # Image CLI remains stateless.
        cells, symbols = [], []

        def read_colored_cells():
            for cell in planned:
                evidence = [dots[index] for index in cell['source_dot_ids']]
                if not evidence:
                    continue
                # Match the measured colored centers directly; no second YOLO classification.
                result = read_cell(cell, evidence, np.eye(3, dtype=np.float64))
                result['colored_dot_count'] = len(evidence)
                result['read_source'] = 'color'
                # Missing/unpainted cells cannot make widely separated markers adjacent.
                # This bound is scale-relative and still permits wide printed cell spacing.
                result['adjacency_limit'] = 4.2*cell['dot_spacing']
                yield result

        for group in pair_markers(read_colored_cells(), lang):
            start = len(cells)
            cells.extend(group)
            symbols.append(dict(cell_start=start, cell_end=len(cells)-1,
                                line_id=group[0]['line_id'], symbol=group[0].get('symbol')))
        read_dots = [dot for cell in cells for dot in cell['read_dots']]
        return cells, dict(method='colored_cell_stream', dots=read_dots, symbols=symbols,
            candidate_source='color', read_source='color', yolo_inference=False,
            num_detections=len(read_dots), overview_detections=len(dots),
            line_count=len({cell['line_id'] for cell in cells}), crop_count=len(cells),
            crop_disagreements=sum(cell['crop_disagreement'] for cell in cells),
            **diagnostics)
