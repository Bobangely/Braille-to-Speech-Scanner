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


def split_red_components(dots, mask, *, max_dots=1536):
    """Split touching red ink only at measured lobes with an intervening valley.

    A fixed fraction of the largest lobe can join unequal cores or omit a
    smaller one. Local distance maxima preserve their separate evidence. This
    runs before cell planning and does not rely on any proposed grid slots.
    """
    if len(dots) < 4 or len(dots) > max_dots:
        return dots
    points = np.asarray([d['center'] for d in dots])
    spacing = _spacing(points, np.sqrt([d['area'] for d in dots]))
    if spacing is None or spacing < 2:
        return dots
    radius = max(1, round(.35*spacing))
    # Rectangular max filtering is separable/fast. Its diagonal reach stays
    # below the minimum lobe separation used below (.7 * spacing).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*radius+1, 2*radius+1))
    refined = []
    for dot in dots:
        x1, y1, x2, y2 = dot['bbox']
        if max(x2-x1, y2-y1) < .7*spacing:
            refined.append(dot)
            continue
        patch = cv2.copyMakeBorder(mask[y1:y2, x1:x2], 1, 1, 1, 1,
                                   cv2.BORDER_CONSTANT, value=0)
        # An irregular parent bbox can enclose a separate neighbouring dot.
        # Match the original component's measured area and centroid, then keep
        # only its pixels: a neighbour must neither veto nor join this split.
        if cv2.countNonZero(patch) != dot['area']:
            _, labels, stats, centers = cv2.connectedComponentsWithStats(patch)
            parent_center = np.asarray(dot['center']) - (x1-1, y1-1)
            parents = [i for i in range(1, len(stats))
                       if stats[i, cv2.CC_STAT_AREA] == dot['area']
                       and np.allclose(centers[i], parent_center, rtol=0, atol=1e-6)]
            if len(parents) != 1:
                refined.append(dot)
                continue
            patch = (labels == parents[0]).astype(np.uint8)*255
        distance = cv2.distanceTransform(patch, cv2.DIST_L2, 5)
        maxima = ((distance >= cv2.dilate(distance, kernel)-1e-6)
                  & (distance >= max(1., .35*float(distance.max()))))
        peak_y, peak_x = np.nonzero(maxima)
        # If all maxima fit within one separation radius, two valid lobes are
        # impossible. Skip component labelling for the common single-dot case.
        if not len(peak_x) or np.hypot(np.ptp(peak_x), np.ptp(peak_y)) < .7*spacing:
            refined.append(dot)
            continue
        _, _, _, centers = cv2.connectedComponentsWithStats(maxima.astype(np.uint8))
        # A rough flat summit may have several equal maxima. Keep one per lobe,
        # using the same minimum dot separation as the existing BLUE splitter.
        ranked = sorted(centers[1:], key=lambda p: (-distance[round(p[1]), round(p[0])], p[1], p[0]))
        peaks = []
        for center in ranked:
            if all(np.linalg.norm(center-peak) >= .7*spacing for peak in peaks):
                peaks.append(center)
        if len(peaks) < 2 or len(peaks) > 6:
            refined.append(dot)
            continue
        # Every lobe must be connected to another supported lobe through a thin
        # neck; a broad, solid smudge must not turn into fabricated dots.
        edges = [set() for _ in peaks]
        for i, first in enumerate(peaks):
            for j, second in enumerate(peaks[i+1:], i+1):
                if np.linalg.norm(first-second) > 1.5*spacing:
                    continue
                midpoint = (first+second)/2
                depth = min(distance[round(p[1]), round(p[0])] for p in (first, second))
                if distance[round(midpoint[1]), round(midpoint[0])] < .7*depth:
                    edges[i].add(j)
                    edges[j].add(i)
        reached, pending = set(), [0]
        while pending:
            current = pending.pop()
            if current not in reached:
                reached.add(current)
                pending.extend(edges[current]-reached)
        if len(reached) != len(peaks):
            refined.append(dot)
            continue
        yy, xx = np.nonzero(patch)
        pixels = np.column_stack((xx, yy))
        labels = np.linalg.norm(pixels[:, None]-np.asarray(peaks)[None], axis=2).argmin(axis=1)
        parts = []
        for index, peak in enumerate(peaks):
            part = pixels[labels == index]
            if len(part) < 6 or np.linalg.norm(part.mean(axis=0)-peak) > .35*spacing:
                break
            part = part + (x1-1, y1-1)
            low, high = part.min(axis=0), part.max(axis=0)+1
            width, height = high-low
            if max(width, height) > 3*min(width, height) or len(part)/(width*height) < .2:
                break
            parts.append(dict(dot, center=tuple(part.mean(axis=0)), area=float(len(part)),
                              bbox=tuple(map(int, (*low, *high))), color_pixels=len(part),
                              split_from_bbox=dot['bbox']))
        refined.extend(parts if len(parts) == len(peaks) else [dot])
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
        # Estimate scale from marks that already pass the original safety
        # limit. At small scales quantisation can leave a real dot with 4-5
        # pixels while its neighbours have 6-7; do not discard it before the
        # relative/local ink filter can examine it. Isolated one-pixel noise
        # cannot lower this limit, and normal-size pages keep the old floor.
        reference_areas = [area for x, y, width, height, area in stats[1:]
                           if self.min_area <= area <= self.max_area
                           and max(width, height) <= 3*min(width, height)
                           and area/(width*height) >= .2]
        min_area = self.min_area
        if len(reference_areas) >= 4:
            min_area = min(self.min_area, max(2., .12*float(np.median(reference_areas))))
        dots = []
        for index in range(1, count):
            x, y, width, height, area = map(int, stats[index])
            if not min_area <= area <= self.max_area or area/(width*height) < .2:
                continue
            dots.append(dict(center=tuple(map(float, centers[index])), area=float(area),
                             bbox=(x, y, x+width, y+height), confidence=1.0,
                             source='color', color_pixels=area))
        if min_area < self.min_area and len(dots) > 6*self.max_cells:
            raise UnreadableBrailleFrame('Too many small colored components for scale estimation')
        # Reject small speckles relative to actual ink marks while retaining partial dots.
        normal_areas = [d['area'] for d in dots if
            max(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1]) <=
            3*min(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1])]
        if len(normal_areas) >= 4:
            floor = max(min_area, .12*float(np.median(normal_areas)))
            # Ink size may differ between text lines. Only reconsider marks
            # below the page-wide floor when nearby dots support their size.
            if any(dot['area'] < max(floor, self.min_area) for dot in dots):
                points = np.asarray([dot['center'] for dot in dots])
                areas = np.asarray([dot['area'] for dot in dots])
                spacing = _spacing(points, np.sqrt(areas))
                if spacing is not None:
                    keep = []
                    for index, dot in enumerate(dots):
                        if dot['area'] >= max(floor, self.min_area):
                            keep.append(dot)
                            continue
                        distance = np.linalg.norm(points-points[index], axis=1)
                        neighbours = areas[distance <= 2.5*spacing]
                        # Only rescue few-pixel marks in a supported
                        # neighbourhood. An isolated 2-5 pixel speck must not
                        # become a new cell just because the page is small.
                        if dot['area'] < self.min_area and len(neighbours) < 3:
                            continue
                        # The candidate plus two nearby marks is enough to
                        # support a sparse cell at the end of a small-dot line.
                        local_floor = (max(min_area, .12*float(np.median(neighbours)))
                                       if len(neighbours) >= 3 else floor)
                        if dot['area'] >= local_floor:
                            keep.append(dot)
                    dots = keep
        # A component clipped by the camera boundary is not a reliable dot.
        # Also discard only tiny nearby fragments of the same off-frame mark,
        # using the measured dot diameter instead of a frame-specific threshold.
        regular_sizes = [min(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1])
                         for d in dots if
                         max(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1]) <=
                         3*min(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1])]
        if regular_sizes:
            height, width = image.shape[:2]
            typical_size = float(np.median(regular_sizes))
            edge_margin = max(2., 1.5*typical_size)
            def clipped_or_near_edge(dot):
                x1, y1, x2, y2 = dot['bbox']
                min_size = min(x2-x1, y2-y1)
                clipped = x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height
                tiny_fragment = (min_size < .5*typical_size and
                                 (x1 < edge_margin or y1 < edge_margin or
                                  width-x2 < edge_margin or height-y2 < edge_margin))
                return clipped or tiny_fragment
            dots = [dot for dot in dots if not clipped_or_near_edge(dot)]
        # A joined pair can be wider than one dot. Apply the original aspect
        # limit AFTER attempting a pixel-supported split; unsplit lines stay out.
        splitter = split_red_components if self.color == 'red' else split_color_components
        dots = splitter(dots, mask, max_dots=6*self.max_cells)
        dots = [d for d in dots if max(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1]) <=
                3*min(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1])]
        split_parts = [tuple(d['split_from_bbox']) for d in dots if 'split_from_bbox' in d]
        split_count = len(set(split_parts))
        accepted_components = len(dots)-len(split_parts)+split_count
        return dots, dict(mask=mask, color=self.color, color_components=count-1,
                          colored_dots=len(dots), rejected_color_components=count-1-accepted_components,
                          split_color_components=split_count,
                          color_config=dict(hue_ranges=HUE_RANGES.get(self.color, ((0, 179),)),
                              saturation_min=60, value_min=15, min_area=min_area,
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
