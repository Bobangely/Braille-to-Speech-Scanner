"""Read only chromatic Braille dots; unpainted relief never proposes a cell."""

import cv2
import numpy as np

from yolo_cell_stream import UnreadableBrailleFrame, plan_cells, read_cell, pair_markers


DOT_COLORS = ('blue', 'red')
# OpenCV HSV hue is 0..179. Neutral paper/shadows are excluded by saturation.
HUE_RANGES = {'blue': ((100, 145),), 'red': ((0, 15), (165, 179))}


class ColoredCellStream:
    def __init__(self, color='blue', max_cells=256, min_area=6, max_area=8000):
        if color not in DOT_COLORS:
            raise ValueError(f'Unsupported dot color: {color}')
        if max_cells < 1 or min_area < 1 or max_area < min_area:
            raise ValueError('Invalid colored-dot or cell limits')
        self.color, self.max_cells = color, max_cells
        self.min_area, self.max_area = min_area, max_area

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
            if (not self.min_area <= area <= self.max_area or
                    max(width, height) > 3*min(width, height) or area/(width*height) < .2):
                continue
            dots.append(dict(center=tuple(map(float, centers[index])), area=float(area),
                             bbox=(x, y, x+width, y+height), confidence=1.0,
                             source='color', color_pixels=area))
        # Reject small speckles relative to actual ink marks while retaining partial dots.
        if len(dots) >= 4:
            floor = max(self.min_area, .12*float(np.median([dot['area'] for dot in dots])))
            dots = [dot for dot in dots if dot['area'] >= floor]
        return dots, dict(color=self.color, color_components=count-1,
                          colored_dots=len(dots), rejected_color_components=count-1-len(dots),
                          color_config=dict(hue_ranges=HUE_RANGES.get(self.color, ((0, 179),)),
                              saturation_min=60, value_min=15, min_area=self.min_area,
                              max_area=self.max_area))

    def detect(self, image, lang='thai'):
        dots, diagnostics = self.find_dots(image)
        if len(dots) > 6 * self.max_cells:
            raise UnreadableBrailleFrame('Too many colored dots; narrow the camera ROI')
        # FILTER precedes geometry: gray cells cannot create rows, candidates, or '?'.
        planned = plan_cells(dots, image.shape)
        if len(planned) > self.max_cells:
            raise UnreadableBrailleFrame(f'{len(planned)} colored cells exceed limit {self.max_cells}')
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
