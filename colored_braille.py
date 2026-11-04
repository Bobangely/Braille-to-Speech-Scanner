"""Read only chromatic Braille dots; unpainted relief never proposes a cell."""

import cv2
import numpy as np


# OpenCV HSV hue is 0..179. Neutral paper/shadows are excluded by saturation.
BLUE_HUE = (100, 145)


class UnreadableBrailleFrame(ValueError):
    """The image cannot provide a reliable cell grid; retry with a new frame."""


def _spacing(points, diameters):
    """ระยะห่างจุดโดยประมาณ: ควอร์ไทล์ที่ 25 ของระยะถึงจุดที่ใกล้ที่สุด"""
    if len(points) < 2:
        return None
    nearest = []
    for start in range(0, len(points), 128):
        delta = points[start:start+128, None, :] - points[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        # Residual duplicate centres inside a dot cannot define grid pitch.
        within_dot = .8*np.minimum(diameters[start:start+128, None], diameters[None, :])
        distances[(distances < 1.0) | (distances < within_dot)] = np.inf
        nearest.extend(distances.min(axis=1))
    finite = np.asarray(nearest)[np.isfinite(nearest)]
    return float(np.percentile(finite, 25)) if len(finite) else None


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
    def __init__(self, max_cells=256, min_area=6, max_area=8000):
        if max_cells < 1 or min_area < 1 or max_area < min_area:
            raise ValueError('Invalid colored-dot or cell limits')
        self.max_cells = max_cells
        self.min_area, self.max_area = min_area, max_area

    def find_dots(self, image):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
                image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
            raise UnreadableBrailleFrame('Expected a nonempty BGR uint8 image')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 60, 15), (179, 255, 255))
        mask &= cv2.inRange(hsv[:, :, 0], *BLUE_HUE)
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
        dots = split_color_components(dots, mask, max_dots=6*self.max_cells)
        dots = [d for d in dots if max(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1]) <=
                3*min(d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1])]
        split_parts = [tuple(d['split_from_bbox']) for d in dots if 'split_from_bbox' in d]
        split_count = len(set(split_parts))
        accepted_components = len(dots)-len(split_parts)+split_count
        return dots, dict(mask=mask, color_components=count-1,
                          colored_dots=len(dots), rejected_color_components=count-1-accepted_components,
                          split_color_components=split_count,
                          color_config=dict(hue_ranges=(BLUE_HUE,),
                              saturation_min=60, value_min=15, min_area=min_area,
                              max_area=self.max_area))
