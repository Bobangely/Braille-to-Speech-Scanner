"""Optional YOLO-dot ROI proposals with bounded, color-independent tracking."""

import time

import cv2
import numpy as np

from frame_motion import estimate_motion, full_size_motion, tracking_gray
from yolo_cell_stream import UnreadableBrailleFrame


def translate_cells(cells, x, y):
    """Restore crop coordinates for the existing decoder, overlay, and exporter."""
    def point(value):
        return (float(value[0] + x), float(value[1] + y))

    def box(value):
        return tuple(float(v + (x if i % 2 == 0 else y)) for i, v in enumerate(value))

    translated = []
    for cell in cells:
        grid = cell['grid']
        dots = [dict(dot, center=point(dot['center']),
                     bbox=box(dot['bbox']) if dot.get('bbox') is not None else None)
                for dot in cell['read_dots']]
        translated.append(dict(cell, center=point(cell['center']), x=cell['x'] + x, y=cell['y'] + y,
            crop_quad=[point(p) for p in cell['crop_quad']], read_dots=dots,
            grid=dict(grid, bbox=box(grid['bbox']),
                expected_cols=[v + x for v in grid['expected_cols']],
                expected_rows=[v + y for v in grid['expected_rows']],
                slots={k: point(p) for k, p in grid['slots'].items()})))
    return translated


class ColoredRoiReader:
    """Use one padded text envelope; never use YOLO dots as decoded evidence.

    Existing weights detect dots, not lines. The envelope deliberately preserves
    all proposed lines and marker context until colored geometry splits them.
    Tracking always refers to the last YOLO frame, so errors do not accumulate.
    """

    def __init__(self, predict_dots, refresh_seconds=.35):
        self.predict_dots = predict_dots
        self.refresh_seconds = refresh_seconds
        self.reset()

    def reset(self):
        self.reference = self.reference_shape = self.box = None
        self.context = None
        self.detected_at = -float('inf')

    def detect(self, image, reader, lang='thai', context=None):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
                image.ndim != 3 or image.shape[2] != 3 or not image.size):
            raise UnreadableBrailleFrame('Expected a nonempty BGR frame for ROI detection')
        if self.context != context or self.reference_shape != image.shape:
            self.reset()
            self.context = context
        gray = tracking_gray(image)
        now = time.monotonic()
        box, source, predicted = None, 'none', False
        if self.box is not None and now - self.detected_at < self.refresh_seconds:
            motion = estimate_motion(self.reference, gray)
            if motion is not None:
                motion = full_size_motion(motion, image.shape, gray.shape)
                x1, y1, x2, y2 = self.box
                corners = np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
                corners = cv2.perspectiveTransform(corners, motion)[0]
                box = (*corners.min(axis=0), *corners.max(axis=0))
                source = 'tracked_yolo_dots'
        if box is None:
            # Invalidate before inference; an exception must not keep an old ROI alive.
            self.box = None
            dots = self.predict_dots(image)
            predicted = True
            if len(dots) >= 2:
                points = np.asarray([dot['center'] for dot in dots], dtype=float)
                diameter = float(np.median([np.sqrt(max(1, dot['area'])) for dot in dots]))
                padding = max(8., 3 * diameter)
                box = (*points.min(axis=0) - padding, *points.max(axis=0) + padding)
                self.box, self.reference, self.reference_shape = box, gray, image.shape
                self.detected_at = time.monotonic()
                source = 'yolo_dot_envelope'
        if box is None:
            return [], dict(method='colored_cell_stream', dots=[], color=reader.color,
                roi_source=source, roi_box=None, roi_status='not_found', yolo_inference=predicted,
                candidate_source='yolo_roi', read_source='color', crop_count=0, line_count=0)
        h, w = image.shape[:2]
        x1, y1 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
        x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
        if x2 <= x1 or y2 <= y1:
            self.reset()
            raise UnreadableBrailleFrame('Tracked ROI is outside the camera frame')
        cells, debug = reader.detect(image[y1:y2, x1:x2], lang)
        cells = translate_cells(cells, x1, y1)
        debug.update(dots=[dot for cell in cells for dot in cell['read_dots']],
            roi_source=source, roi_box=(x1, y1, x2, y2), roi_status='read',
            candidate_source='yolo_roi', yolo_inference=predicted)
        return cells, debug
