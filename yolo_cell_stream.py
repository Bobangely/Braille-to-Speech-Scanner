"""YOLO-guided line/cell crops, isolated dot reads, and ordered cell streaming.

The supplied weights detect dots, not whole cells. Geometry proposes crop boxes
from those YOLO dots. A second YOLO pass reads each crop; no colour-mask union or
global CV cell clustering is used. Coordinates and line IDs survive every stage.
"""

from itertools import islice
import math

import cv2
import numpy as np


class UnreadableBrailleFrame(ValueError):
    """The image cannot provide a reliable cell grid; retry with a new frame."""


def _clusters(values, tolerance):
    """Bounded cluster span, avoiding transitive chains between rows."""
    groups = []
    for index in np.argsort(values):
        if not groups or values[index] - values[groups[-1][0]] > tolerance:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    return groups


def _spacing(points, diameters=None):
    if len(points) < 2:
        return None
    nearest = []
    for start in range(0, len(points), 128):
        delta = points[start:start+128, None, :] - points[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        distances[distances < 1.0] = np.inf
        if diameters is not None:
            # Residual duplicate centres inside a dot cannot define grid pitch.
            within_dot = .8*np.minimum(diameters[start:start+128, None], diameters[None, :])
            distances[distances < within_dot] = np.inf
        nearest.extend(distances.min(axis=1))
    finite = np.asarray(nearest)[np.isfinite(nearest)]
    return float(np.percentile(finite, 25)) if len(finite) else None


def _angle(points, spacing):
    angles = []
    for start in range(0, len(points), 128):
        delta = points[None, :, :] - points[start:start+128, None, :]
        dx, dy = delta[..., 0], delta[..., 1]
        valid = (dx > .7*spacing) & (dx < 2.8*spacing) & (np.abs(dy) < .25*dx)
        angles.extend(np.arctan2(dy[valid], dx[valid]))
    return float(np.median(angles)) if angles else 0.0


def _cell_pitch(column_centers, anchors, spacing):
    if len(anchors) < 2:
        return 2.375 * spacing
    differences = np.diff(anchors)
    candidates = [2.375 * spacing]
    for difference in differences:
        candidates.extend(difference / divisor for divisor in range(1, 5)
                          if 1.8 * spacing <= difference / divisor <= 6 * spacing)
    def score(pitch):
        offsets = (column_centers - anchors[0]) % pitch
        residual = np.minimum(np.minimum(offsets, pitch-offsets), np.abs(offsets-spacing))
        return (float(np.mean(np.minimum(residual / spacing, 1)**2)), -pitch)
    return min(candidates, key=score)


def _line_rows(rows, centers, spacing):
    """Separate incompatible row gaps and reject runs with no clear line phase."""
    gaps = np.diff(centers)
    nearby = gaps[(gaps >= .65*spacing) & (gaps <= 1.5*spacing)]
    row_pitch = float(np.median(nearby)) if len(nearby) else spacing
    bands = []
    for row, center in zip(rows, centers):
        gap = (center-bands[-1][-1][0])/row_pitch if bands else None
        if gap is None or gap > 2.15 or abs(gap-round(gap)) > .15:
            bands.append([])
        bands[-1].append((center, row))

    lines = []
    for band in bands:
        extent = int(round((band[-1][0]-band[0][0])/row_pitch)) + 1
        if extent > 3 and extent % 3:
            raise UnreadableBrailleFrame('Ambiguous Braille line boundaries: include complete '
                             'three-row cells or isolate a line in the camera ROI')
        current = []
        for item in band:
            if current and item[0]-current[0][0] > 2.35*row_pitch:
                lines.append(current)
                current = []
            current.append(item)
        lines.append(current)
    return lines, row_pitch


def plan_cells(dots, image_shape):
    """Split rows into lines of at most three rows BEFORE assigning any cell.

    Line crops are clipped at separators, even when the inter-line gap is less
    than the old 2.2-dot-spacing threshold. A source dot belongs to one line only.
    """
    if not dots:
        return []
    points = np.asarray([d['center'] for d in dots], dtype=float)
    diameters = np.sqrt([max(0, dot.get('area', 0)) for dot in dots])
    spacing = _spacing(points, diameters)
    if spacing is None or spacing < 2:
        return []  # an isolated dot has no observable 2x3 reference grid
    angle = _angle(points, spacing)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    rectified = points @ rotation
    rows = _clusters(rectified[:, 1], .45 * spacing)
    row_centers = [float(np.median(rectified[row, 1])) for row in rows]
    lines, row_pitch = _line_rows(rows, row_centers, spacing)

    cells = []
    for line_id, line in enumerate(lines):
        indices = [i for _, group in line for i in group]
        observed_rows = [v for v, _ in line]
        if len(line) == 3:
            dy = (observed_rows[-1] - observed_rows[0]) / 2
            expected_rows = [observed_rows[0] + i*dy for i in range(3)]
        elif len(line) == 2:
            gap = observed_rows[1]-observed_rows[0]
            dy = gap/2 if gap > 1.5*row_pitch else gap
            expected_rows = [observed_rows[0] + i*dy for i in range(3)]
        else:
            dy = row_pitch
            expected_rows = [observed_rows[0] + i*dy for i in range(3)]
        lower = expected_rows[0] - .45*dy
        upper = expected_rows[-1] + .45*dy
        if line_id:
            lower = max(lower, (lines[line_id-1][-1][0]+observed_rows[0])/2)
        if line_id+1 < len(lines):
            upper = min(upper, (observed_rows[-1]+lines[line_id+1][0][0])/2)
        local_x = rectified[indices, 0]
        column_groups = _clusters(local_x, .45*spacing)
        columns = np.asarray([np.median(local_x[group]) for group in column_groups])
        groups = []
        for i, x in enumerate(columns):
            if not groups or len(groups[-1]) == 2 or x-columns[groups[-1][-1]] > 1.22*spacing:
                groups.append([])
            groups[-1].append(i)
        pair_widths = [columns[g[1]]-columns[g[0]] for g in groups if len(g) == 2]
        dx = float(np.median(pair_widths)) if pair_widths else spacing
        anchors = [float(columns[g[0]]) for g in groups if len(g) == 2]
        pitch = _cell_pitch(columns, anchors, dx)

        for line_cell_index, group in enumerate(groups):
            column_values = columns[group]
            column_ambiguous = False
            if len(group) == 2:
                left, right = map(float, column_values)
            else:
                x = float(column_values[0])
                if anchors:
                    anchor = min(anchors, key=lambda a: abs(a-x))
                    def error(left):
                        delta = anchor-left
                        return abs(delta-round(delta/pitch)*pitch)
                    errors = [error(x), error(x-dx)]
                    column_ambiguous = abs(errors[0]-errors[1]) < .15*dx
                    is_right = errors[1] < errors[0]
                    # With only one anchor, a prefix at the beginning has no
                    # measurable character pitch. Keep this uncertainty explicit.
                    if len(anchors) == 1:
                        column_ambiguous = True
                else:
                    is_right, column_ambiguous = False, True
                left, right = (x-dx, x) if is_right else (x, x+dx)
            source_ids = [indices[i] for column in group for i in column_groups[column]]
            slot_uv = np.array([(left if d < 3 else right, expected_rows[d % 3]) for d in range(6)])
            source_slots = slot_uv @ rotation.T
            overview_pattern = set()
            for index in source_ids:
                delta = (rectified[index]-slot_uv) / np.array([dx, dy])
                nearest = int(np.argmin(np.linalg.norm(delta, axis=1)))
                overview_pattern.add(nearest+1)
            quad_uv = np.array([[left-.45*dx, lower], [right+.45*dx, lower],
                                [right+.45*dx, upper], [left-.45*dx, upper]])
            if line_cell_index:
                separator = (columns[groups[line_cell_index-1][-1]] + column_values[0]) / 2
                quad_uv[[0, 3], 0] = np.maximum(quad_uv[[0, 3], 0], separator)
            if line_cell_index+1 < len(groups):
                separator = (column_values[-1] + columns[groups[line_cell_index+1][0]]) / 2
                quad_uv[[1, 2], 0] = np.minimum(quad_uv[[1, 2], 0], separator)
            quad = quad_uv @ rotation.T
            center = tuple(np.mean(source_slots, axis=0))
            bbox = (max(0, int(np.floor(quad[:, 0].min()))), max(0, int(np.floor(quad[:, 1].min()))),
                    min(image_shape[1], int(np.ceil(quad[:, 0].max()))),
                    min(image_shape[0], int(np.ceil(quad[:, 1].max()))))
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            cells.append(dict(dots=frozenset(overview_pattern), center=center,
                x=center[0], y=center[1], line_id=line_id, line_cell_index=line_cell_index,
                reading_x=float((left+right)/2), dot_spacing=dx, cell_pitch=pitch,
                pitch_ambiguous=len(anchors) < 2,
                column_ambiguous=column_ambiguous, row_ambiguous=(len(line) == 1 or (len(line) == 2 and observed_rows[-1]-observed_rows[0] < 1.5*row_pitch)),
                source_dot_ids=source_ids, crop_quad=quad.tolist(),
                grid=dict(expected_cols=[float(source_slots[1, 0]), float(source_slots[4, 0])],
                          expected_rows=[float(np.mean(source_slots[[r, r+3], 1])) for r in range(3)],
                          slots={d+1: tuple(p) for d, p in enumerate(source_slots)}, bbox=bbox)))
    return cells


def crop_cell(image, cell, canvas_size=320):
    """Warp the isolated ROI at a fixed dot scale; white padding prevents giant dots.

    Returns the image for YOLO and its transform back to source-image coordinates.
    """
    quad = np.asarray(cell['crop_quad'], np.float32)
    scale = 32.0 / cell['dot_spacing']
    width = max(2, int(round(np.linalg.norm(quad[1]-quad[0])*scale)))
    height = max(2, int(round(np.linalg.norm(quad[3]-quad[0])*scale)))
    shrink = min(1.0, (canvas_size-8)/max(width, height))
    width, height = max(2, round(width*shrink)), max(2, round(height*shrink))
    x, y = (canvas_size-width)//2, (canvas_size-height)//2
    destination = np.float32([[x, y], [x+width-1, y], [x+width-1, y+height-1], [x, y+height-1]])
    transform = cv2.getPerspectiveTransform(quad, destination)
    patch = cv2.warpPerspective(image, transform, (canvas_size, canvas_size),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    # warpPerspective can sample outside the ROI; erase that context explicitly.
    patch[:y] = 255
    patch[y+height:] = 255
    patch[:, :x] = 255
    patch[:, x+width:] = 255
    return patch, np.linalg.inv(transform)


def read_cell(cell, crop_dots, inverse):
    """Assign only this crop's YOLO detections to this cell's six slots."""
    slots = np.asarray(list(cell['grid']['slots'].values()))
    accepted = {}
    detections = outside_grid = duplicate_slots = 0
    for dot in crop_dots:
        detections += 1
        source = cv2.perspectiveTransform(np.float32([[dot['center']]]), inverse)[0, 0]
        distances = np.linalg.norm(slots-source, axis=1)
        index = int(np.argmin(distances))
        if distances[index] > .38*cell['dot_spacing']:
            outside_grid += 1
            continue
        if index in accepted:
            duplicate_slots += 1
            if accepted[index]['confidence'] >= dot['confidence']:
                continue
        accepted[index] = dict(dot, center=tuple(map(float, source)),
                               area=(cell['dot_spacing']*.5)**2, bbox=None)
    result = dict(cell, overview_dots=cell['dots'], dots=frozenset(i+1 for i in accepted),
                  read_pattern=frozenset(i+1 for i in accepted),
                  crop_status='read' if accepted else 'empty',
                  crop_diagnostics=dict(detections=detections, matched_slots=len(accepted),
                      outside_grid=outside_grid, duplicate_slots=duplicate_slots),
                  read_dots=list(accepted.values()))
    result['crop_disagreement'] = result['dots'] != result['overview_dots']
    return result


class CellStream:
    def __init__(self, predict_crops, batch_size=8, max_cells=256):
        if not 1 <= batch_size <= 32 or max_cells < 1:
            raise ValueError('crop batch must be 1..32; max_cells must be positive')
        self.predict_crops, self.batch_size, self.max_cells = predict_crops, batch_size, max_cells

    def iter_cells(self, image, overview_dots):
        planned = plan_cells(overview_dots, image.shape)
        if len(planned) > self.max_cells:
            raise UnreadableBrailleFrame(f'{len(planned)} cell candidates exceed limit {self.max_cells}; narrow the camera ROI')
        iterator = iter(planned)
        while batch := list(islice(iterator, self.batch_size)):
            crops = [crop_cell(image, cell) for cell in batch]
            predictions = list(self.predict_crops([patch for patch, _ in crops]))
            if len(predictions) != len(batch):
                raise RuntimeError('YOLO crop result count does not match submitted cells')
            for cell, dots, (_, inverse) in zip(batch, predictions, crops):
                yield read_cell(cell, dots, inverse)


def pair_markers(cells, lang='thai'):
    """One-cell lookahead: emit marker+body together, never across a line or gap."""
    from config_thai import THAI_MULTI_CELL
    iterator = iter(cells)
    current = next(iterator, None)
    while current is not None:
        following = next(iterator, None)
        if following is None:
            yield [current]
            break
        distance = following['reading_x']-current['reading_x']
        adjacent = (current['line_id'] == following['line_id']
                    and 0 < distance <= 1.5*max(current['cell_pitch'], following['cell_pitch']))
        if (lang.lower() in ('thai', 'th') and adjacent
                and not current.get('row_ambiguous') and not following.get('row_ambiguous')):
            # Never blindly turn a genuine dot-3 vowel into prefix-6. This repair
            # is restricted to a leading marker whose column phase is ambiguous.
            if (current['dots'] == frozenset({3}) and current['column_ambiguous']
                    and current['line_cell_index'] == 0
                    and (frozenset({6}), following['dots']) in THAI_MULTI_CELL):
                current = dict(current, dots=frozenset({6}), marker_repair='leading_3_to_6')
                slots = current['grid']['slots']
                shift = np.asarray(slots[4])-slots[1]
                adjusted = {key: tuple(np.asarray(point)-shift) for key, point in slots.items()}
                current['grid'] = dict(current['grid'], slots=adjusted,
                    expected_cols=[x-shift[0] for x in current['grid']['expected_cols']],
                    expected_rows=[y-shift[1] for y in current['grid']['expected_rows']],
                    bbox=tuple(max(0, int(round(value-shift[index % 2])))
                               for index, value in enumerate(current['grid']['bbox'])))
                current['x'] -= shift[0]
                current['y'] -= shift[1]
                current['center'] = (current['x'], current['y'])
                current['reading_x'] -= current['dot_spacing']
                # Once the leading prefix has a known phase, these two cells
                # provide a pitch measurement where there was only one anchor.
                if current.get('pitch_ambiguous') and following.get('pitch_ambiguous'):
                    measured_pitch = following['reading_x']-current['reading_x']
                    current['cell_pitch'] = measured_pitch
                    following = dict(following, cell_pitch=measured_pitch)
                # crop_quad and read_pattern retain the actual stage-2 input/read.
            symbol = THAI_MULTI_CELL.get((current['dots'], following['dots']))
            if symbol:
                current = dict(current, symbol=symbol, symbol_span=2)
                following = dict(following, symbol_continuation=True)
                yield [current, following]
                current = next(iterator, None)
                continue
        yield [current]
        current = following
