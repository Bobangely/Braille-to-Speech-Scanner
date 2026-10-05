"""Validate logical six-slot geometry independently of active dot count."""

import cv2
import numpy as np


def slot_points(cell):
    slots = {int(k): v for k, v in cell['grid']['slots'].items()}
    if set(slots) != set(range(1, 7)):
        raise ValueError('incomplete_grid')
    points = np.asarray([slots[i] for i in range(1, 7)], dtype=float)
    if points.shape != (6, 2) or not np.isfinite(points).all():
        raise ValueError('nonfinite_grid')
    return points


def grid_quad(points):
    a, b, c, d, e, f = points
    return np.array([a-.45*(d-a)-.45*(b-a), d+.45*(d-a)-.45*(e-d),
                     f+.45*(f-c)+.45*(f-e), c-.45*(f-c)+.45*(c-b)])


def grid_bbox(points, shape):
    quad = grid_quad(points)
    return (max(0, int(np.floor(quad[:, 0].min()))),
            max(0, int(np.floor(quad[:, 1].min()))),
            min(shape[1], int(np.ceil(quad[:, 0].max()))),
            min(shape[0], int(np.ceil(quad[:, 1].max()))))


def _lattice_error(points):
    corners = np.float32(points[[0, 3, 5, 2]])
    if not cv2.isContourConvex(corners) or cv2.contourArea(corners) <= 0:
        return 'collapsed_grid'
    transform = cv2.getPerspectiveTransform(corners, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
    normalized = cv2.perspectiveTransform(np.float32([points]), transform)[0]
    expected = np.array([[0, 0], [0, .5], [0, 1], [1, 0], [1, .5], [1, 1]])
    # Numerical allowance for float32 perspective conversion, not dot fitting.
    if not np.isfinite(normalized).all() or np.max(np.abs(normalized-expected)) > .002:
        return 'malformed_grid'
    return None


def _grid_errors(cell, shape=None):
    """Return ordered slots, lattice error and final error in a single pass."""
    points = None
    try:
        points = slot_points(cell)
        error = _lattice_error(points)
        if error:
            return points, error, error
        quad = np.asarray(cell['crop_quad'], np.float32)
        if quad.shape != (4, 2) or not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
            return points, None, 'invalid_crop'
        if any(cv2.pointPolygonTest(quad, tuple(map(float, p)), False) < -.001 for p in points):
            return points, None, 'clipped_grid'
        if shape is not None and (np.any(points < 0) or np.any(points[:, 0] >= shape[1])
                                  or np.any(points[:, 1] >= shape[0])):
            return points, None, 'off_frame_grid'
    except (KeyError, TypeError, ValueError, cv2.error):
        return points, 'incomplete_grid', 'incomplete_grid'
    return points, None, None


def grid_error(cell, shape=None):
    return _grid_errors(cell, shape)[2]


def validate_cell_grids(cells, shape, *, repair=True):
    """Repair lost logical slots from nearby reliable grids; never from dot bits."""
    measurements = [(cell, *_grid_errors(cell, shape)) for cell in cells]
    trusted = [(cell, points) for cell, points, _, error in measurements if error is None]
    output = []
    for cell, points, lattice_error, error in measurements:
        result = dict(cell)
        repaired = False
        repairable = lattice_error or error in ('invalid_crop', 'clipped_grid')
        if repair and repairable and not cell.get('row_ambiguous') and not cell.get('column_ambiguous'):
            neighbours = [(other, p) for other, p in trusted if other is not cell
                          and other.get('line_id') == cell.get('line_id')]
            try:
                centre = np.asarray(cell.get('center'), dtype=float)
            except (TypeError, ValueError):
                centre = np.array([])
            if neighbours and centre.shape == (2,) and np.isfinite(centre).all():
                neighbours.sort(key=lambda pair: np.linalg.norm(pair[1].mean(axis=0)-centre))
                local = [p for _, p in neighbours[:3]]
                col = np.median([p[3]-p[0] for p in local], axis=0)
                row = np.median([(p[2]-p[0])/2 for p in local], axis=0)
                points = np.array([centre+(d//3-.5)*col+(d%3-1)*row for d in range(6)])
                quad = grid_quad(points)
                crosses_neighbour = any(cv2.pointPolygonTest(np.float32(quad), tuple(map(float, p[d-1])), False) >= 0
                                        for other, p in trusted if other is not cell
                                        for d in range(1, 7))
                if not crosses_neighbour and _lattice_error(points) is None:
                    result['crop_quad'] = quad.tolist()
                    repaired = True
        if not lattice_error or repaired:
            result['grid'] = dict(cell.get('grid') or {},
                                  slots={i+1: tuple(p) for i, p in enumerate(points)},
                                  expected_cols=[float(points[:3, 0].mean()), float(points[3:, 0].mean())],
                                  expected_rows=[float(points[[i, i+3], 1].mean()) for i in range(3)],
                                  bbox=grid_bbox(points, shape))
            if repaired:
                result['dot_spacing'] = float(np.linalg.norm(points[3]-points[0]))
        if repaired:
            error = grid_error(result, shape)
        result['geometry_status'] = 'invalid' if error else ('repaired' if repaired else 'valid')
        result['geometry_error'] = error
        output.append(result)
    return validate_cell_relations(output)


def validate_cell_relations(cells):
    """Reject intersecting logical cells and duplicate ownership, never guess phase.

    Compare the actual 2x3 footprint, not padded axis-aligned display boxes:
    neighbouring rotated boxes can overlap without sharing any logical slot.
    """
    output = [dict(cell, overlap_ratio=0.) for cell in cells]
    owners = {}
    polygons = []
    for index, cell in enumerate(output):
        for source in set(cell.get('source_dot_ids', ())):
            owners.setdefault(source, []).append(index)
        try:
            polygon = np.float32(slot_points(cell)[[0, 3, 5, 2]])
            area = cv2.contourArea(polygon)
            polygons.append((polygon, area) if area > 0 and cv2.isContourConvex(polygon) else None)
        except (KeyError, TypeError, ValueError):
            polygons.append(None)
    for indices in owners.values():
        if len(indices) > 1:
            for index in indices:
                output[index].update(geometry_status='invalid', geometry_error='shared_source_dot')
    bounds = np.array([(p[0].min(axis=0), p[0].max(axis=0)) if p is not None
                       else ([np.inf, np.inf], [-np.inf, -np.inf]) for p in polygons])
    for index, first in enumerate(polygons):
        if first is None:
            continue
        polygon, area = first
        low, high = bounds[index]
        candidates = np.flatnonzero(np.all(bounds[:, 0] < high, axis=1) &
                                    np.all(bounds[:, 1] > low, axis=1))
        for other_index in candidates[candidates > index]:
            second = polygons[other_index]
            if second is None:
                continue
            other, other_area = second
            intersection, _ = cv2.intersectConvexConvex(polygon, other)
            ratio = float(intersection)/min(area, other_area)
            # Only allow float32 numerical error at touching edges.
            if ratio > 1e-6:
                for target in (index, other_index):
                    cell = output[target]
                    cell['overlap_ratio'] = max(cell['overlap_ratio'], ratio)
                    cell.update(geometry_status='invalid',
                                geometry_error=cell.get('geometry_error') or 'overlapping_grid')
    for index, cell in enumerate(output):
        cell['geometry_debug'] = dict(
            cell_id=cell.get('track_id', index+1), line_id=cell.get('line_id'),
            source_dots=list(cell.get('source_dot_ids', ())),
            bbox=cell.get('grid', {}).get('bbox'), spacing=cell.get('dot_spacing'),
            overlap_ratio=cell['overlap_ratio'], validation_status=cell['geometry_status'],
            validation_error=cell.get('geometry_error'))
    return output
