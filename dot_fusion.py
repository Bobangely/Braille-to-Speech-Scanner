"""Native-resolution tiled inference and conservative, size-aware dot fusion."""

import math


def tile_windows(width, height, size=1280, overlap=0.2, max_tiles=16):
    """Return complete overlapping coverage, or None when the work budget is exceeded.

    Never silently process only one corner of a large image. The caller can fall
    back to its overview prediction and report that the tile budget was exceeded.
    """
    if size == 0 or max(width, height) <= size:
        return []
    if size < 32 or not 0 <= overlap < 1 or max_tiles < 1:
        raise ValueError('Invalid tiling configuration')
    stride = max(1, int(size * (1 - overlap)))

    def positions(length):
        if length <= size:
            return [0]
        last = length - size
        values = list(range(0, last + 1, stride))
        if values[-1] != last:
            values.append(last)
        return values

    xs, ys = positions(width), positions(height)
    if len(xs) * len(ys) > max_tiles:
        return None
    return [(x, y, min(x + size, width), min(y + size, height))
            for y in ys for x in xs]


def merge_dots(preferred, candidates):
    """Keep CV centroids when available; merge only within the smaller dot's radius.

    Unlike a fixed distance threshold, this preserves distinct adjacent small
    dots. Input dictionaries are copied so detector results remain independent.
    """
    merged = [dict(dot) for dot in preferred]
    for dot in sorted(candidates, key=lambda d: d.get('confidence', 0), reverse=True):
        cx, cy = dot['center']
        best, best_distance = None, float('inf')
        for existing in merged:
            ex, ey = existing['center']
            distance = math.hypot(cx - ex, cy - ey)
            radius = math.sqrt(min(dot['area'], existing['area']) / math.pi)
            if distance <= max(1.0, radius * 0.8) and distance < best_distance:
                best, best_distance = existing, distance
        if best is None:
            merged.append(dict(dot))
        else:
            if 'confidence' in dot:
                best['confidence'] = max(best.get('confidence', 0), dot['confidence'])
            if best.get('source') == 'opencv':
                best['source'] = 'both'
    return merged
