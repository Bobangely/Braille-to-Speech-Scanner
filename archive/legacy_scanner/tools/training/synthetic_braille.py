"""Deterministic Thai Braille scenes with ground truth from drawing coordinates.

OpenCV renders/transforms images here; no detector generates training labels.
"""
import hashlib

import cv2
import numpy as np

from config_thai import (THAI_CONSONANTS, THAI_MULTI_CELL, THAI_VOWELS,
                        THAI_TONE_MARKS, THAI_CHAR_TO_BRAILLE, COMPOUND_VOWELS,
                        LEADING_VOWELS, THAI_DIGIT_MAP, NUMBER_INDICATOR)

GENERATOR_VERSION = 1
COLORS = {'black': (30, 30, 30), 'blue': (150, 65, 25),
          'red': (30, 35, 160), 'green': (35, 120, 35)}


def derived_seed(*parts):
    value = '|'.join(map(str, parts)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'big')


def corpus():
    entries = [(char, [sorted(pattern)]) for pattern, char in THAI_CONSONANTS.items()]
    entries += [(char, [sorted(p) for p in patterns]) for patterns, char in THAI_MULTI_CELL.items()]
    for pattern, char in THAI_VOWELS.items():
        if char in COMPOUND_VOWELS:
            entries.append((char.replace('◌', 'ก'), [[1, 2, 4, 5], sorted(pattern)]))
        elif char in LEADING_VOWELS:
            entries.append((char+'ก', [sorted(pattern), [1, 2, 4, 5]]))
        else:
            entries.append(('ก'+char, [[1, 2, 4, 5], sorted(pattern)]))
    entries += [('ก'+char+'า', [[1, 2, 4, 5], sorted(pattern), [1, 6]])
                for pattern, char in THAI_TONE_MARKS.items()]
    # Words without contracted compound-vowel spelling; two-cell letters stay atomic.
    for word in ('ญา', 'ภาษา', 'ธรรม', 'ศรี', 'ภูมิ', 'ผู้ใหญ่', 'หญิง', 'บ้าน', 'น้ำ',
                 'วัน', 'ดี', 'คน', 'รัก', 'ข้าว', 'ไทย', 'โรง', 'แมว', 'เด็ก'):
        patterns = []
        for char in word:
            pattern = THAI_CHAR_TO_BRAILLE[char]  # unknown characters must fail, never disappear
            patterns.extend([sorted(p) for p in pattern] if isinstance(pattern, tuple) else [sorted(pattern)])
        entries.append((word, patterns))
    digit_patterns = {value: sorted(pattern) for pattern, value in THAI_DIGIT_MAP.items()}
    entries.append(('123', [sorted(NUMBER_INDICATOR)]+[digit_patterns[n] for n in '123']))
    return sorted(entries, key=lambda entry: entry[0])


def _transform(points, matrix):
    return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), matrix).reshape(-1, 2)


def make_scene(seed, group_index):
    rng = np.random.default_rng(seed)
    # Exact 60% pages, 25% isolated cells, 15% negatives over each 20 groups.
    kind = 'page' if group_index % 20 < 12 else 'cell' if group_index % 20 < 17 else 'negative'
    width, height = (320, 320) if kind == 'cell' else (640, 480)
    spacing = float(rng.uniform(26, 38) if kind == 'cell' else rng.choice([9, 12, 18, 24, 30]))
    pitch = spacing*float(rng.uniform(2.3, 3.9))
    radius = max(1.6, spacing*float(rng.uniform(.18, .29)))
    color_name = list(COLORS)[int(rng.integers(len(COLORS)))]
    color = np.clip(np.array(COLORS[color_name])+rng.integers(-15, 16, 3), 0, 180)
    background = rng.integers(228, 256, 3)
    image = np.empty((height*2, width*2, 3), np.uint8)
    image[:] = background
    # Printed rules/text and hollow circles are background distractors, never dot labels.
    if kind == 'negative' or rng.random() < .25:
        cv2.rectangle(image, (40, 40), (width*2-40, height*2-40), (190, 190, 190), 2)
        cv2.putText(image, 'BRAILLE 2026 0123456789', (70, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    .7, (90, 90, 90), 2, cv2.LINE_AA)
    if kind == 'negative':
        for _ in range(int(rng.integers(3, 20))):
            x, y = rng.integers(60, width-60), rng.integers(70, height-60)
            cv2.circle(image, (int(2*x), int(2*y)), int(rng.integers(6, 18)),
                       (130, 130, 130), 2, cv2.LINE_AA)
    entries = corpus()
    hard = [entry for entry in entries if entry[0] in ('ญ', 'ศ', 'ภ', 'ธ', 'หญิง', 'ภูมิ', 'ธรรม')]
    objects, cells, lines = [], [], []

    def add_cell(pattern, x, y, line_id, symbol_index, symbol_text):
        cell_index = len(cells)
        slots = [(x+(d//3)*spacing, y+(d % 3)*spacing) for d in range(6)]
        cells.append(dict(pattern=pattern, line_id=line_id, symbol_index=symbol_index,
                          symbol_text=symbol_text, slots=slots))
        for dot_number in pattern:
            cx, cy = slots[dot_number-1]
            jitter = rng.uniform(-.025, .025, 2)*spacing
            cx, cy = cx+jitter[0], cy+jitter[1]
            rx, ry = radius*float(rng.uniform(.9, 1.08)), radius*float(rng.uniform(.9, 1.08))
            center = (int(round(cx*2)), int(round(cy*2)))
            axes = (max(2, int(round(rx*2))), max(2, int(round(ry*2))))
            cv2.ellipse(image, center, axes, 0, 0, 360, tuple(map(int, color)), -1, cv2.LINE_AA)
            # Include the antialiased boundary; labels follow exactly the same transform.
            half_x, half_y = (axes[0]+1)/2, (axes[1]+1)/2
            px, py = center[0]/2, center[1]/2
            objects.append(dict(cell_index=cell_index, dot_number=dot_number,
                                center=(px, py), bbox=(px-half_x, py-half_y, px+half_x, py+half_y)))

    if kind == 'cell':
        symbol, patterns = entries[int(rng.integers(len(entries)))]
        pattern = patterns[int(rng.integers(len(patterns)))]
        add_cell(pattern, width/2-spacing/2, height/2-spacing, 0, 0, symbol)
    elif kind == 'page':
        line_count = int(rng.integers(1, 4))
        line_pitch = spacing*float(rng.uniform(3.2, 4.6))
        y_start = max(80, (height-((line_count-1)*line_pitch+2*spacing))/2)
        for line_id in range(line_count):
            x = 65.0 + float(rng.uniform(0, 20))
            labels = []
            symbol_index = 0
            while x+spacing+radius < width-65:
                pool = hard if symbol_index == 0 or rng.random() < .35 else entries
                symbol, patterns = pool[int(rng.integers(len(pool)))]
                if x+(len(patterns)-1)*pitch+spacing+radius >= width-65:
                    break  # never cut a two-cell symbol at the image edge
                for pattern in patterns:
                    add_cell(pattern, x, y_start+line_id*line_pitch, line_id, symbol_index, symbol)
                    x += pitch
                labels.append(symbol)
                x += pitch  # an explicit blank Braille cell between corpus tokens
                symbol_index += 1
            lines.append(' '.join(labels))
    return dict(image=image, width=width, height=height, kind=kind, spacing=spacing,
                radius=radius, color=color_name, objects=objects, cells=cells,
                text='\n'.join(lines) if kind == 'page' else None)


def render_variant(scene, seed):
    rng = np.random.default_rng(seed)
    w, h = scene['width'], scene['height']
    angle = float(rng.uniform(-7, 7))
    rotation = np.vstack([cv2.getRotationMatrix2D((w/2, h/2), angle, .88), [0, 0, 1]])
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    skew = rng.uniform(-.015, .015, (4, 2))*[w, h]
    matrix = cv2.getPerspectiveTransform(corners, np.float32(corners+skew)) @ rotation
    scale = np.diag([2., 2., 1.])
    high = cv2.warpPerspective(scene['image'], scale @ matrix @ np.linalg.inv(scale),
                               (w*2, h*2), borderValue=(248, 248, 248))
    image = cv2.resize(high, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    lighting = 1+rng.uniform(-.13, .13)*xx+rng.uniform(-.13, .13)*yy
    noise = rng.normal(0, rng.uniform(.5, 3), (h, w, 1))
    image = np.clip(image*lighting[..., None]+noise, 0, 255).astype(np.uint8)
    blur_sigma = float(rng.uniform(.1, .65))
    image = cv2.GaussianBlur(image, (3, 3), blur_sigma)
    dots, labels = [], []
    for obj in scene['objects']:
        x1, y1, x2, y2 = obj['bbox']
        quad = _transform([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], matrix)
        low, high = quad.min(axis=0), quad.max(axis=0)
        if np.any(low < 0) or np.any(high > [w, h]):
            raise ValueError('Synthetic dot clipped by transform; adjust scene margins')
        bbox = [float(low[0]), float(low[1]), float(high[0]), float(high[1])]
        normalized = [(low[0]+high[0])/(2*w), (low[1]+high[1])/(2*h),
                      (high[0]-low[0])/w, (high[1]-low[1])/h]
        labels.append('0 '+' '.join(f'{value:.7f}' for value in normalized))
        dots.append(dict(obj, bbox=bbox, center=_transform([obj['center']], matrix)[0].tolist()))
    cells = [dict(cell, slots=_transform(cell['slots'], matrix).tolist()) for cell in scene['cells']]
    metadata = dict(kind=scene['kind'], width=w, height=h, dot_spacing=scene['spacing'],
                    dot_radius=scene['radius'], color=scene['color'], angle=angle,
                    blur_sigma=blur_sigma, transform=matrix.tolist(), text=scene['text'],
                    cells=cells, dots=dots)
    quality = int(rng.integers(85, 98))
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError('JPEG encoding failed')
    metadata['jpeg_quality'] = quality
    return encoded.tobytes(), '\n'.join(labels)+('\n' if labels else ''), metadata
