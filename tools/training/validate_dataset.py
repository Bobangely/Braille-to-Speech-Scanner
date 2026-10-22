"""Audit YOLO labels, scene split isolation, file hashes, and rendered metadata."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from collections import Counter
import hashlib
import json

import cv2
import numpy as np
import yaml


def _records(root, visited=None):
    visited = set() if visited is None else visited
    if root in visited:
        raise ValueError('Dataset parent cycle')
    visited.add(root)
    info = json.loads((root/'dataset.json').read_text(encoding='utf-8'))
    if info['parent']:
        yield from _records(Path(info['parent']).resolve(), visited)
    for line in (root/'records.jsonl').read_text(encoding='utf-8').splitlines():
        yield root, json.loads(line)


def validate_dataset(directory):
    root = Path(directory).resolve()
    config = yaml.safe_load((root/'data.yaml').read_text(encoding='utf-8'))
    # Ultralytics resolves relative split lists against this field. Audit the
    # same files that training will read, including when the YAML was edited.
    configured_root = Path(config.get('path', ''))
    if not configured_root.is_absolute() or configured_root.resolve() != root:
        raise ValueError('Dataset path must be the absolute path of this dataset version')
    if config.get('names') != {0: 'braille_dot'} or config.get('nc') != 1:
        raise ValueError('Dataset must contain exactly class 0: braille_dot')
    expected = {}
    for split in ('train', 'val', 'test'):
        if config.get(split) != f'{split}.txt':
            raise ValueError(f'Unexpected split list: {split}')
        paths = (root/f'{split}.txt').read_text(encoding='utf-8').splitlines()
        if not paths:
            raise ValueError(f'Empty split: {split}')
        for path in paths:
            resolved = Path(path).resolve()
            if resolved in expected:
                raise ValueError(f'Repeated image in split lists: {resolved}')
            expected[resolved] = split
    counts, groups, images_seen, hashes, dot_count, negative_count = Counter(), {}, set(), {}, 0, 0
    for record_root, record in _records(root):
        paths = [(record_root/record[key]).resolve() for key in ('image', 'label', 'metadata')]
        if any(not path.is_relative_to(record_root) for path in paths):
            raise ValueError('Record path escapes its dataset version')
        image_path, label_path, metadata_path = paths
        split = record['split']
        if expected.get(image_path) != split or image_path in images_seen:
            raise ValueError(f'Record does not match split list: {image_path}')
        images_seen.add(image_path)
        group = record['group_id']
        if group in groups and groups[group] != split:
            raise ValueError(f'Scene variants leak across splits: {group}')
        groups[group] = split
        content, label_bytes = image_path.read_bytes(), label_path.read_bytes()
        image_hash = hashlib.sha256(content).hexdigest()
        if image_hash != record['image_sha256'] or hashlib.sha256(label_bytes).hexdigest() != record['label_sha256']:
            raise ValueError(f'File hash mismatch: {image_path}')
        if image_hash in hashes and hashes[image_hash] != split:
            raise ValueError('Identical image content occurs in different splits')
        hashes[image_hash] = split
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata['group_id'] != group or metadata['split'] != split:
            raise ValueError('Metadata split/group mismatch')
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (metadata['height'], metadata['width']):
            raise ValueError(f'Image dimensions mismatch: {image_path}')
        rows = label_bytes.decode('utf-8').splitlines()
        if len(rows) != len(metadata['dots']):
            raise ValueError('Dot count differs between labels and rendering metadata')
        negative_count += not rows
        for row, dot in zip(rows, metadata['dots']):
            fields = row.split()
            if len(fields) != 5 or fields[0] != '0':
                raise ValueError(f'Invalid YOLO label: {label_path}')
            x, y, w, h = map(float, fields[1:])
            if (not np.isfinite([x, y, w, h]).all() or min(w, h) <= 0
                    or min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6):
                raise ValueError(f'Invalid normalized box: {label_path}')
            x1, y1, x2, y2 = dot['bbox']
            mw, mh = metadata['width'], metadata['height']
            expected_box = [(x1+x2)/(2*mw), (y1+y2)/(2*mh), (x2-x1)/mw, (y2-y1)/mh]
            if not np.allclose([x, y, w, h], expected_box, atol=1e-6):
                raise ValueError('YOLO box does not follow the rendered transform')
            cell = metadata['cells'][dot['cell_index']]
            if dot['dot_number'] not in cell['pattern']:
                raise ValueError('Dot label does not belong to its physical Braille cell')
        counts[split] += 1
        dot_count += len(rows)
    if images_seen != set(expected):
        raise ValueError('Split list contains images without generation records')
    return dict(valid=True, images=dict(counts), scene_groups=len(groups), dots=dot_count,
                negatives=negative_count, cross_split_group_leaks=0, cross_split_exact_duplicates=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset), ensure_ascii=False, indent=2))
