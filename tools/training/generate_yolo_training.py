"""Generate a versioned Thai Braille YOLO dataset; extend training with fixed holdouts."""
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

from tools.training.synthetic_braille import GENERATOR_VERSION, corpus, derived_seed, make_scene, render_variant

ROOT = Path(__file__).resolve().parents[2]


def generate_dataset(output_dir=None, train_groups=1200, val_groups=150, test_groups=150,
                     variants=2, seed=20260908, extend_from=None):
    if any(n < 0 for n in (train_groups, val_groups, test_groups)) or train_groups < 1:
        raise ValueError('Training groups must be positive; split counts cannot be negative')
    if not 1 <= variants <= 8:
        raise ValueError('variants must be 1..8')
    parent = Path(extend_from).resolve() if extend_from else None
    seed_history = []
    files = {split: [] for split in ('train', 'val', 'test')}
    if parent:
        from tools.training.validate_dataset import validate_dataset
        validate_dataset(parent)
        parent_info = json.loads((parent/'dataset.json').read_text(encoding='utf-8'))
        seed_history = parent_info['seed_history']
        if seed in seed_history:
            raise ValueError('Extension seed already exists; use a new seed for new training scenes')
        if val_groups or test_groups:
            raise ValueError('Extensions must keep val/test fixed: use zero val/test groups')
        files = {split: (parent/f'{split}.txt').read_text(encoding='utf-8').splitlines() for split in files}
    elif val_groups < 1 or test_groups < 1:
        raise ValueError('A new dataset needs both validation and test groups')
    destination = Path(output_dir or ROOT/'datasets/thai_synthetic'/f'seed_{seed}').resolve()
    destination.mkdir(parents=True, exist_ok=False)  # never overwrite an existing dataset
    counts, kinds, colors, symbols = Counter(), Counter(), Counter(), Counter()
    previews = []
    with (destination/'records.jsonl').open('w', encoding='utf-8') as records:
        for split, group_count in [('train', train_groups), ('val', val_groups), ('test', test_groups)]:
            for folder in ('images', 'labels', 'metadata'):
                (destination/folder/split).mkdir(parents=True)
            for index in range(group_count):
                group_id = f'{seed}_{split}_{index:06d}'
                scene = make_scene(derived_seed(seed, split, index, 'scene'), index)
                for variant in range(variants):
                    stem = f'{group_id}_v{variant}'
                    image_bytes, labels, metadata = render_variant(scene, derived_seed(seed, split, index, variant))
                    image_path = destination/'images'/split/(stem+'.jpg')
                    label_path = destination/'labels'/split/(stem+'.txt')
                    metadata_path = destination/'metadata'/split/(stem+'.json')
                    metadata.update(group_id=group_id, split=split, variant=variant, seed=seed)
                    image_path.write_bytes(image_bytes)
                    label_path.write_text(labels, encoding='utf-8', newline='\n')
                    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding='utf-8')
                    record = dict(group_id=group_id, split=split,
                                  image=image_path.relative_to(destination).as_posix(),
                                  label=label_path.relative_to(destination).as_posix(),
                                  metadata=metadata_path.relative_to(destination).as_posix(),
                                  image_sha256=hashlib.sha256(image_bytes).hexdigest(),
                                  label_sha256=hashlib.sha256(labels.encode()).hexdigest())
                    records.write(json.dumps(record, ensure_ascii=False)+'\n')
                    files[split].append(image_path.as_posix())
                    counts[split] += 1
                    kinds[scene['kind']] += 1
                    colors[scene['color']] += 1
                    symbols.update(set(c['symbol_text'] for c in scene['cells']))
                    if split == 'train' and index < 20 and variant == 0:
                        picture = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                        for dot in metadata['dots']:
                            x1, y1, x2, y2 = map(round, dot['bbox'])
                            cv2.rectangle(picture, (x1, y1), (x2, y2), (0, 180, 255), 1)
                        thumb = cv2.resize(picture, (256, 192), interpolation=cv2.INTER_AREA)
                        cv2.putText(thumb, f'{index}: {scene["kind"]} / {len(metadata["dots"])} dots',
                                    (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 180), 1)
                        previews.append(thumb)
                if (index+1) % 100 == 0:
                    print(f'{split}: {index+1}/{group_count} scene groups', flush=True)
    for split, image_files in files.items():
        (destination/f'{split}.txt').write_text('\n'.join(image_files)+'\n', encoding='utf-8', newline='\n')
    config = dict(path=destination.as_posix(), train='train.txt', val='val.txt', test='test.txt',
                  nc=1, names={0: 'braille_dot'})
    (destination/'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (destination/'corpus.json').write_text(json.dumps(corpus(), ensure_ascii=False, indent=2), encoding='utf-8')
    info = dict(generator_version=GENERATOR_VERSION, seed=seed, seed_history=seed_history+[seed],
                parent=str(parent) if parent else None, variants=variants,
                new_counts=dict(counts), total_counts={split: len(paths) for split, paths in files.items()},
                scene_kinds=dict(kinds), colors=dict(colors), symbol_image_counts=dict(symbols),
                class_names=config['names'], label_source='rendered dot geometry; no CV or auto-label detector',
                holdout_policy='all views of a scene share one split; extensions reuse the original val/test')
    (destination/'dataset.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    if previews:
        previews += [np.full((192, 256, 3), 255, np.uint8)]*((-len(previews)) % 4)
        sheet = np.vstack([np.hstack(previews[i:i+4]) for i in range(0, len(previews), 4)])
        if not cv2.imwrite(str(destination/'preview.jpg'), sheet):
            raise OSError('Cannot save dataset preview')
    from tools.training.validate_dataset import validate_dataset
    report = validate_dataset(destination)
    (destination/'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(dataset=str(destination), **report), ensure_ascii=False), flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=None)
    parser.add_argument('--train-groups', type=int, default=1200)
    parser.add_argument('--val-groups', type=int, default=None)
    parser.add_argument('--test-groups', type=int, default=None)
    parser.add_argument('--variants', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--extend-from', default=None, help='Previous dataset directory; append train only')
    args = parser.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    generate_dataset(args.output, args.train_groups,
                     args.val_groups if args.val_groups is not None else (0 if args.extend_from else 150),
                     args.test_groups if args.test_groups is not None else (0 if args.extend_from else 150),
                     args.variants, args.seed, args.extend_from)


if __name__ == '__main__':
    main()
