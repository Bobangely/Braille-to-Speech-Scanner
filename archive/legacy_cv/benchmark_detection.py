"""Measure exact sentence accuracy on repository fixtures, including downscaled images."""

import argparse
import json
from pathlib import Path
import time

import cv2

from decoder import decode_cells
from tests.test_accuracy import TEST_DATASET
from yolo_detector import YOLOBrailleDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['opencv', 'hybrid', 'yolo'], default='hybrid')
    parser.add_argument('--dataset', choices=['standard', 'legacy'], default='standard')
    parser.add_argument('--scales', type=float, nargs='+', default=[1, .25, .2])
    parser.add_argument('--output', default='output/detection_benchmark.json')
    args = parser.parse_args()
    if any(scale <= 0 for scale in args.scales):
        parser.error('scales must be positive')
    detector = YOLOBrailleDetector(mode=args.mode)
    if args.mode != 'opencv' and not detector.is_yolo_ready():
        parser.error('YOLO benchmark requires local model weights and ultralytics')
    records = []
    root = Path(__file__).resolve().parent
    if args.dataset == 'standard':
        from tests.test_thai_standard import CASES, render
        dataset = [(f'reference:{text}', 'blue', 'thai', text, render(patterns))
                   for text, patterns in CASES]
    else:
        dataset = [(path, color, lang, expected, cv2.imread(str(root / path)))
                   for path, color, lang, expected in TEST_DATASET]
    for scale in args.scales:
        for path, color, lang, expected, source in dataset:
            image = source
            if image is None:
                raise FileNotFoundError(path)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if detector.color != color:
                detector.set_color(color)
            start = time.perf_counter()
            cells, debug = detector.detect(image)
            actual = decode_cells(cells, lang=lang)
            records.append(dict(image=path, scale=scale, expected=expected, actual=actual,
                                passed=actual == expected, method=debug['method'],
                                dots=len(debug['dots']), seconds=time.perf_counter()-start,
                                yolo_error=debug.get('yolo_error')))
        rows = [r for r in records if r['scale'] == scale]
        print(f"{args.mode} scale={scale}: {sum(r['passed'] for r in rows)}/{len(rows)}")
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(mode=args.mode, dataset=args.dataset, model=detector.model_path,
                                     records=records), ensure_ascii=False, indent=2), encoding='utf-8')
    if any(row['yolo_error'] for row in records):
        raise SystemExit('YOLO errors occurred; inspect the report before accepting results')
    if not all(row['passed'] for row in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
