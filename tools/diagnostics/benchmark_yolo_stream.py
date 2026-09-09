"""Headless evaluation of YOLO cell streaming on labelled synthetic pages."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import hashlib
import time
import cv2

from decoder import decode_cells
from yolo_detector import YOLOBrailleDetector
from tests.test_yolo_cell_stream import page
from tests.test_thai_standard import CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='output/yolo_stream_benchmark.json')
    parser.add_argument('--scales', nargs='+', type=float, default=[1.0, .25, .2])
    parser.add_argument('--model', default=None, help='Local candidate weights to evaluate')
    args = parser.parse_args()
    if any(not 0 < scale <= 2 for scale in args.scales):
        parser.error('scales must be in (0, 2]')
    detector = YOLOBrailleDetector(model_path=args.model, tile_size=0)
    records = []
    cases = [(text, [patterns], 95, 128, 0) for text, patterns in CASES]
    cases += [('ญา\nภ้', [['6', '13456', '16'], ['6', '1456', '256']], pitch, height, angle)
              for pitch, height, angle in [(95, 120, 0), (95, 128, 6), (148, 140, -6)]]
    cases += [('ญ', [['6', '13456']], 148, 128, 0)]
    for (expected, lines, pitch, height, angle), scale in (
            (case, scale) for case in cases for scale in args.scales):
        image, _, patterns = page(lines, pitch=pitch, line_pitch=height, angle=angle)
        if scale != 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        start = time.perf_counter()
        cells, debug = detector.detect(image)
        actual = decode_cells(cells, lang='thai')
        records.append(dict(expected=expected, actual=actual,
                            scale=scale, pitch=pitch, line_pitch=height, angle=angle,
                            patterns_expected=[sorted(p) for p in patterns],
                            patterns_actual=[sorted(c['dots']) for c in cells],
                            passed=actual == expected and [c['dots'] for c in cells] == patterns,
                            line_count=debug['line_count'], crops=debug['crop_count'],
                            seconds=time.perf_counter()-start))
        if not records[-1]['passed']:
            print('FAIL', repr(expected), 'scale', scale, repr(actual), records[-1]['patterns_actual'])
    output = Path(__file__).resolve().parents[2] / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    import ultralytics
    import torch
    output.write_text(json.dumps(dict(mode='yolo', pipeline='stream',
                                     dataset='synthetic reference fixtures; not camera accuracy',
                                     proposal_confidence=detector.proposal_confidence,
                                     read_confidence=detector.confidence, overview_imgsz=detector.yolo_imgsz,
                                     crop_imgsz=320, tile_size=detector.tile_size,
                                     model_sha256=hashlib.sha256(Path(detector.model_path).read_bytes()).hexdigest(),
                                     ultralytics_version=ultralytics.__version__, torch_version=torch.__version__,
                                     passed=sum(r['passed'] for r in records), total=len(records), records=records),
                                     ensure_ascii=False, indent=2), encoding='utf-8')
    print('YOLO stream:', sum(r['passed'] for r in records), '/', len(records))
    if not all(r['passed'] for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
