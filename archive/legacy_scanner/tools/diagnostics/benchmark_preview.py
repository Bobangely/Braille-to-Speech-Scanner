"""Headless render-cost comparison, not a camera or board FPS measurement."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import statistics
import time
from unittest.mock import patch
import numpy as np
from live_preview import LivePreview
from yolo_detector import YOLOBrailleDetector


def main():
    with patch.object(YOLOBrailleDetector, '_load_model'):
        detector = YOLOBrailleDetector()
    frame = np.full((2160, 3840, 3), 220, np.uint8)
    result = dict(result_id=1, dots=[], cells=[], decoded_text='ภาษาไทย', verbose_results=[])
    preview = LivePreview()
    operations = {
        'full_4k_annotation': lambda: detector.annotate_with_text(frame, [], [], decoded_text='ภาษาไทย'),
        'preview_cached_overlay': lambda: preview.render(frame, detector, result, 'thai'),
    }
    report = {}
    for name, operation in operations.items():
        operation()
        samples = []
        for _ in range(20):
            start = time.perf_counter()
            operation()
            samples.append((time.perf_counter() - start) * 1000)
        report[name] = {'median_ms': round(statistics.median(samples), 3)}
    path = Path(__file__).resolve().parents[2] / 'output/preview_benchmark.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
