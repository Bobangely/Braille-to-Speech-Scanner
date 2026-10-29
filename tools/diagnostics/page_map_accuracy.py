"""Compare a camera image of a printed page with its answer text, cell by cell.

The answer file holds one line of Thai text per colored Braille line.
Usage: page_map_accuracy.py page.png --truth page.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from generate_test import text_to_braille_cells  # noqa: E402
from page_calibration import calibrate, page_format  # noqa: E402
from page_layout import PageMap  # noqa: E402
from page_map import read_page  # noqa: E402


def cell_accuracy(page: PageMap, texts) -> tuple[int, int]:
    """(cells read with the right dots, cells in the answer); spaces are not counted."""
    correct = total = 0
    for line_id, text in enumerate(texts):
        expected = text_to_braille_cells(text, 'thai')
        read = page.lines[line_id].cells if line_id < len(page.lines) else ()
        for index, dots in enumerate(expected):
            if not dots:
                continue
            total += 1
            correct += index < len(read) and read[index].dots == frozenset(dots)
    return correct, total


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--marker-size-mm', type=float, help='printed marker side, as in the book template')
    args = parser.parse_args()
    fmt = page_format(args.marker_size_mm)
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f'cannot read {args.image}')
    calibration = calibrate(image, fmt)
    if not calibration.ok:
        print(f'calibration failed: {calibration.reason}')
        return 1
    page = read_page(image, calibration, fmt)
    texts = [t for t in args.truth.read_text(encoding='utf-8').splitlines() if t.strip()]
    correct, total = cell_accuracy(page, texts)
    print(f'{correct}/{total} cells = {100 * correct / total:.1f}%  words: {" ".join(page.words)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
