"""Write the four ArUco corner markers as PNG files to place in the Word template."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from page_calibration import CORNERS, DEFAULT_FORMAT, PageFormat  # noqa: E402


def export_markers(out: Path, px: int = 600, fmt: PageFormat = DEFAULT_FORMAT) -> list[Path]:
    dictionary = cv2.aruco.getPredefinedDictionary(fmt.aruco_dictionary)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for corner, mid in zip(CORNERS, fmt.marker_ids):
        path = out / f"marker_{mid}_{corner}.png"
        if not cv2.imwrite(str(path), cv2.aruco.generateImageMarker(dictionary, mid, px)):
            raise OSError(f"cannot write {path}")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/markers"))
    parser.add_argument("--px", type=int, default=600)
    args = parser.parse_args()
    for path in export_markers(args.out, args.px):
        print(path)
    print(f"Place each at {DEFAULT_FORMAT.marker_size_mm / 10:g} x {DEFAULT_FORMAT.marker_size_mm / 10:g} cm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
