"""Write the printable calibration page (markers + sample Braille line) as PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from page_calibration import DEFAULT_FORMAT, render_page  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/calibration_page.png"))
    parser.add_argument("--dpi", type=float, default=300.0)
    args = parser.parse_args()
    page = render_page(DEFAULT_FORMAT, px_per_mm=args.dpi / 25.4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), page):
        raise RuntimeError(f"cannot write {args.out}")
    print(f"{args.out} {page.shape[1]}x{page.shape[0]} px at {args.dpi:g} dpi "
          f"({DEFAULT_FORMAT.width_mm:g}x{DEFAULT_FORMAT.height_mm:g} mm, print at 100%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
