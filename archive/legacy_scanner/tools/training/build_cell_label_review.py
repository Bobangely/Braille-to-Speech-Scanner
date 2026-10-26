"""Build a human-reviewable cell dataset from exported diagnostic folders.

This tool copies files; it never moves or removes diagnostic inputs.  Labels in
diagnostic manifests are model-generated proposals and must be reviewed before
they are used as training ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_thai import (
    THAI_CONSONANTS,
    THAI_DIGIT_MAP,
    THAI_SPECIAL_MARKS,
    THAI_TONE_MARKS,
    THAI_VOWELS,
)


DIAG_NAME = re.compile(r"diag_\d{3}$")


def _pattern(dots) -> str:
    values = sorted({int(dot) for dot in dots if 1 <= int(dot) <= 6})
    return "".join(map(str, values)) or "empty"


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _category(char: str) -> str:
    if char in set(THAI_CONSONANTS.values()):
        return "01_พยัญชนะ"
    if char in set(THAI_VOWELS.values()):
        return "02_สระ"
    if char in set(THAI_TONE_MARKS.values()):
        return "03_วรรณยุกต์"
    if char in set(THAI_SPECIAL_MARKS.values()):
        return "04_เครื่องหมาย"
    if char in set(THAI_DIGIT_MAP.values()):
        return "05_ตัวเลข"
    return "06_อื่นๆ"


def _char_folder(char: str) -> str:
    codepoints = "-".join(f"U+{ord(value):04X}" for value in char)
    visible = char.replace("◌", "วงกลมประ").replace("/", "_")
    return f"{codepoints}_{visible}"


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build(source_root: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}. Choose a new folder."
        )

    # Build the complete numeric interval so a temporarily omitted directory
    # listing cannot silently drop a diag folder between the first and last.
    numbers = [
        int(match.group(1))
        for path in source_root.iterdir()
        if path.is_dir() and (match := re.fullmatch(r"diag_(\d{3})", path.name))
    ]
    if not numbers:
        raise FileNotFoundError(f"No diag_NNN folders found under {source_root}")
    diagnostic_dirs = [
        source_root / f"diag_{number:03d}"
        for number in range(min(numbers), max(numbers) + 1)
    ]

    destination.mkdir(parents=True)
    rows = []
    seen_hashes: dict[str, str] = {}
    counts = Counter()

    for diagnostic in diagnostic_dirs:
        manifest_file = diagnostic / "manifest.json"
        if not manifest_file.is_file():
            counts["missing_manifest"] += 1
            continue
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        for index, cell in enumerate(manifest.get("cells", []), 1):
            crop_name = cell.get("crop_file", "")
            source = diagnostic / crop_name
            if not crop_name or not source.is_file():
                counts["missing_crop"] += 1
                continue

            token = cell.get("token") or {}
            pattern = _pattern(cell.get("dots", []))
            char = str(token.get("char") or "")
            warning = str(token.get("warning") or "")
            consumed = bool(token.get("consumed"))
            cell_start = token.get("cell_start")
            cell_end = token.get("cell_end")
            multi_head = (cell_start is not None and cell_end is not None
                          and int(cell_end) > int(cell_start))
            digest = _digest(source)
            duplicate_of = seen_hashes.get(digest, "")
            base = f"{diagnostic.name}__{Path(crop_name).stem}__p{pattern}.png"
            pattern_relative = Path("by_pattern") / f"p_{pattern}" / base

            if not duplicate_of:
                _copy(source, destination / pattern_relative)
                seen_hashes[digest] = pattern_relative.as_posix()
                counts["unique_crops"] += 1
                counts[f"pattern:{pattern}"] += 1
            else:
                counts["exact_duplicates"] += 1

            reasons = []
            if consumed:
                reasons.append("multi_cell_continuation")
            if multi_head:
                reasons.append("multi_cell_head")
            if warning:
                reasons.append("decoder_warning")
            if not char or char == "\ufffd":
                reasons.append("unknown_character")

            thai_relative = ""
            review_relative = ""
            # A Thai-character view is only a convenience for review.  Keep
            # multi-cell and ambiguous items out so one crop is not mislabeled
            # as the complete Thai character.
            if not duplicate_of and not reasons:
                category = _category(char)
                thai_relative_path = (
                    Path("by_thai_label") / category / _char_folder(char) / base
                )
                _copy(source, destination / thai_relative_path)
                thai_relative = thai_relative_path.as_posix()
                counts[f"thai:{category}"] += 1
            elif not duplicate_of:
                reason_folder = reasons[0] if reasons else "other"
                review_path = Path("needs_review") / reason_folder / base
                _copy(source, destination / review_path)
                review_relative = review_path.as_posix()
                counts[f"review:{reason_folder}"] += 1

            rows.append({
                "source_diag": diagnostic.name,
                "source_crop": crop_name,
                "sha256": digest,
                "pattern": pattern,
                "thai_char_proposal": char,
                "category": _category(char) if char and char != "\ufffd" else "",
                "warning": warning,
                "consumed": consumed,
                "multi_cell_head": multi_head,
                "review_reasons": "|".join(reasons),
                "duplicate_of": duplicate_of,
                "pattern_file": "" if duplicate_of else pattern_relative.as_posix(),
                "thai_file": thai_relative,
                "review_file": review_relative,
            })

    fields = list(rows[0]) if rows else []
    with (destination / "records.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": str(source_root.resolve()),
        "diagnostic_folders": [path.name for path in diagnostic_dirs],
        "manifest_records": len(rows),
        "counts": dict(sorted(counts.items())),
        "notice": "Auto labels are proposals. Human review is required before training.",
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "README.md").write_text(
        "# Cell label review dataset\n\n"
        "สร้างจาก `output/diag_NNN` โดยคัดลอกไฟล์ ไม่ได้ย้ายหรือลบต้นฉบับ\n\n"
        "- `by_pattern/`: มุมมองหลักสำหรับตรวจ pattern 1-6 และเตรียม training class\n"
        "- `by_thai_label/`: มุมมอง ก-ฮ/สระ/วรรณยุกต์ เฉพาะ single-cell ที่ไม่มี warning\n"
        "- `needs_review/`: multi-cell, continuation, warning และอักษรที่ยังไม่ทราบ\n"
        "- `records.csv`: ที่มา pattern อักษรที่ระบบเสนอ และสถานะตรวจสอบ\n\n"
        "ชื่ออักษรใน manifest เป็น pseudo-label จากระบบเดิม ต้องตรวจด้วยคนก่อนแบ่ง "
        "train/val/test ห้ามถือเป็น ground truth โดยอัตโนมัติ\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
