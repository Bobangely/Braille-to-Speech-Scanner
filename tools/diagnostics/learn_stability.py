"""ดูว่าทำไมการเรียนหน้าช้า: อ่านกล้องสด ~20 วินาที แล้วบอกทีละเฟรมว่าการอ่านต่อเนื่องหยุดเพราะอะไร

การเรียนหน้าต้องอ่านได้เหมือนกัน 3 ครั้งติดกันโดยไม่มีมือ เฟรมที่ทำให้นับใหม่มีสามแบบ:
เห็นมือ, อ่านไม่ได้ (จุดรบกวนเยอะ), หรือเซลล์ต่างจากเฟรมก่อน (บอกเซลล์ที่ต่าง)
ต้องปิด finger_reader.py ก่อน เพราะใช้กล้องตัวเดียวกัน

    .venv/bin/python tools/diagnostics/learn_stability.py --seconds 20
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
logging.disable(logging.CRITICAL)

import cv2  # noqa: E402

from colored_braille import UnreadableBrailleFrame  # noqa: E402
from fingertip import FingertipTracker, open_camera  # noqa: E402
from page_calibration import DEFAULT_FORMAT, calibrate  # noqa: E402
from page_map import page_signature, read_page  # noqa: E402


def differences(before, after) -> list[str]:
    """เซลล์ที่ต่างกัน เป็นข้อความสั้น ๆ เช่น 'L1C4 12->124' หรือ 'บรรทัด 4 -> 5'"""
    if len(before) != len(after):
        return [f'บรรทัด {len(before)} -> {len(after)}']
    out = []
    for line, (a, b) in enumerate(zip(before, after)):
        if len(a) != len(b):
            out.append(f'L{line} เซลล์ {len(a)} -> {len(b)}')
            continue
        for cell, (x, y) in enumerate(zip(a, b)):
            if x != y:
                out.append(f"L{line}C{cell} {''.join(map(str, sorted(x))) or '-'}->{''.join(map(str, sorted(y))) or '-'}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=20.0)
    parser.add_argument('--save', type=Path, help='โฟลเดอร์เก็บภาพเฟรมที่อ่านต่างจากเฟรมก่อน (สูงสุด 8 ภาพ)')
    args = parser.parse_args()
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
    saved = 0
    capture = open_camera(args.camera, 1920, 1080, 30)
    tracker = FingertipTracker()
    previous, run, counts = None, 0, {'hand': 0, 'markers': 0, 'unreadable': 0, 'changed': 0, 'same': 0}
    start = time.monotonic()
    try:
        while time.monotonic() - start < args.seconds:
            ok, frame = capture.read()
            if not ok:
                print('อ่านกล้องไม่ได้')
                return 1
            now = time.monotonic() - start
            tracker.update(frame, round(now * 1000))
            if tracker.hand_present:
                counts['hand'] += 1
                print(f'{now:5.1f}s  เห็นมือ (นับใหม่)')
                previous, run = None, 0
                continue
            calibration = calibrate(frame, DEFAULT_FORMAT)
            if not calibration.ok:
                counts['markers'] += 1
                print(f'{now:5.1f}s  ไม่เห็น marker ครบ: {calibration.reason}')
                continue
            try:
                page = read_page(frame, calibration, DEFAULT_FORMAT)
                signature = page_signature(page)
            except UnreadableBrailleFrame as error:
                counts['unreadable'] += 1
                print(f'{now:5.1f}s  อ่านไม่ได้: {error}')
                previous, run = None, 0
                continue
            if signature == previous:
                run += 1
                counts['same'] += 1
                print(f'{now:5.1f}s  เหมือนเดิม ({run} ครั้งติด){"  <- เรียนสำเร็จได้" if run == 3 else ""}')
            else:
                counts['changed'] += 1
                diff = differences(previous, signature) if previous is not None else ['เฟรมแรก']
                print(f'{now:5.1f}s  ต่างจากเฟรมก่อน: {", ".join(diff[:8])}{" ..." if len(diff) > 8 else ""}')
                if args.save and saved < 8 and previous is not None:
                    # เก็บภาพและบอกตำแหน่งของแต่ละบรรทัด (mm บนหน้า) เพื่อหาจุดแปลกปลอม
                    saved += 1
                    cv2.imwrite(str(args.save / f'odd_{saved}_{now:.1f}s.jpg'), frame)
                    for line in page.lines:
                        filled = [c for c in line.cells if c.dots]
                        print(f'         บรรทัด y={line.raised_y1_mm:5.1f}  เซลล์ {len(line.cells)}  '
                              f'x={filled[0].x1_mm:5.1f}..{filled[-1].x2_mm:5.1f}  มีจุด {len(filled)} เซลล์')
                run = 1
            previous = signature
    finally:
        capture.release()
    print('สรุป:', counts)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
