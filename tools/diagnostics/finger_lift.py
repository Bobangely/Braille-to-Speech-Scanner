"""เทียบไฟล์ --log-finger ตอนอ่านกับตอนยกมือ เพื่อดูว่ากฎความเร็วหรือขนาดมือแยกเสียงเกินได้ไหม

ตอนอ่าน: อ่านปกติ จบโดยนิ้วยังอยู่บนหน้า
ตอนยกมือ: วางนิ้วบนเซลล์ ยกมือออกนอกภาพ ทำซ้ำอย่างน้อย 15 ครั้ง
เซลล์แรกหลังมือหายไปคือการวางนิ้ว จึงไม่นับทั้งสองไฟล์

python tools/diagnostics/finger_lift.py output/finger_read.csv output/finger_lift.csv
"""
import argparse
import csv
import math
from pathlib import Path
from statistics import median

WINDOW_S = 0.3  # เท่ากับเวลาที่นิ้วต้องอยู่บนเซลล์ก่อนพูด (FingerState.dwell_s)
GAP_S = 0.3  # ไม่เห็นนิ้วนานเท่านี้ เซลล์ถัดไปคือการวางนิ้วใหม่
BASE_S = 1.0  # ขนาดมือช่วงก่อนหน้านี้ ใช้เป็นฐานเทียบว่ามือใหญ่ขึ้นแค่ไหน


def load(path: Path) -> list[dict]:
    with path.open(encoding='utf-8', newline='') as source:
        return [dict(t=float(row['t_s']), x=float(row['x_mm']) if row['x_mm'] else None,
                     y=float(row['y_mm']) if row['y_mm'] else None,
                     size=float(row['hand_px']) if row['hand_px'] else None, entered=row['entered'])
                for row in csv.DictReader(source)]


def events(rows: list[dict]) -> list[tuple[float, float]]:
    """(ความเร็ว mm/s, ขนาดมือเทียบฐาน) ของทุกเซลล์ที่พูด ยกเว้นเซลล์แรกหลังวางนิ้ว"""
    found, last_seen = [], None
    landing = True
    for i, row in enumerate(rows):
        if row['x'] is None:
            continue
        if last_seen is not None and row['t'] - last_seen > GAP_S:
            landing = True
        last_seen = row['t']
        if not row['entered']:
            continue
        if landing:
            landing = False
            continue
        recent = [r for r in rows[max(0, i - 60):i + 1] if r['x'] is not None and row['t'] - r['t'] <= WINDOW_S]
        path_mm = sum(math.hypot(b['x'] - a['x'], b['y'] - a['y']) for a, b in zip(recent, recent[1:]))
        speed = path_mm / max(row['t'] - recent[0]['t'], 1e-3)
        base = [r['size'] for r in rows[max(0, i - 120):i]
                if r['size'] and WINDOW_S < row['t'] - r['t'] <= WINDOW_S + BASE_S]
        found.append((speed, row['size'] / median(base) if base and row['size'] else 1.0))
    return found


def best_cut(reading: list[float], lifting: list[float]) -> tuple[float, float, float]:
    """ค่าตัดที่กันเสียงเกินได้มากสุด โดยเสียเซลล์ที่อ่านจริงไม่เกิน 5%: (ค่าตัด, เสียจริง, กันได้)"""
    best = (float('inf'), 0.0, 0.0)
    for cut in sorted(set(reading + lifting)):
        lost = sum(v >= cut for v in reading) / max(1, len(reading))
        blocked = sum(v >= cut for v in lifting) / max(1, len(lifting))
        if lost <= .05 and blocked > best[2]:
            best = (cut, lost, blocked)
    return best


def report(name: str, reading: list[float], lifting: list[float]) -> None:
    def spread(values):
        ordered = sorted(values)
        return ' / '.join(f'{ordered[min(len(ordered) - 1, int(p * len(ordered)))]:.2f}' for p in (.5, .95))
    cut, lost, blocked = best_cut(reading, lifting)
    print(f'{name}: อ่าน p50/p95 {spread(reading)} | ยกมือ p50/p95 {spread(lifting)}')
    print(f'  ค่าตัด {cut:.2f}: เสียเซลล์ที่อ่านจริง {lost:.0%} กันเสียงเกินได้ {blocked:.0%}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('reading', type=Path)
    parser.add_argument('lifting', type=Path)
    args = parser.parse_args()
    reading, lifting = events(load(args.reading)), events(load(args.lifting))
    print(f'เซลล์ตอนอ่าน {len(reading)} ครั้ง, เสียงเกินตอนยกมือ {len(lifting)} ครั้ง')
    if not reading or not lifting:
        print('ข้อมูลไม่พอ: ต้องมีทั้งสองแบบ')
        return 1
    report('ความเร็ว (mm/s)', [e[0] for e in reading], [e[0] for e in lifting])
    report('ขนาดมือเทียบฐาน', [e[1] for e in reading], [e[1] for e in lifting])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
