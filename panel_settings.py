"""ค่าตั้งที่หน้าเว็บครูเปลี่ยนได้ขณะระบบทำงาน ดู docs/web-api.md

web server เรียก ``apply_settings`` ถ้าค่าผิดจะเกิด ValueError พร้อมข้อความไทยสำหรับหน้าเว็บ
และไม่เปลี่ยนค่าใดเลย
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

DEFAULTS = {'mode': 'spell', 'spell_max_cells': 2, 'camera': True, 'finger': 'index', 'hand': 'any'}
DEFAULT_PATH = Path(__file__).resolve().parent / 'output' / 'panel_settings.json'
MODES = ('spell', 'word')
FINGERS = ('thumb', 'index', 'middle')  # ตรงกับ fingertip.FINGERS
HANDS = ('any', 'left', 'right')
log = logging.getLogger(__name__)
_lock = threading.Lock()


def validate(partial: dict, current: dict) -> dict:
    """คืน ``current`` ที่อัปเดตด้วย ``partial`` หรือเกิด ValueError ถ้ามีค่าผิด"""
    if not isinstance(partial, dict):
        raise ValueError('ค่าตั้งต้องเป็น JSON object')
    unknown = set(partial) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'ไม่รู้จักค่าตั้ง: {", ".join(sorted(unknown))}')
    if 'mode' in partial and partial['mode'] not in MODES:
        raise ValueError('mode ต้องเป็น "spell" หรือ "word"')
    cells = partial.get('spell_max_cells', 1)
    if type(cells) is not int or not 1 <= cells <= 5:  # bool เป็น subclass ของ int จึงต้องปฏิเสธเอง
        raise ValueError('spell_max_cells ต้องเป็นจำนวนเต็ม 1–5')
    if 'camera' in partial and type(partial['camera']) is not bool:
        raise ValueError('camera ต้องเป็น true หรือ false')
    if 'finger' in partial and partial['finger'] not in FINGERS:
        raise ValueError('finger ต้องเป็น "thumb", "index" หรือ "middle"')
    if 'hand' in partial and partial['hand'] not in HANDS:
        raise ValueError('hand ต้องเป็น "any", "left" หรือ "right"')
    return {**current, **partial}


def load(path: Path) -> dict:
    """ค่าตั้งที่บันทึกไว้ หรือค่าเริ่มต้นถ้าไฟล์หายหรือเสีย"""
    try:
        return validate(json.loads(path.read_text(encoding='utf-8')), DEFAULTS)
    except FileNotFoundError:
        return dict(DEFAULTS)
    except (OSError, ValueError) as error:  # JSONDecodeError เป็นชนิดย่อยของ ValueError
        log.warning('ignoring settings file %s: %s', path, error)
        return dict(DEFAULTS)


def save(path: Path, settings: dict) -> None:
    """เขียนผ่านไฟล์ชั่วคราว ไฟดับจึงไม่ทิ้งไฟล์ที่เขียนค้างครึ่งเดียว"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def apply_to(speech, settings: dict) -> None:
    speech.mode, speech.spell_max_cells = settings['mode'], settings['spell_max_cells']


def apply_settings(speech, path: Path, partial: dict) -> dict:
    """ตรวจ ``partial`` เทียบกับค่าที่บันทึกไว้ ตั้งค่าให้ ``speech`` บันทึก แล้วคืนค่าทั้งหมด"""
    with _lock:  # server อาจรับสองคำขอพร้อมกัน ค่าที่เปลี่ยนต้องไม่หาย
        settings = validate(partial, load(path))
        save(path, settings)  # บันทึกก่อน apply_to: ถ้าบันทึกไม่ได้ เสียงกับไฟล์จะยังตรงกัน
        apply_to(speech, settings)
    return settings
