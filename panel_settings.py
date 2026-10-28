"""Settings the teacher panel can change while the reader runs. See docs/web-api.md.

The web server calls ``apply_settings``; a wrong value raises ValueError with a Thai
message for the page, and changes nothing.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

DEFAULTS = {'mode': 'spell', 'spell_max_cells': 2, 'camera': True}
DEFAULT_PATH = Path(__file__).resolve().parent / 'output' / 'panel_settings.json'
MODES = ('spell', 'word')
log = logging.getLogger(__name__)


def validate(partial: dict, current: dict) -> dict:
    """Return ``current`` updated with ``partial``, or raise ValueError if any value is wrong."""
    if not isinstance(partial, dict):
        raise ValueError('ค่าตั้งต้องเป็น JSON object')
    unknown = set(partial) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'ไม่รู้จักค่าตั้ง: {", ".join(sorted(unknown))}')
    if 'mode' in partial and partial['mode'] not in MODES:
        raise ValueError('mode ต้องเป็น "spell" หรือ "word"')
    cells = partial.get('spell_max_cells', 1)
    if type(cells) is not int or not 1 <= cells <= 5:  # bool is an int subclass: reject it
        raise ValueError('spell_max_cells ต้องเป็นจำนวนเต็ม 1–5')
    if 'camera' in partial and type(partial['camera']) is not bool:
        raise ValueError('camera ต้องเป็น true หรือ false')
    return {**current, **partial}


def load(path: Path) -> dict:
    """Saved settings, or the defaults when the file is missing or broken."""
    try:
        return validate(json.loads(path.read_text(encoding='utf-8')), DEFAULTS)
    except FileNotFoundError:
        return dict(DEFAULTS)
    except (OSError, ValueError) as error:  # JSONDecodeError is a ValueError
        log.warning('ignoring settings file %s: %s', path, error)
        return dict(DEFAULTS)


def save(path: Path, settings: dict) -> None:
    """Write through a temporary file so a power cut never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def apply_to(speech, settings: dict) -> None:
    speech.mode, speech.spell_max_cells = settings['mode'], settings['spell_max_cells']


def apply_settings(speech, path: Path, partial: dict) -> dict:
    """Validate ``partial`` against the saved settings, apply it to ``speech``, save, and return all settings."""
    settings = validate(partial, load(path))
    apply_to(speech, settings)
    save(path, settings)
    return settings
