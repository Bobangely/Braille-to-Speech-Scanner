"""Read Braille aloud under the fingertip, from a two-line book page.

Learning: with no hand in view, calibrate the page by its corner markers and
read the colored Braille lines once. Reading: every frame maps the fingertip
to page millimetres and speaks the symbol of each new raised cell, then the
word when the finger leaves it to the right.

Usage: finger_reader.py --tts-thai-model voices/th.onnx [--camera 0] [--log-latency out.csv]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from finger_state import EnterCell, FingerState
from page_calibration import DEFAULT_FORMAT, CalibrationTracker, PageFormat
from page_map import PageLearner, page_signature, read_page as read_colored_page
from speech_queue import UNKNOWN, SpeechQueue
from thai_speech import SYMBOL_NAMES, THAI_CONSONANT_NAMES

ROOT = Path(__file__).resolve().parent
log = logging.getLogger(__name__)
PLACE_PAGE = 'วางหน้าให้เห็นมุมครบ'
LIFT_HAND = 'ยกมือออกก่อน'
RECHECK_S = 1.0  # re-read the colored lines this often while no hand is in view


class FingerReader:
    """One camera frame in, one status out: waiting_page, learning or reading."""

    def __init__(self, tracker, calibrator, speech: SpeechQueue, fmt: PageFormat = DEFAULT_FORMAT,
                 read_page=read_colored_page, required_maps: int = 3):
        self.tracker, self.calibrator, self.speech, self.fmt = tracker, calibrator, speech, fmt
        self.read_page, self.required_maps = read_page, required_maps
        self.status = 'waiting_page'
        self.last_events: list = []
        self._told: str | None = None
        self._learner = PageLearner(required_maps)
        self._finger: FingerState | None = None

    def step(self, frame, now_s: float) -> str:
        self.last_events = []
        calibration, changed = self.calibrator.update(frame, now_s)
        tip = self.tracker.update(frame, round(now_s * 1000))
        if calibration is None:
            self._set('waiting_page')
            self._tell(PLACE_PAGE)
        elif changed or self.status == 'waiting_page':
            self._set('learning')
        if self.status == 'learning':
            self._learn(frame, calibration, now_s)
        elif self.status == 'reading':
            point = calibration.to_page([tip])[0] if tip is not None else None
            self.last_events = self._finger.update(None if point is None else tuple(point), now_s)
            self.speech.handle(self.last_events)
            if not self.tracker.hand_present and now_s - self._checked_s >= RECHECK_S:
                self._recheck(frame, calibration, now_s)
        return self.status

    def _recheck(self, frame, calibration, now_s: float) -> None:
        """A spiral-book page turned onto the same markers changes only its dots."""
        self._checked_s = now_s
        same = page_signature(self.read_page(frame, calibration, self.fmt)) == self._signature
        self._mismatches = 0 if same else self._mismatches + 1
        if self._mismatches >= self.required_maps:
            self._set('learning')

    @property
    def entered_s(self) -> float | None:
        """When the fingertip reached the last spoken cell (start of the speech latency)."""
        return self._finger.entered_s if self._finger else None

    def _learn(self, frame, calibration, now_s: float) -> None:
        if self.tracker.hand_present:
            self._tell(LIFT_HAND)
            return
        page = self._learner.update(self.read_page(frame, calibration, self.fmt))
        if page is None:
            return
        self.speech.prepare_page(page)
        self._set('reading')
        self._finger = FingerState(page)
        self._signature, self._checked_s, self._mismatches = page_signature(page), now_s, 0
        self._tell(f'พร้อม {len(page.lines)} บรรทัด')

    def _set(self, status: str) -> None:
        if status != self.status:
            self.status, self._told, self._finger = status, None, None
            self._learner = PageLearner(self.required_maps)

    def _tell(self, text: str) -> None:
        """Say a status message once per status."""
        if self._told != text:
            self._told = text
            self.speech.say(text)


class CachedWavSpeaker:
    """Synthesize each text once to a WAV file on disk; play cached files at once.

    Synthesis runs in the Piper/MMS Python environment (``python_executable``),
    the same way ``piper_tts`` does; playback uses winsound.
    """

    def __init__(self, model: Path, cache_dir: Path, python_executable: str | None = None,
                 backend: str = 'piper', synthesize=None, play=None):
        if not Path(model).exists():
            raise FileNotFoundError(f'Thai speech model not found: {model}')
        self.model, self.backend = Path(model), backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.python = python_executable or sys.executable
        self._synthesize = synthesize or self._synthesize_in_tts_env
        self._play = play or _winsound_play

    def _path(self, text: str) -> Path:
        key = hashlib.sha1(f'{self.backend}|{self.model}|{text}'.encode('utf-8')).hexdigest()
        return self.cache_dir / f'{key}.wav'

    def prepare_many(self, texts, lang: str = 'thai') -> set[str]:
        """Synthesize the uncached texts in one batch; return the texts ready to play."""
        missing = [(t, self._path(t)) for t in dict.fromkeys(texts) if not self._path(t).is_file()]
        if missing:
            try:
                self._synthesize(self.model, missing)
            except Exception as error:
                log.warning('speech synthesis failed: %s', error)
        return {t for t in texts if self._path(t).is_file()}

    def speak(self, text: str, lang: str = 'thai') -> bool:
        path = self._path(text)
        if not path.is_file():  # never synthesize inside the frame loop
            log.warning('no prepared speech for %r', text)
            return False
        self._play(path)
        return True

    def stop(self) -> None:
        self._play(None)

    def _synthesize_in_tts_env(self, model: Path, jobs: list[tuple[str, Path]]) -> None:
        # One process, one model load: synthesize_wav caches the loaded voice per model.
        payload = json.dumps(dict(model=str(model), backend=self.backend,
                                  jobs=[[text, str(path)] for text, path in jobs]))
        code = ('import json, os, sys; sys.path.insert(0, %r); from piper_tts import synthesize_wav\n'
                'j = json.load(sys.stdin)\n'
                'for text, out in j["jobs"]:\n'
                '    tmp = out + ".tmp.wav"\n'
                '    synthesize_wav(j["model"], text, tmp, backend=j["backend"])\n'
                '    os.replace(tmp, out)\n' % str(ROOT))
        done = subprocess.run([self.python, '-X', 'utf8', '-c', code], input=payload.encode('utf-8'),
                              capture_output=True)
        if done.returncode:
            raise RuntimeError(done.stderr.decode('utf-8', 'replace')[-500:])


def _winsound_play(path: Path | None) -> None:
    # shortcut: Windows-only playback; add an aplay process for Linux boards when needed.
    import winsound
    if path is None:
        winsound.PlaySound(None, winsound.SND_PURGE)
    else:
        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)


def symbol_texts() -> list[str]:
    """Every fixed phrase the reader may speak, to synthesize before the first page."""
    return [UNKNOWN, PLACE_PAGE, LIFT_HAND, *SYMBOL_NAMES.values(), *THAI_CONSONANT_NAMES.values(),
            *(f'พร้อม {n} บรรทัด' for n in range(1, 11))]


def main() -> int:
    import cv2

    from fingertip import FingertipTracker
    from tools.diagnostics.hand_landmarker_spike import _open_camera

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--tts-thai-model', type=Path, required=True)
    parser.add_argument('--tts-thai-backend', choices=['piper', 'mms'], default='piper')
    parser.add_argument('--tts-python', help='Python executable of the Piper/MMS environment')
    parser.add_argument('--cache', type=Path, default=ROOT / 'output' / 'finger_speech')
    parser.add_argument('--log-latency', type=Path, help='CSV of per-event timings in seconds')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    speaker = CachedWavSpeaker(args.tts_thai_model, args.cache, args.tts_python, args.tts_thai_backend)
    print('preparing fixed phrases...')
    missing = [t for t in symbol_texts() if t not in speaker.prepare_many(symbol_texts())]
    if missing:  # fail at startup, never silently while reading
        raise RuntimeError(f'Thai voice cannot synthesize {len(missing)} phrases, e.g. {missing[:3]}')
    reader = FingerReader(FingertipTracker(), CalibrationTracker(DEFAULT_FORMAT), SpeechQueue(speaker))
    capture = _open_camera(args.camera, 1920, 1080, 30)
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open camera {args.camera}')
    latency = None
    if args.log_latency:
        log_file = args.log_latency.open('w', newline='', encoding='utf-8')
        latency = csv.writer(log_file)
        # entered_s: frame where the fingertip first reached the cell; played_s: playback started.
        latency.writerow(['event', 'entered_s', 'played_s', 'latency_s'])
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_s = time.perf_counter() - started
            status = reader.step(frame, frame_s)
            done_s = time.perf_counter() - started
            if latency is not None and reader.last_events and isinstance(reader.last_events[-1], EnterCell):
                latency.writerow([repr(reader.last_events[-1]), f'{reader.entered_s:.4f}', f'{done_s:.4f}',
                                  f'{done_s - reader.entered_s:.4f}'])
            if not args.headless:
                cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
                cv2.imshow('finger reader', cv2.resize(frame, (1280, 720)))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if latency is not None:
            log_file.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
