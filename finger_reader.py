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
import wave
from pathlib import Path

from finger_state import EnterCell, FingerState
from page_calibration import DEFAULT_FORMAT, CalibrationTracker, PageFormat, page_format
from page_map import PageLearner, new_ink_dots, read_page as read_colored_page
from speech_queue import UNKNOWN, SpeechQueue
from thai_speech import SYMBOL_NAMES, THAI_CONSONANT_NAMES, tts_spelling

ROOT = Path(__file__).resolve().parent
log = logging.getLogger(__name__)
PLACE_PAGE = 'วางหน้าให้เห็นมุมครบ'
LIFT_HAND = 'ยกมือออกก่อน'
RECHECK_S = 1.0  # re-read the colored lines this often while no hand is in view
NEW_INK_DOTS = 3  # dots where the learned page had paper, before the page counts as changed


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
        self.calibration = self.tip = None  # last frame's, for the preview
        self._hand_seen_s = float('-inf')
        self._verify = False  # page moved: recheck every hand-free frame until sure

    def step(self, frame, now_s: float) -> str:
        self.last_events = []
        calibration, changed = self.calibrator.update(frame, now_s)
        tip = self.tracker.update(frame, round(now_s * 1000))
        self.calibration, self.tip = calibration, tip
        if self.tracker.hand_present:
            self._hand_seen_s = now_s
        if calibration is None:
            self._set('waiting_page', 'no corner markers')
            self._tell(PLACE_PAGE)
        elif self.status == 'waiting_page':
            self._set('learning', 'markers found')
        elif changed and self.status == 'reading':
            # The hand dragged the sheet: the map is in page mm, so keep reading, but make sure
            # it is still the same sheet before trusting it again.
            log.info('page moved; checking it is the same page')
            self._verify = True
        if self.status == 'learning':
            self._learn(frame, calibration, now_s)
        elif self.status == 'reading':
            point = calibration.to_page([tip])[0] if tip is not None else None
            self.last_events = self._finger.update(None if point is None else tuple(point), now_s)
            self.speech.handle(self.last_events)
            # MediaPipe misses a covering hand for a frame now and then: recheck only after a full hand-free second.
            if now_s - self._hand_seen_s >= RECHECK_S and (self._verify or now_s - self._checked_s >= RECHECK_S):
                self._recheck(frame, calibration, now_s)
        return self.status

    def _recheck(self, frame, calibration, now_s: float) -> None:
        """A spiral-book page turned onto the same markers changes only its dots."""
        self._checked_s = now_s
        # Only new ink means a new page: a hand MediaPipe failed to see still only hides dots.
        same = new_ink_dots(self.read_page(frame, calibration, self.fmt), self._finger.page, self.fmt) < NEW_INK_DOTS
        self._mismatches = 0 if same else self._mismatches + 1
        self._verify = self._verify and not same
        if self._mismatches >= self.required_maps:
            self._set('learning', 'page content changed')

    @property
    def page(self):
        return self._finger.page if self._finger else None

    @property
    def active_cell(self):
        return self._finger.active_cell if self._finger else None

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
        self._set('reading', f'{len(page.lines)} lines learned')
        self._finger = FingerState(page)
        self._checked_s, self._mismatches, self._verify = now_s, 0, False
        self._tell(f'พร้อม {len(page.lines)} บรรทัด')

    def _set(self, status: str, reason: str) -> None:
        if status != self.status:
            log.info('%s -> %s: %s', self.status, status, reason)
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
        if backend != 'edge' and not Path(model).exists():  # Edge is an online voice: no model file
            raise FileNotFoundError(f'Thai speech model not found: {model}')
        self.model, self.backend = Path(model or ''), backend
        self.spell = tts_spelling if backend == 'piper' else str  # respellings were tuned on Piper
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.python = str(Path(python_executable).resolve()) if python_executable else sys.executable
        self._synthesize = synthesize or self._synthesize_in_tts_env
        self._play = play or _winsound_play

    def _path(self, text: str) -> Path:
        # Keyed by what is synthesized, so a new respelling makes new audio.
        key = hashlib.sha1(f'{self.backend}|{self.model}|{self.spell(text)}'.encode('utf-8')).hexdigest()
        return self.cache_dir / f'{key}.wav'

    def prepare_many(self, texts, lang: str = 'thai') -> set[str]:
        """Synthesize the uncached texts in one batch; return the texts ready to play."""
        texts = [t for t in texts if t.strip()]  # Piper writes no audio for blank text
        missing = [(t, self._path(t)) for t in dict.fromkeys(texts) if not self._path(t).is_file()]
        if missing:
            try:
                self._synthesize(self.model, [(self.spell(t), path) for t, path in missing])
            except Exception as error:
                log.warning('speech synthesis failed: %s', error)
            for _, path in missing:
                if path.is_file():
                    try:
                        trim_leading_silence(path)
                    except (OSError, EOFError, wave.Error) as error:  # not a PCM WAV: play it untrimmed
                        log.warning('cannot trim %s: %s', path.name, error)
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
        done = subprocess.run([self.python, '-X', 'utf8', '-c', BATCH_SCRIPT % str(ROOT)],
                              input=payload.encode('utf-8'), capture_output=True)
        if done.returncode:
            raise RuntimeError(done.stderr.decode('utf-8', 'replace')[-500:])
        if done.stderr.strip():  # texts that failed one by one; the rest are cached
            log.warning('speech synthesis skipped: %s', done.stderr.decode('utf-8', 'replace')[-500:])


class HybridSpeaker:
    """Texts in ``fixed`` (or every text when it is None) in a clear online voice, cached;
    the rest in Piper. A text without online audio (offline, service down) falls back to Piper."""

    def __init__(self, online, piper, fixed=None):
        self.online, self.piper = online, piper
        self.fixed = None if fixed is None else set(fixed)
        self._online_ready: set[str] = set()

    def prepare_many(self, texts, lang: str = 'thai') -> set[str]:
        texts = list(texts)
        wanted = [t for t in texts if self.fixed is None or t in self.fixed]
        self._online_ready = self._online_ready | self.online.prepare_many(wanted, lang)
        rest = [t for t in texts if t not in self._online_ready]
        return {t for t in texts if t in self._online_ready} | self.piper.prepare_many(rest, lang)

    def speak(self, text: str, lang: str = 'thai') -> bool:
        return (self.online if text in self._online_ready else self.piper).speak(text, lang)

    def stop(self) -> None:
        self.piper.stop()  # one shared player: stopping either stops playback


def trim_leading_silence(path: Path, keep_s: float = 0.02, threshold: float = 0.05) -> None:
    """Cut the voice's silent lead-in (~0.25 s, up to 1 s) so playback starts with the sound."""
    import numpy as np

    with wave.open(str(path)) as source:
        params, frames = source.getparams(), source.readframes(source.getnframes())
    if params.sampwidth != 2:
        return
    samples = np.frombuffer(frames, np.int16).reshape(-1, params.nchannels)
    level = np.abs(samples.astype(np.int32)).max(axis=1)
    loud = np.nonzero(level > threshold * max(1, level.max()))[0]
    start = max(0, int(loud[0]) - round(keep_s * params.framerate)) if len(loud) else 0
    if start:
        with wave.open(str(path), 'wb') as out:
            out.setparams(params)
            out.writeframes(samples[start:].tobytes())


WORD_BGR = ((0, 140, 255), (255, 120, 0), (180, 0, 220), (0, 170, 0), (0, 0, 220), (200, 160, 0))
CELL_BGR = (170, 170, 170)
ACTIVE_BGR = (0, 200, 0)
THAI_FONT = Path('C:/Windows/Fonts/tahoma.ttf')


def draw_overlay(image, page, calibration, tip, active) -> None:
    """Outline every cell and word of the page, fill the finger's raised cell, label it.

    Rectangles are in page mm and projected through the calibration, so they
    follow the paper's tilt. The colored cell sits above its raised cell.
    """
    import cv2
    import numpy as np

    def box(x1, y1, x2, y2):
        corners = calibration.to_image([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return [np.round(corners).astype(np.int32)]

    for line in page.lines:
        colored_top = line.raised_y1_mm - 2 * DEFAULT_FORMAT.dot_spacing_mm - 2.5
        words: dict[int, list] = {}
        for cell in line.cells:
            if cell.word_index is None:
                continue
            words.setdefault(cell.word_index, []).append(cell)
            for y1, y2 in ((colored_top, line.raised_y1_mm), (line.raised_y1_mm, line.raised_y2_mm)):
                cv2.polylines(image, box(cell.x1_mm, y1, cell.x2_mm, y2), True, CELL_BGR, 1)
        for word, cells in words.items():
            cv2.polylines(image, box(cells[0].x1_mm, colored_top, cells[-1].x2_mm, line.raised_y2_mm),
                          True, WORD_BGR[word % len(WORD_BGR)], 2)
    if active is not None:
        line, cell = active
        fill = image.copy()
        cv2.fillPoly(fill, box(cell.x1_mm, line.raised_y1_mm, cell.x2_mm, line.raised_y2_mm), ACTIVE_BGR)
        cv2.addWeighted(fill, 0.5, image, 0.5, 0, dst=image)
    if tip is not None:
        cv2.circle(image, tuple(map(int, tip)), 9, (0, 0, 0), -1)
        cv2.circle(image, tuple(map(int, tip)), 6, ACTIVE_BGR, -1)
    if active is not None:
        cell = active[1]
        word = page.words[cell.word_index] if cell.word_index is not None else ''
        _thai_bar(image, f'กำลังอ่าน: {cell.name or UNKNOWN}    คำ: {word}')


ZONE_BGR = {'reading': (0, 200, 0), 'learning': (0, 200, 255)}  # no frame without markers


def draw_reading_zone(image, calibration, status: str, fmt=DEFAULT_FORMAT, dash_px: int = 14) -> None:
    """Dashed frame through the markers' outer corners, labelled, coloured by reader status (dev view)."""
    import cv2
    import numpy as np

    m = fmt.margin_mm
    corners = calibration.to_image([[m, m], [fmt.width_mm - m, m],
                                    [fmt.width_mm - m, fmt.height_mm - m], [m, fmt.height_mm - m]])
    colour = ZONE_BGR.get(status, (200, 200, 200))
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        length = float(np.hypot(*(end - start)))
        for t in np.arange(0.0, length, 2 * dash_px):
            a = start + (end - start) * t / length
            b = start + (end - start) * min(t + dash_px, length) / length
            cv2.line(image, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), colour, 2, cv2.LINE_AA)
    x, y = np.round(corners[0]).astype(int)
    cv2.putText(image, f'READING ZONE: {status}', (int(x), max(15, int(y) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)


def _thai_bar(image, text: str) -> None:
    """Dark bar with Thai text at the bottom (OpenCV fonts have no Thai glyphs)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    height = max(24, image.shape[0] // 18)
    bar = Image.fromarray(np.full((height, image.shape[1], 3), 30, np.uint8))
    try:
        font = ImageFont.truetype(str(THAI_FONT), int(height * 0.7))
    except OSError:
        font = ImageFont.load_default()  # shortcut: no Thai glyphs without Tahoma; fine off Windows tests
    ImageDraw.Draw(bar).text((10, height * 0.1), text, font=font, fill=(255, 255, 255))
    image[-height:] = np.asarray(bar)


# Runs inside the Piper/MMS/Edge environment: one model load, one failed text never stops the rest.
# Edge waits on the network, so its texts go 8 at a time (8 words: 1.1 s instead of ~7 s).
BATCH_SCRIPT = '''import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, %r)
from piper_tts import synthesize_wav
job = json.load(sys.stdin)
def one(item):
    text, out = item
    tmp = out + ".tmp.wav"
    try:
        synthesize_wav(job["model"], text, tmp, backend=job["backend"])
        os.replace(tmp, out)
    except Exception as error:
        print(repr(text), error, file=sys.stderr)
with ThreadPoolExecutor(8 if job["backend"] == "edge" else 1) as pool:
    list(pool.map(one, job["jobs"]))
'''


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
    from tools.diagnostics.hand_landmarker_spike import _open_camera, draw_hand_skeletons

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%H:%M:%S')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--tts-thai-model', type=Path, required=True)
    parser.add_argument('--tts-thai-backend', choices=['piper', 'mms'], default='piper')
    parser.add_argument('--tts-python', help='Python executable of the Piper/MMS environment')
    parser.add_argument('--cache', type=Path, default=ROOT / 'output' / 'finger_speech')
    parser.add_argument('--symbol-voice', choices=['piper', 'edge'], default='piper',
                        help='edge: symbol names and messages in the online Edge voice (internet only the '
                             'first time; cached after), page words still in Piper')
    parser.add_argument('--word-voice', choices=['piper', 'edge'], default='piper',
                        help='edge: page words in the Edge voice too (needs internet for new words; '
                             'Piper speaks them when offline); requires --symbol-voice edge')
    parser.add_argument('--edge-python', help='Python executable of the edge-tts environment')
    parser.add_argument('--symbol-cache', type=Path, default=ROOT / 'output' / 'symbol_voice_edge')
    parser.add_argument('--log-latency', type=Path, help='CSV of per-event timings in seconds')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--marker-size-mm', type=float, help='printed marker side, as in the book template')
    parser.add_argument('--pad-fraction', type=float, default=0.25,
                        help='reading point from the fingertip toward the first joint: 0 = nail tip, 1 = joint')
    parser.add_argument('--debug-dir', type=Path, help='save the camera frame each time the status changes')
    args = parser.parse_args()
    fmt = page_format(args.marker_size_mm)

    speaker = CachedWavSpeaker(args.tts_thai_model, args.cache, args.tts_python, args.tts_thai_backend)
    if args.symbol_voice == 'edge':
        online = CachedWavSpeaker(None, args.symbol_cache, args.edge_python, 'edge')
        speaker = HybridSpeaker(online, speaker, fixed=None if args.word_voice == 'edge' else symbol_texts())
    print('preparing fixed phrases...')
    missing = [t for t in symbol_texts() if t not in speaker.prepare_many(symbol_texts())]
    if missing:  # fail at startup, never silently while reading
        raise RuntimeError(f'Thai voice cannot synthesize {len(missing)} phrases, e.g. {missing[:3]}')
    reader = FingerReader(FingertipTracker(pad_fraction=args.pad_fraction), CalibrationTracker(fmt),
                          SpeechQueue(speaker, background=True), fmt)
    capture = _open_camera(args.camera, 1920, 1080, 30)
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open camera {args.camera}')
    latency = None
    if args.log_latency:
        args.log_latency.parent.mkdir(parents=True, exist_ok=True)
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
            previous = reader.status
            status = reader.step(frame, frame_s)
            done_s = time.perf_counter() - started
            if args.debug_dir and status != previous:
                args.debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.debug_dir / f'{time.strftime("%H%M%S")}_{previous}_to_{status}.jpg'), frame)
            if latency is not None and reader.last_events and isinstance(reader.last_events[-1], EnterCell):
                latency.writerow([repr(reader.last_events[-1]), f'{reader.entered_s:.4f}', f'{done_s:.4f}',
                                  f'{done_s - reader.entered_s:.4f}'])
            if not args.headless:
                draw_hand_skeletons(frame, reader.tracker.hand_landmarks)  # under the reading point
                if reader.calibration is not None:
                    draw_reading_zone(frame, reader.calibration, status, fmt)
                if reader.page is not None and reader.calibration is not None:
                    draw_overlay(frame, reader.page, reader.calibration, reader.tip, reader.active_cell)
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
