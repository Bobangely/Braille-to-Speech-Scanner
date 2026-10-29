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
import logging
import sys
import time
from pathlib import Path

import panel_settings
from finger_state import EnterCell, FingerState
from page_calibration import DEFAULT_FORMAT, CalibrationTracker, PageFormat, page_format
from page_map import PageLearner, new_ink_dots, read_page as read_colored_page
from speech_cache import CachedWavSpeaker, HybridSpeaker
from speech_queue import UNKNOWN, SpeechQueue
from thai_speech import SYMBOL_NAMES, THAI_CONSONANT_NAMES

ROOT = Path(__file__).resolve().parent
log = logging.getLogger(__name__)
PLACE_PAGE = 'วางหน้าให้เห็นมุมทั้งหมด'
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
    parser.add_argument('--pad-fraction', type=float, default=0.1,
                        help='reading point from the fingertip toward the first joint: 0 = nail tip, 1 = joint')
    parser.add_argument('--debug-dir', type=Path, help='save the camera frame each time the status changes')
    parser.add_argument('--speech-mode', choices=['spell', 'word'],
                        help='spell: name every cell then the word; word: name cells of short words only')
    parser.add_argument('--spell-max-cells', type=int, help='in word mode, words up to this many cells are spelled (1-5)')
    parser.add_argument('--settings', type=Path, default=panel_settings.DEFAULT_PATH,
                        help='teacher-panel settings file; the options above override it and are saved')
    parser.add_argument('--web', type=int, metavar='PORT',
                        help='serve the teacher panel on this port to every device on the network (docs/web-api.md)')
    parser.add_argument('--web-host', default='127.0.0.1',
                        help='0.0.0.0 opens the panel to other devices on the network (no login)')
    parser.add_argument('--web-fps', type=int, default=5, metavar='N',
                        help='camera frames per second sent to the panel (1-30); more is smoother but costs CPU')
    args = parser.parse_args()
    if not 1 <= args.web_fps <= 30:
        parser.error('--web-fps ต้องอยู่ระหว่าง 1–30')
    # Check settings before the slow voice preparation, so a typo fails at once.
    cli_settings = {key: value for key, value in
                    (('mode', args.speech_mode), ('spell_max_cells', args.spell_max_cells)) if value is not None}
    try:
        settings = panel_settings.validate(cli_settings, panel_settings.load(args.settings))
    except ValueError as error:
        parser.error(str(error))
    fmt = page_format(args.marker_size_mm)

    speaker = CachedWavSpeaker(args.tts_thai_model, args.cache, args.tts_python, args.tts_thai_backend)
    if args.symbol_voice == 'edge':
        online = CachedWavSpeaker(None, args.symbol_cache, args.edge_python, 'edge')
        speaker = HybridSpeaker(online, speaker, fixed=None if args.word_voice == 'edge' else symbol_texts())
    print('preparing fixed phrases...')
    missing = [t for t in symbol_texts() if t not in speaker.prepare_many(symbol_texts())]
    if missing:  # fail at startup, never silently while reading
        raise RuntimeError(f'Thai voice cannot synthesize {len(missing)} phrases, e.g. {missing[:3]}')
    speech = SpeechQueue(speaker, background=True)
    panel_settings.apply_to(speech, settings)
    if cli_settings:
        panel_settings.save(args.settings, settings)
    reader = FingerReader(FingertipTracker(pad_fraction=args.pad_fraction), CalibrationTracker(fmt), speech, fmt)
    panel = None
    if args.web is not None:
        import web_panel
        panel = web_panel.PanelState(settings, stream_fps=args.web_fps)
        server = web_panel.start(args.web, panel, speech, args.settings, host=args.web_host)  # port in use: OSError now
        print(f'teacher panel: http://localhost:{server.server_address[1]}'
              + (f'  (other devices: http://<this IP>:{args.web})' if args.web_host != '127.0.0.1' else ''))
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
    fps, last_frame_s = 0.0, 0.0
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
            if panel is not None:
                fps = 0.9 * fps + 0.1 / max(frame_s - last_frame_s, 1e-3)
                last_frame_s = frame_s
                panel.update(reader, fps, speech.last_spoken)
            to_panel = panel is not None and panel.wants_frame(frame_s)
            if not args.headless or to_panel:
                draw_hand_skeletons(frame, reader.tracker.hand_landmarks)  # under the reading point
                if reader.calibration is not None:
                    draw_reading_zone(frame, reader.calibration, status, fmt)
                if reader.page is not None and reader.calibration is not None:
                    draw_overlay(frame, reader.page, reader.calibration, reader.tip, reader.active_cell)
                cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
            if to_panel:
                panel.set_frame(frame, frame_s)
            if not args.headless:
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
