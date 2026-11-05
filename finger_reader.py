"""อ่านเบรลล์ใต้ปลายนิ้วออกเสียงจากหน้าหนังสือแบบสองบรรทัด

ขั้นจำหน้า: เมื่อไม่มีมือในภาพ ปรับเทียบหน้าจาก marker ที่มุมและอ่านบรรทัดสีหนึ่งรอบ
ขั้นอ่าน: ทุกเฟรมแปลงปลายนิ้วเป็นมิลลิเมตรบนหน้า พูดชื่อสัญลักษณ์เมื่อเข้าเซลล์นูนใหม่
แล้วพูดคำเมื่อนิ้วออกจากคำไปทางขวา

Usage: finger_reader.py --tts-thai-model voices/th.onnx [--camera 0] [--log-latency out.csv]
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import socket
import sys
import time
from pathlib import Path

import panel_settings
from colored_braille import UnreadableBrailleFrame
from finger_state import EnterCell, FingerState
from page_calibration import DEFAULT_FORMAT, CalibrationTracker, PageFormat, page_format
from page_map import PageLearner, new_ink_dots, read_page as read_colored_page
from speech_cache import WORD_RATE, CachedWavSpeaker, HybridSpeaker
from speech_queue import UNKNOWN, SpeechQueue
from thai_speech import SYMBOL_NAMES, THAI_CONSONANT_NAMES

ROOT = Path(__file__).resolve().parent
log = logging.getLogger(__name__)
PLACE_PAGE = 'วางหน้าให้เห็นมุมทั้งหมด'
LIFT_HAND = 'ยกมือออกก่อน'
RECHECK_S = 1.0  # อ่านบรรทัดสีซ้ำทุกช่วงเวลานี้ตอนไม่มีมืออยู่ในภาพ
TIP_OFFSET_MM = 1.0  # shortcut: ค่าเดียวทุกคน ถ้าต่างกันตามขนาดนิ้วให้วัดจากภาพแทน
NEW_INK_DOTS = 3  # จำนวนจุดที่อยู่ตรงกระดาษเปล่าของหน้าเดิม ก่อนจะถือว่าเปลี่ยนหน้า


class FingerReader:
    """รับภาพกล้องหนึ่งเฟรม คืนสถานะ: waiting_page, learning หรือ reading"""

    def __init__(self, tracker, calibrator, speech: SpeechQueue, fmt: PageFormat = DEFAULT_FORMAT,
                 read_page=read_colored_page, required_maps: int = 3, tip_offset_mm: float = 0.0):
        self.tracker, self.calibrator, self.speech, self.fmt = tracker, calibrator, speech, fmt
        self.read_page, self.required_maps, self.tip_offset_mm = read_page, required_maps, tip_offset_mm
        self.point_mm = None  # จุดอ่านบนหน้าของเฟรมล่าสุด
        self.status = 'waiting_page'
        self.last_events: list = []
        self._told: str | None = None
        self._learner = PageLearner(required_maps)
        self._finger: FingerState | None = None
        self.calibration = self.tip = None  # ค่าของเฟรมล่าสุด ใช้วาดภาพตัวอย่าง
        self._hand_seen_s = float('-inf')
        self._unreadable = False  # บันทึก log ครั้งเดียวต่อช่วงที่อ่านหน้าไม่ได้
        self._verify = False  # หน้าถูกขยับ: ตรวจซ้ำทุกเฟรมที่ไม่มีมือจนกว่าจะแน่ใจ

    def step(self, frame, now_s: float) -> str:
        self.last_events, self.point_mm = [], None
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
            # มือลากกระดาษ: แผนที่อยู่ในหน่วย mm ของหน้าจึงอ่านต่อได้ แต่ต้องตรวจ
            # ว่ายังเป็นแผ่นเดิมก่อนจะเชื่อถืออีกครั้ง
            log.info('page moved; checking it is the same page')
            self._verify = True
        if self.status == 'learning':
            self._learn(frame, calibration, now_s)
        elif self.status == 'reading':
            # landmark ปลายนิ้วอยู่ค่อนไปทางซ้ายของนิ้วจริง จึงเลื่อนจุดอ่านไปทางที่นิ้วกำลังอ่าน
            self.point_mm = calibration.to_page([tip])[0] + (self.tip_offset_mm, 0) if tip is not None else None
            self.last_events = self._finger.update(None if self.point_mm is None else tuple(self.point_mm), now_s)
            self.speech.handle(self.last_events)
            # MediaPipe พลาดมือที่บังหน้าเป็นบางเฟรม จึงตรวจซ้ำเมื่อไม่มีมือครบหนึ่งวินาทีเท่านั้น
            if now_s - self._hand_seen_s >= RECHECK_S and (self._verify or now_s - self._checked_s >= RECHECK_S):
                self._recheck(frame, calibration, now_s)
        return self.status

    def _recheck(self, frame, calibration, now_s: float) -> None:
        """หน้าสมุดสันห่วงที่พลิกทับ marker เดิมจะเปลี่ยนแค่จุดเบรลล์"""
        self._checked_s = now_s
        # หมึกใหม่เท่านั้นที่แปลว่าหน้าใหม่ มือที่ MediaPipe มองไม่เห็นทำได้แค่บังจุด
        try:
            seen = self.read_page(frame, calibration, self.fmt)
        except UnreadableBrailleFrame:
            return  # เฟรมนี้บอกไม่ได้ว่าเป็นหน้าเดิมหรือไม่ รอตรวจรอบถัดไป
        same = new_ink_dots(seen, self._finger.page, self.fmt) < NEW_INK_DOTS
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
        """เวลาที่ปลายนิ้วถึงเซลล์ที่พูดล่าสุด (จุดเริ่มนับความหน่วงเสียง)"""
        return self._finger.entered_s if self._finger else None

    def _learn(self, frame, calibration, now_s: float) -> None:
        if self.tracker.hand_present:
            self._tell(LIFT_HAND)
            return
        try:
            seen = self.read_page(frame, calibration, self.fmt)
        except UnreadableBrailleFrame as error:
            # แสงน้อยทำให้มีจุดรบกวนได้ทั้งเฟรม ต้องไม่ทำให้โปรแกรมดับ เริ่มนับการอ่านที่ตรงกันใหม่
            if not self._unreadable:
                log.warning('page unreadable, check the light: %s', error)
            self._unreadable = True
            self._learner = PageLearner(self.required_maps)
            return
        self._unreadable = False
        page = self._learner.update(seen)
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
        """พูดข้อความสถานะครั้งเดียวต่อหนึ่งสถานะ"""
        if self._told != text:
            self._told = text
            self.speech.say(text)


WORD_BGR = ((0, 140, 255), (255, 120, 0), (180, 0, 220), (0, 170, 0), (0, 0, 220), (200, 160, 0))
CELL_BGR = (170, 170, 170)
ACTIVE_BGR = (0, 200, 0)
THAI_FONT = Path('C:/Windows/Fonts/tahoma.ttf')


def draw_overlay(image, page, calibration, tip, active) -> None:
    """วาดกรอบทุกเซลล์และคำบนหน้า ระบายเซลล์นูนใต้นิ้ว และเขียนชื่อกำกับ

    กรอบอยู่ในหน่วย mm ของหน้าและฉายผ่านค่าปรับเทียบ จึงเอียงตามกระดาษ
    เซลล์สีอยู่เหนือเซลล์นูนของมัน
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


ZONE_BGR = {'reading': (0, 200, 0), 'learning': (0, 200, 255)}  # ถ้าไม่เห็น marker จะไม่วาดกรอบ


def draw_reading_zone(image, calibration, status: str, fmt=DEFAULT_FORMAT, dash_px: int = 14) -> None:
    """กรอบเส้นประผ่านมุมนอกของ marker มีป้ายกำกับ และเปลี่ยนสีตามสถานะ (หน้าจอ dev)"""
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
    """แถบเข้มพร้อมข้อความไทยด้านล่าง (ฟอนต์ของ OpenCV ไม่มีตัวอักษรไทย)"""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    height = max(24, image.shape[0] // 18)
    bar = Image.fromarray(np.full((height, image.shape[1], 3), 30, np.uint8))
    try:
        font = ImageFont.truetype(str(THAI_FONT), int(height * 0.7))
    except OSError:
        font = ImageFont.load_default()  # shortcut: นอก Windows ไม่มีตัวอักษรไทยจาก Tahoma ใช้ได้กับเทสต์ แก้ถ้าต้องดูหน้าจอบนระบบอื่น
    ImageDraw.Draw(bar).text((10, height * 0.1), text, font=font, fill=(255, 255, 255))
    image[-height:] = np.asarray(bar)


def symbol_texts() -> list[str]:
    """ทุกข้อความตายตัวที่ระบบอาจพูด เพื่อสร้างเสียงไว้ก่อนหน้าแรก"""
    return [UNKNOWN, PLACE_PAGE, LIFT_HAND, *SYMBOL_NAMES.values(), *THAI_CONSONANT_NAMES.values(),
            *(f'พร้อม {n} บรรทัด' for n in range(1, 11))]


def main() -> int:
    import cv2

    from fingertip import PAD_FRACTION, FingertipTracker, draw_hand_skeletons, hand_size_px, open_camera

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
    parser.add_argument('--log-finger', type=Path,
                        help='CSV of the fingertip every frame, for tools/diagnostics/finger_lift.py')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--resolution', default='1920x1080', metavar='WxH',
                        help='camera size; 1280x720 on slow machines (e.g. ARM mini PC)')
    parser.add_argument('--marker-size-mm', type=float, help='printed marker side, as in the book template')
    parser.add_argument('--tip-offset-mm', type=float, default=TIP_OFFSET_MM,
                        help='move the reading point this far right on the page, toward the middle of the finger')
    parser.add_argument('--pad-fraction', type=float, default=PAD_FRACTION,
                        help='reading point from the fingertip toward the first joint: 0 = nail tip, 1 = joint')
    parser.add_argument('--debug-dir', type=Path, help='save the camera frame each time the status changes')
    parser.add_argument('--speech-mode', choices=['spell', 'word'],
                        help='spell: name every cell then the word; word: name cells of short words only')
    parser.add_argument('--spell-max-cells', type=int, help='in word mode, words up to this many cells are spelled (1-5)')
    parser.add_argument('--settings', type=Path, default=panel_settings.DEFAULT_PATH,
                        help='teacher-panel settings file; the options above override it and are saved')
    parser.add_argument('--web', type=int, metavar='PORT',
                        help='serve the teacher panel on this port, this machine only unless --web-host (docs/web-api.md)')
    parser.add_argument('--web-host', default='127.0.0.1',
                        help='0.0.0.0 opens the panel to other devices on the network; needs --web-pin')
    parser.add_argument('--web-pin', metavar='PIN', help='6 digits the teacher types once on each device')
    parser.add_argument('--web-fps', type=int, default=5, metavar='N',
                        help='camera frames per second sent to the panel (1-30); more is smoother but costs CPU')
    args = parser.parse_args()
    if not 1 <= args.web_fps <= 30:
        parser.error('--web-fps ต้องอยู่ระหว่าง 1–30')
    # เครือข่ายร่วมเห็นภาพกล้องและสั่งเครื่องพูดได้ จึงห้ามเปิดโดยไม่มี PIN
    needs_pin = args.web_host not in ('127.0.0.1', 'localhost')
    if (needs_pin or args.web_pin is not None) and not re.fullmatch(r'\d{6}', args.web_pin or ''):
        parser.error('--web-pin ต้องเป็นเลข 6 หลัก และต้องใส่เมื่อใช้ --web-host กับเครื่องอื่น')
    if not re.fullmatch(r'[1-9]\d*x[1-9]\d*', args.resolution.lower()):
        parser.error('--resolution ต้องเป็นรูปแบบ WxH เช่น 1280x720')
    # ตรวจค่าตั้งก่อนเตรียมเสียงที่ใช้เวลานาน เพื่อให้พิมพ์ผิดแล้วหยุดทันที
    cli_settings = {key: value for key, value in
                    (('mode', args.speech_mode), ('spell_max_cells', args.spell_max_cells)) if value is not None}
    try:
        settings = panel_settings.validate(cli_settings, panel_settings.load(args.settings))
    except ValueError as error:
        parser.error(str(error))
    fmt = page_format(args.marker_size_mm)

    speaker = CachedWavSpeaker(args.tts_thai_model, args.cache, args.tts_python, args.tts_thai_backend)
    if args.symbol_voice == 'edge':
        fixed = set(symbol_texts())
        online = CachedWavSpeaker(None, args.symbol_cache, args.edge_python, 'edge',
                                  rate_of=lambda text: None if text in fixed else WORD_RATE)
        speaker = HybridSpeaker(online, speaker, fixed=None if args.word_voice == 'edge' else fixed)
    print('preparing fixed phrases...')
    missing = [t for t in symbol_texts() if t not in speaker.prepare_many(symbol_texts())]
    if missing:  # ให้พังตั้งแต่เริ่มโปรแกรม ไม่ใช่เงียบไปตอนกำลังอ่าน
        raise RuntimeError(f'Thai voice cannot synthesize {len(missing)} phrases, e.g. {missing[:3]}')
    speech = SpeechQueue(speaker, background=True)
    panel_settings.apply_to(speech, settings)
    if cli_settings:
        panel_settings.save(args.settings, settings)
    reader = FingerReader(FingertipTracker(pad_fraction=args.pad_fraction), CalibrationTracker(fmt), speech, fmt,
                          tip_offset_mm=args.tip_offset_mm)
    panel = None
    if args.web is not None:
        import web_panel
        panel = web_panel.PanelState(settings, stream_fps=args.web_fps)
        server = web_panel.start(args.web, panel, speech, args.settings, host=args.web_host,
                                 pin=args.web_pin)  # ถ้าพอร์ตถูกใช้อยู่จะเกิด OSError ทันที
        port = server.server_address[1]
        print(f'teacher panel: http://localhost:{port}')
        if needs_pin:
            print(f'phone or iPad on the same network: http://{web_panel.lan_ip() or "<this IP>"}:{port}'
                  f'  or  http://{socket.gethostname()}.local:{port}')
    width, height = map(int, args.resolution.lower().split('x'))
    capture = open_camera(args.camera, width, height, 30)
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open camera {args.camera}')
    latency = None
    if args.log_latency:
        args.log_latency.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.log_latency.open('w', newline='', encoding='utf-8')
        latency = csv.writer(log_file)
        # entered_s: เฟรมที่ปลายนิ้วถึงเซลล์ครั้งแรก, played_s: เวลาที่เริ่มเล่นเสียง
        latency.writerow(['event', 'entered_s', 'played_s', 'latency_s'])
    finger_log = None
    if args.log_finger:
        args.log_finger.parent.mkdir(parents=True, exist_ok=True)
        finger_file = args.log_finger.open('w', newline='', encoding='utf-8')
        finger_log = csv.writer(finger_file)
        finger_log.writerow(['t_s', 'x_mm', 'y_mm', 'hand_px', 'entered'])
    started = time.perf_counter()
    fps, last_frame_s = 0.0, 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                log.error('camera %s stopped returning frames (unplugged or used by another program?)', args.camera)
                return 1
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
            if finger_log is not None:
                point = reader.point_mm if reader.point_mm is not None else ('', '')
                size = hand_size_px(reader.tracker.hand_landmarks, reader.tip, frame.shape[1], frame.shape[0])
                entered = next((e.name or UNKNOWN for e in reader.last_events if isinstance(e, EnterCell)), '')
                finger_log.writerow([f'{frame_s:.4f}', *point, '' if size is None else f'{size:.1f}', entered])
            if panel is not None:
                fps = 0.9 * fps + 0.1 / max(frame_s - last_frame_s, 1e-3)
                last_frame_s = frame_s
                panel.update(reader, fps, speech.last_spoken)
            to_panel = panel is not None and panel.wants_frame(frame_s)
            if not args.headless or to_panel:
                draw_hand_skeletons(frame, reader.tracker.hand_landmarks)  # วาดไว้ใต้จุดอ่าน
                if reader.calibration is not None:
                    draw_reading_zone(frame, reader.calibration, status, fmt)
                if reader.page is not None and reader.calibration is not None:
                    point = reader.point_mm
                    tip = None if point is None else reader.calibration.to_image([point])[0]
                    draw_overlay(frame, reader.page, reader.calibration, tip, reader.active_cell)
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
        if finger_log is not None:
            finger_file.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
