"""web server ของหน้าเว็บครู (docs/web-api.md) รันใน background thread ของ finger_reader.py

ลูปหลักเรียก ``PanelState.update`` ทุกเฟรม และเรียก ``set_frame`` เฉพาะเมื่อ ``wants_frame``
บอกว่ามีคนดู server ไม่แตะกล้องหรือตัวอ่าน คำขอที่ผิดจึงหยุดการอ่านไม่ได้
"""

from __future__ import annotations

import hmac
import json
import logging
import socket
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

import panel_settings

INDEX = Path(__file__).resolve().parent / 'web' / 'index.html'
STREAM_FPS = 5  # ภาพที่ส่งแต่ละเฟรมต้องย่อและเข้ารหัส JPEG ใช้ CPU เครื่องอ่าน
STREAM_WIDTH = 960
log = logging.getLogger(__name__)


def _lines(page) -> list[list[dict]]:
    """คำของแต่ละบรรทัดตามลำดับการอ่าน ในรูป {index, text}"""
    lines = []
    for line in page.lines:
        words = dict.fromkeys(c.word_index for c in line.cells if c.word_index is not None)
        lines.append([{'index': w, 'text': page.words[w]} for w in words])
    return lines


class PanelState:
    """สถานะล่าสุดและภาพกล้อง ที่ลูปหลักกับ server ใช้ร่วมกัน"""

    def __init__(self, settings: dict, stream_fps: float = STREAM_FPS):
        self.settings = settings
        self.audio_config = {'engine': 'edge', 'volume': 100}
        self._frame_gap_s = 1 / stream_fps
        self.viewers = 0
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._frame, self._frame_s, self._frame_id = None, float('-inf'), 0
        self._page, self._page_lines = None, []
        self._snap = {'status': 'waiting_page', 'markers': False, 'hand': False, 'fps': 0.0, 'symbol': None,
                      'dots': [], 'word': None, 'active_word': None, 'last_spoken': None, 'lines': [], 'muted': False,
                      'audio_config': dict(self.audio_config)}

    def update(self, reader, fps: float, last_spoken: str | None) -> None:
        page = reader.page
        if page is not self._page:  # สร้างรายการคำใหม่เฉพาะตอนจำหน้าใหม่
            self._page, self._page_lines = page, _lines(page) if page else []
        active = reader.active_cell
        cell = active[1] if active else None
        word = cell.word_index if cell else None
        speech = reader.speech
        self.audio_config = {'engine': speech.engine, 'volume': speech.volume}
        snap = {'status': reader.status, 'markers': reader.calibration is not None, 'hand': reader.tip is not None,
                'fps': round(fps, 1), 'symbol': cell.name if cell else None,
                'dots': sorted(cell.dots) if cell and cell.dots else [],
                'word': page.words[word] if word is not None else None, 'active_word': word,
                'last_spoken': last_spoken, 'lines': self._page_lines, 'muted': speech.muted,
                'audio_config': dict(self.audio_config)}
        with self._lock:
            self._snap = snap

    def snapshot(self) -> dict:
        with self._lock:
            return {**self._snap, 'settings': dict(self.settings)}

    def wants_frame(self, now_s: float) -> bool:
        return self.settings['camera'] and self.viewers > 0 and now_s - self._frame_s >= self._frame_gap_s - 1e-9

    def set_frame(self, frame, now_s: float) -> None:
        with self._new_frame:
            self._frame, self._frame_s = frame, now_s
            self._frame_id += 1
            self._new_frame.notify_all()

    def next_frame(self, after_id: int, timeout_s: float = 2.0):
        """รอเฟรมที่ใหม่กว่า ``after_id`` คืน (frame, id) หรือ (None, after_id) เมื่อหมดเวลา"""
        with self._new_frame:
            self._new_frame.wait_for(lambda: self._frame_id != after_id, timeout_s)
            return (self._frame, self._frame_id) if self._frame_id != after_id else (None, after_id)

    def add_viewer(self, delta: int) -> None:
        with self._lock:
            self.viewers += delta


def _jpeg(frame) -> bytes:
    height, width = frame.shape[:2]
    if width > STREAM_WIDTH:
        frame = cv2.resize(frame, (STREAM_WIDTH, round(height * STREAM_WIDTH / width)), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise RuntimeError('cannot encode camera frame')
    return data.tobytes()


def _qr_png(text: str) -> bytes:
    """QR code เป็น PNG ด้วย OpenCV ที่ติดตั้งอยู่แล้ว ไม่ต้องใช้อินเทอร์เน็ต"""
    modules = cv2.QRCodeEncoder.create().encode(text)
    # ขอบขาว 4 ช่องตามมาตรฐาน แล้วขยายแบบไม่เบลอให้กล้องมือถืออ่านง่าย
    modules = cv2.copyMakeBorder(modules, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    image = cv2.resize(modules, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('cannot encode QR code')
    return data.tobytes()


def lan_ip() -> str | None:
    """IP ของเครื่องนี้ในวง WiFi หรือ hotspot ปัจจุบัน UDP connect แค่เลือกเส้นทาง ไม่ส่งข้อมูลจริง"""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(('10.255.255.255', 1))
            return probe.getsockname()[0]
        except OSError:
            return None


def start(port: int, state: PanelState, speech, settings_path: Path, host: str = '127.0.0.1',
          pin: str | None = None) -> ThreadingHTTPServer:
    """เปิดหน้าเว็บที่ ``host:port`` ใน daemon thread ถ้าพอร์ตถูกใช้จะเกิด OSError ทันที

    เมื่อมี ``pin`` จะเปิดได้แค่ตัวหน้าเว็บโดยไม่ต้องใช้ PIN ส่วน HTTP กันคนทั่วไปได้
    แต่ไม่กันคนที่ดักข้อมูลในเครือข่าย
    """
    login_lock = threading.Lock()  # ลองผิดได้ครั้งละหนึ่งทั้งเครื่อง จึงเดา PIN รัวๆ ไม่ได้

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # ให้หน้าจอ console ของเครื่องอ่านดูง่าย
            log.debug('%s ' + fmt, self.address_string(), *args)

        def _unlocked(self) -> bool:
            if pin is None:
                return True
            cookie = SimpleCookie(self.headers.get('Cookie', '')).get('pin')
            return cookie is not None and hmac.compare_digest(cookie.value.encode(), pin.encode())

        def _login(self) -> None:
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                given = str(body['pin'])
            except (ValueError, KeyError, TypeError):
                self._json(400, {'error': 'คำขอไม่ถูกต้อง'})
                return
            with login_lock:
                ok = pin is None or hmac.compare_digest(given.encode(), pin.encode())
                if not ok:
                    time.sleep(1)
            if not ok:
                self._json(401, {'error': 'PIN ไม่ถูกต้อง'})
                return
            self.send_response(200)
            self.send_header('Set-Cookie', f'pin={pin}; Path=/; Max-Age=2592000; HttpOnly; SameSite=Strict')
            self.send_header('Content-Length', '0')
            self.end_headers()

        def _connect(self, path: str) -> None:
            """การ์ด QR บนจอสถานี: ที่อยู่และ PIN ให้เฉพาะเครื่องสถานีเอง เครื่องอื่นเห็นเป็น 404"""
            if self.client_address[0] not in ('127.0.0.1', '::1'):
                self._json(404, {'error': 'ไม่พบหน้านี้'})
                return
            network = host not in ('127.0.0.1', 'localhost')
            ip = (lan_ip() if host == '0.0.0.0' else host) if network else None
            url = f'http://{ip}:{self.server.server_address[1]}' if ip else None
            if path == '/api/connect':
                self._json(200, {'network': network, 'url': url, 'pin': pin})
            elif url is None:
                self._json(404, {'error': 'ยังไม่ได้เปิดให้เครื่องอื่นเข้า'})
            else:
                self._send(200, _qr_png(url), 'image/png')

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data: dict) -> None:
            self._send(code, json.dumps(data, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == '/':
                self._send(200, INDEX.read_bytes(), 'text/html; charset=utf-8')
            elif path in ('/api/connect', '/qr.png'):
                self._connect(path)
            elif not self._unlocked():
                self._json(401, {'error': 'ต้องใส่ PIN'})
            elif path == '/api/state':
                self._json(200, state.snapshot())
            elif path == '/stream.mjpg':
                self._stream()
            else:
                self._json(404, {'error': 'ไม่พบหน้านี้'})

        def do_POST(self):
            path = self.path.split('?', 1)[0]
            if path == '/api/login':
                self._login()
            elif not self._unlocked():
                self._json(401, {'error': 'ต้องใส่ PIN'})
            elif path == '/api/settings':
                try:
                    partial = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'null')
                    state.settings = panel_settings.apply_settings(speech, settings_path, partial)
                except json.JSONDecodeError:
                    self._json(400, {'error': 'JSON ไม่ถูกต้อง'})
                    return
                except ValueError as error:
                    self._json(400, {'error': str(error)})
                    return
                except OSError:
                    log.exception('cannot save panel settings')
                    self._json(500, {'error': 'บันทึกค่าตั้งไม่ได้'})
                    return
                self._json(200, state.settings)
            elif path == '/api/action':
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
                    if not isinstance(body, dict):
                        self._json(400, {'error': 'คำขอต้องเป็น JSON object'})
                        return
                    action = body.get('action')
                    if action == 'replay':
                        last = speech.last_spoken or 'พร้อม'
                        speech.say(last)
                        self._json(200, {'ok': True, 'action': 'replay', 'last_spoken': last})
                    elif action == 'speak':
                        text = str(body.get('text', '')).strip()
                        if text:
                            speech.say(text)
                        self._json(200, {'ok': True, 'action': 'speak', 'text': text})
                    elif action == 'toggle_mute':
                        speech.muted = not speech.muted
                        if speech.muted:
                            speech.speaker.stop()
                        self._json(200, {'ok': True, 'muted': speech.muted})
                    elif action == 'set_audio':
                        preview = bool(body.get('preview', False))
                        state.audio_config.update(speech.set_audio_config(engine=body.get('engine'),
                                                                          volume=body.get('volume')))
                        if preview:
                            speech.say(speech.last_spoken or 'พร้อม')
                        self._json(200, {'ok': True, 'audio_config': state.audio_config})
                    else:
                        self._json(400, {'error': f'ไม่รู้จักคำสั่ง: {action}'})
                except json.JSONDecodeError:
                    self._json(400, {'error': 'JSON ไม่ถูกต้อง'})
                except ValueError as error:
                    self._json(400, {'error': str(error)})
                except Exception as error:
                    log.exception('cannot handle action')
                    self._json(500, {'error': str(error)})
            else:
                self._json(404, {'error': 'ไม่พบหน้านี้'})

        def _stream(self):
            if not state.settings['camera']:
                self._json(503, {'error': 'ภาพกล้องปิดอยู่'})
                return
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            state.add_viewer(1)
            frame_id = 0
            try:
                while state.settings['camera']:
                    frame, frame_id = state.next_frame(frame_id)
                    if frame is None:
                        continue
                    jpeg = _jpeg(frame)
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                                     + f'Content-Length: {len(jpeg)}\r\n\r\n'.encode() + jpeg + b'\r\n')
                    try:
                        self.wfile.flush()
                    except (ConnectionError, OSError):
                        break
            except (ConnectionError, OSError):
                pass  # ปิดหน้าเว็บหรือเครือข่ายหลุด
            finally:
                state.add_viewer(-1)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name='web-panel', daemon=True).start()
    return server
