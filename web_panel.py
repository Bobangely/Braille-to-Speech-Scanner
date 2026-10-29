"""Teacher-panel web server (docs/web-api.md), run in a background thread of finger_reader.py.

The main loop calls ``PanelState.update`` every frame and ``set_frame`` only when
``wants_frame`` says a viewer is watching; the server never touches the camera or
the reader, so a broken request can never stop reading.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

import panel_settings

INDEX = Path(__file__).resolve().parent / 'web' / 'index.html'
STREAM_FPS = 5  # each streamed frame costs a resize + JPEG encode on the reader's CPU
STREAM_WIDTH = 960
log = logging.getLogger(__name__)


def _lines(page) -> list[list[dict]]:
    """Each line's words in reading order, as {index, text}."""
    lines = []
    for line in page.lines:
        words = dict.fromkeys(c.word_index for c in line.cells if c.word_index is not None)
        lines.append([{'index': w, 'text': page.words[w]} for w in words])
    return lines


def get_camera_device_names() -> list[str]:
    """Enumerate Windows video capture friendly names from registry."""
    names = []
    # 1. Physical cameras from KSCATEGORY_CAPTURE
    guid = '{e5323777-f976-4f5b-9b55-b94699c46e44}'
    base_path = fr'SYSTEM\CurrentControlSet\Control\DeviceClasses\{guid}'
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base_path) as k:
            count = winreg.QueryInfoKey(k)[0]
            for i in range(count):
                subkey_name = winreg.EnumKey(k, i)
                sub_path = fr'{base_path}\{subkey_name}\#GLOBAL\Device Parameters'
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, sub_path) as dp:
                        name, _ = winreg.QueryValueEx(dp, 'FriendlyName')
                        if name and name not in names:
                            names.append(name)
                except OSError:
                    pass
    except Exception:
        pass

    # 2. Virtual/software cameras from DirectShow Filter Category
    ds_path = r'SOFTWARE\Classes\CLSID\{860BB310-5D01-11d0-BD3B-00A0C911CE86}\Instance'
    try:
        import winreg
        for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(root, ds_path) as k:
                    count = winreg.QueryInfoKey(k)[0]
                    for i in range(count):
                        subkey_name = winreg.EnumKey(k, i)
                        try:
                            with winreg.OpenKey(k, subkey_name) as sub:
                                name, _ = winreg.QueryValueEx(sub, 'FriendlyName')
                                if name and name not in names:
                                    names.append(name)
                        except OSError:
                            pass
            except OSError:
                pass
    except Exception:
        pass
    return names


def scan_camera_devices(max_cams=6) -> list[dict]:
    """Scan and return available cameras with index and human-readable device names from registry.
    This queries the Windows device manager classes directly without opening hardware captures,
    guaranteeing it never collides with active streaming, never crashes the process, and returns instantly."""
    named_devices = get_camera_device_names()
    if named_devices:
        return [{'index': i, 'name': name} for i, name in enumerate(named_devices[:max_cams])]
    return [{'index': 0, 'name': 'EMEET SmartCam C60E 4K'}]


def scan_cameras(max_cams=6) -> list[int]:
    """Scan and return available camera indexes."""
    return [d['index'] for d in scan_camera_devices(max_cams)]


class PanelState:
    """The latest reader state and camera frame, shared between the main loop and the server."""

    def __init__(self, settings: dict, stream_fps: float = STREAM_FPS, reader=None):
        self.settings = settings
        self.reader = reader
        self.current_camera = 0
        self.requested_camera = None
        self.camera_devices = scan_camera_devices()
        self.available_cameras = [d['index'] for d in self.camera_devices]
        self.audio_config = {'engine': 'edge', 'rate': 1.0, 'volume': 100}
        self.history: list[dict] = []
        self._last_logged_word: str | None = None
        self._frame_gap_s = 1 / stream_fps
        self.viewers = 0
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._frame, self._frame_s, self._frame_id = None, float('-inf'), 0
        self._page, self._page_lines = None, []
        current_name = next((d['name'] for d in self.camera_devices if d['index'] == self.current_camera), f'กล้อง #{self.current_camera}')
        self._snap = {'status': 'waiting_page', 'markers': False, 'hand': False, 'fps': 0.0, 'symbol': None,
                      'dots': [], 'word': None, 'active_word': None, 'last_spoken': None, 'lines': [], 'muted': False,
                      'audio_config': dict(self.audio_config),
                      'camera_config': {'current': 0, 'current_name': current_name,
                                        'available': list(self.available_cameras),
                                        'devices': list(self.camera_devices),
                                        'total': len(self.camera_devices)},
                      'history': []}

    def update(self, reader, fps: float, last_spoken: str | None) -> None:
        self.reader = reader
        page = reader.page
        if page is not self._page:  # rebuild the word list only when a new page is learned
            self._page, self._page_lines = page, _lines(page) if page else []
        active = reader.active_cell
        cell = active[1] if active else None
        word = cell.word_index if cell else None
        speech = getattr(reader, 'speech', None)
        muted = getattr(speech, 'muted', False) if speech else False
        if speech:
            self.audio_config = {
                'engine': getattr(speech, 'engine', 'edge'),
                'rate': getattr(speech, 'rate', 1.0),
                'volume': getattr(speech, 'volume', 100)
            }
        if last_spoken and last_spoken not in ('—', 'กำลังจำหน้า', 'พร้อมอ่าน', 'ยกมือออกก่อน', 'วางหน้าให้เห็นมุมครบ', 'วางหน้าให้เห็นมุมทั้งหมด') and last_spoken != self._last_logged_word:
            self._last_logged_word = last_spoken
            import datetime
            now_time = datetime.datetime.now().strftime('%H:%M:%S')
            self.history.append({'text': last_spoken, 'time': now_time})
            if len(self.history) > 60:
                self.history.pop(0)

        current_name = next((d['name'] for d in self.camera_devices if d['index'] == self.current_camera), f'กล้อง #{self.current_camera}')
        snap = {'status': reader.status, 'markers': reader.calibration is not None, 'hand': reader.tip is not None,
                'fps': round(fps, 1), 'symbol': cell.name if cell else None,
                'dots': sorted(cell.dots) if cell and cell.dots else [],
                'word': page.words[word] if word is not None else None, 'active_word': word,
                'last_spoken': last_spoken, 'lines': self._page_lines, 'muted': muted,
                'audio_config': dict(self.audio_config),
                'camera_config': {'current': self.current_camera, 'current_name': current_name,
                                  'available': list(self.available_cameras),
                                  'devices': list(self.camera_devices),
                                  'total': len(self.camera_devices)},
                'history': list(self.history)}
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
        """Wait for a frame newer than ``after_id``; returns (frame, id), or (None, after_id) on timeout."""
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


def start(port: int, state: PanelState, speech, settings_path: Path, host: str = '0.0.0.0', reader=None) -> ThreadingHTTPServer:
    """Serve the panel on ``host:port`` from a daemon thread. Port in use raises OSError at once."""
    if reader is not None and getattr(state, 'reader', None) is None:
        state.reader = reader

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the reader's console readable
            log.debug('%s ' + fmt, self.address_string(), *args)

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
            elif path == '/api/state':
                self._json(200, state.snapshot())
            elif path == '/stream.mjpg':
                self._stream()
            else:
                self._json(404, {'error': 'ไม่พบหน้านี้'})

        def do_POST(self):
            path = self.path.split('?', 1)[0]
            if path == '/api/settings':
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
                        last = getattr(speech, 'last_spoken', None) or 'พร้อม'
                        if hasattr(speech, 'say'):
                            speech.say(last)
                        self._json(200, {'ok': True, 'action': 'replay', 'last_spoken': last})
                    elif action == 'speak':
                        text = str(body.get('text', '')).strip()
                        if text and hasattr(speech, 'say'):
                            speech.say(text)
                        self._json(200, {'ok': True, 'action': 'speak', 'text': text})
                    elif action == 'rescan':
                        target_reader = getattr(state, 'reader', None) or reader
                        if target_reader is not None and hasattr(target_reader, '_set'):
                            target_reader._set('learning', 'rescan requested by web panel')
                        self._json(200, {'ok': True, 'action': 'rescan'})
                    elif action == 'mute':
                        val = bool(body.get('value', True))
                        if speech is not None:
                            speech.muted = val
                            if val and hasattr(speech, 'speaker') and hasattr(speech.speaker, 'stop'):
                                speech.speaker.stop()
                        self._json(200, {'ok': True, 'muted': getattr(speech, 'muted', False)})
                    elif action == 'toggle_mute':
                        if speech is not None:
                            speech.muted = not getattr(speech, 'muted', False)
                            if speech.muted and hasattr(speech, 'speaker') and hasattr(speech.speaker, 'stop'):
                                speech.speaker.stop()
                        self._json(200, {'ok': True, 'muted': getattr(speech, 'muted', False)})
                    elif action == 'set_audio':
                        engine = body.get('engine')
                        rate = body.get('rate')
                        volume = body.get('volume')
                        preview = bool(body.get('preview', False))
                        res = {}
                        if speech is not None and hasattr(speech, 'set_audio_config'):
                            res = speech.set_audio_config(engine=engine, rate=rate, volume=volume)
                        state.audio_config.update(res)
                        if preview and speech is not None and hasattr(speech, 'say'):
                            sample = getattr(speech, 'last_spoken', None) or 'พร้อม'
                            speech.say(sample)
                        self._json(200, {'ok': True, 'audio_config': state.audio_config})
                    elif action == 'set_camera':
                        idx = int(body.get('index', 0))
                        state.requested_camera = idx
                        state.current_camera = idx
                        self._json(200, {'ok': True, 'camera': idx})
                    elif action == 'scan_cameras':
                        devs = scan_camera_devices()
                        state.camera_devices = devs
                        state.available_cameras = [d['index'] for d in devs]
                        current_name = next((d['name'] for d in devs if d['index'] == state.current_camera), f'กล้อง #{state.current_camera}')
                        self._json(200, {
                            'ok': True,
                            'available': state.available_cameras,
                            'devices': devs,
                            'total': len(devs),
                            'current': state.current_camera,
                            'current_name': current_name
                        })
                    elif action == 'clear_history':
                        state.history.clear()
                        state._last_logged_word = None
                        self._json(200, {'ok': True, 'history': []})
                    else:
                        self._json(400, {'error': f'ไม่รู้จักคำสั่ง: {action}'})
                except json.JSONDecodeError:
                    self._json(400, {'error': 'JSON ไม่ถูกต้อง'})
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
                pass  # the page closed or the network dropped
            finally:
                state.add_viewer(-1)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name='web-panel', daemon=True).start()
    return server
