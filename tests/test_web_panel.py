"""Teacher-panel web server: the routes and JSON of docs/web-api.md, on a real socket."""
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from page_layout import CellSlot, LinePair, PageMap
from panel_settings import DEFAULTS
from web_panel import PanelState, start


def page():
    cells = (CellSlot(0, 0.0, 6.0, frozenset({1}), 'ชอช้าง', 0), CellSlot(1, 6.0, 12.0, frozenset({2}), 'สระอา', 0),
             CellSlot(2, 12.0, 18.0, frozenset(), None, None), CellSlot(3, 18.0, 24.0, frozenset({1}), 'กอไก่', 1))
    second = (CellSlot(0, 0.0, 6.0, frozenset({1}), 'มอม้า', 2),)
    return PageMap((LinePair(0, 10.0, 20.0, cells), LinePair(1, 30.0, 40.0, second)), ('ชา', 'ก', 'ม'))


def reader(status='reading', active=True):
    p = page() if status == 'reading' else None
    line = p.lines[0] if p else None
    return SimpleNamespace(status=status, calibration=object() if status != 'waiting_page' else None,
                           tip=(1, 2) if active else None, page=p,
                           active_cell=(line, line.cells[1]) if p and active else None,
                           speech=SimpleNamespace(engine='edge', volume=100, muted=False))


class PanelStateTests(unittest.TestCase):
    def test_snapshot_reading(self):
        state = PanelState(dict(DEFAULTS))
        state.update(reader(), fps=24.64, last_spoken='ชา')
        snap = state.snapshot()
        self.assertEqual(snap['status'], 'reading')
        self.assertTrue(snap['markers'] and snap['hand'])
        self.assertEqual((snap['symbol'], snap['word'], snap['active_word']), ('สระอา', 'ชา', 0))
        self.assertEqual(snap['last_spoken'], 'ชา')
        self.assertEqual(snap['fps'], 24.6)
        self.assertEqual(snap['lines'], [[{'index': 0, 'text': 'ชา'}, {'index': 1, 'text': 'ก'}],
                                         [{'index': 2, 'text': 'ม'}]])
        self.assertEqual(snap['settings'], DEFAULTS)

    def test_snapshot_without_page(self):
        state = PanelState(dict(DEFAULTS))
        state.update(reader('waiting_page', active=False), fps=0.0, last_spoken=None)
        snap = state.snapshot()
        self.assertEqual((snap['lines'], snap['symbol'], snap['word'], snap['active_word']), ([], None, None, None))
        self.assertFalse(snap['markers'] or snap['hand'])

    def test_snapshot_before_first_update(self):
        snap = PanelState(dict(DEFAULTS)).snapshot()
        self.assertEqual(snap['status'], 'waiting_page')
        self.assertEqual(snap['lines'], [])

    def test_stream_fps_is_configurable(self):
        state = PanelState(dict(DEFAULTS), stream_fps=15)
        state.viewers = 1
        state.set_frame(np.zeros((4, 4, 3), np.uint8), 0.0)
        self.assertFalse(state.wants_frame(0.05))
        self.assertTrue(state.wants_frame(1 / 15))

    def test_wants_frame_only_with_viewer_and_camera(self):
        state = PanelState(dict(DEFAULTS))
        self.assertFalse(state.wants_frame(0.0))
        state.viewers = 1
        self.assertTrue(state.wants_frame(0.0))
        state.set_frame(np.zeros((4, 4, 3), np.uint8), 0.0)
        self.assertFalse(state.wants_frame(0.1))  # max 5 fps
        self.assertTrue(state.wants_frame(0.2))
        state.settings = {**DEFAULTS, 'camera': False}
        self.assertFalse(state.wants_frame(1.0))


class PinTests(unittest.TestCase):
    PIN = '482913'

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.speech = SimpleNamespace(mode='spell', spell_max_cells=2, muted=False)
        self.server = start(0, PanelState(dict(DEFAULTS)), self.speech, Path(self._dir.name) / 's.json', pin=self.PIN)
        self.base = f'http://127.0.0.1:{self.server.server_address[1]}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self._dir.cleanup()

    def call(self, path, body=None, cookie=None):
        request = urllib.request.Request(self.base + path, data=body, headers={'Cookie': cookie} if cookie else {})
        try:
            with urllib.request.urlopen(request, timeout=5) as r:
                return r.status, r.headers.get('Set-Cookie')
        except urllib.error.HTTPError as error:
            return error.code, None

    def test_everything_but_the_page_needs_the_pin(self):
        self.assertEqual(self.call('/')[0], 200)
        for path, body in (('/api/state', None), ('/stream.mjpg', None), ('/api/settings', b'{"mode": "word"}'),
                           ('/api/action', b'{"action": "toggle_mute"}')):
            with self.subTest(path=path):
                self.assertEqual(self.call(path, body)[0], 401)
        self.assertEqual((self.speech.muted, self.speech.mode), (False, 'spell'))

    def test_right_pin_gives_a_cookie_that_opens_the_api(self):
        status, cookie = self.call('/api/login', json.dumps({'pin': self.PIN}).encode())
        self.assertEqual(status, 200)
        self.assertIn('HttpOnly', cookie)
        self.assertIn('SameSite=Strict', cookie)
        self.assertEqual(self.call('/api/state', cookie=cookie.split(';')[0])[0], 200)

    def test_wrong_pin_is_refused_and_slowed_down(self):
        started = time.monotonic()
        for body in (b'{"pin": "000000"}', b'{"pin": "\xe0\xb8\x81"}', b'[1]', b'not json'):
            with self.subTest(body=body):
                self.assertIn(self.call('/api/login', body)[0], (400, 401))
        self.assertGreaterEqual(time.monotonic() - started, 1.0)
        self.assertEqual(self.call('/api/state', cookie='pin=000000')[0], 401)


class WebOptionTests(unittest.TestCase):
    def run_reader(self, *options):
        return subprocess.run([sys.executable, 'finger_reader.py', '--tts-thai-model', 'x', '--web', '8080', *options],
                              capture_output=True, cwd=Path(__file__).resolve().parent.parent, timeout=60)

    def test_network_access_refuses_to_start_without_a_six_digit_pin(self):
        for options in (['--web-host', '0.0.0.0'], ['--web-host', '0.0.0.0', '--web-pin', '1234']):
            with self.subTest(options=options):
                done = self.run_reader(*options)
                self.assertEqual(done.returncode, 2)
                self.assertIn(b'error: --web-pin', done.stderr)


class ServerTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / 'panel_settings.json'
        self.speech = SimpleNamespace(mode='spell', spell_max_cells=2, muted=False)
        self.state = PanelState(dict(DEFAULTS))
        self.state.update(reader(), fps=10.0, last_spoken=None)
        self.server = start(0, self.state, self.speech, self.path, host='127.0.0.1')
        self.base = f'http://127.0.0.1:{self.server.server_address[1]}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self._dir.cleanup()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as r:
            return r.status, r.headers.get('Content-Type'), r.read()

    def post(self, body: bytes):
        request = urllib.request.Request(self.base + '/api/settings', data=body, method='POST',
                                         headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_index_page(self):
        status, kind, body = self.get('/')
        self.assertEqual(status, 200)
        self.assertIn('text/html', kind)
        self.assertIn('สถานีอ่านเบรลล์'.encode(), body)

    def test_state_json(self):
        status, kind, body = self.get('/api/state')
        self.assertEqual(status, 200)
        self.assertEqual(kind, 'application/json; charset=utf-8')
        self.assertEqual(json.loads(body)['word'], 'ชา')

    def test_post_settings_applies_and_saves(self):
        status, body = self.post(b'{"mode": "word", "spell_max_cells": 3}')
        self.assertEqual((status, body), (200, {'mode': 'word', 'spell_max_cells': 3, 'camera': True}))
        self.assertEqual(self.speech.mode, 'word')
        self.assertEqual(json.loads(self.get('/api/state')[2])['settings']['mode'], 'word')

    def test_post_invalid_is_400(self):
        for body in (b'{"spell_max_cells": 9}', b'not json', b'[1]'):
            with self.subTest(body=body):
                status, reply = self.post(body)
                self.assertEqual(status, 400)
                self.assertTrue(reply['error'])
        self.assertEqual(self.speech.mode, 'spell')

    def test_post_save_failure_is_500(self):
        with mock.patch('panel_settings.os.replace', side_effect=PermissionError('in use')):
            status, reply = self.post(b'{"mode": "word"}')
        self.assertEqual((status, reply), (500, {'error': 'บันทึกค่าตั้งไม่ได้'}))
        self.assertEqual(self.speech.mode, 'spell')

    def test_stream_503_when_camera_off(self):
        self.post(b'{"camera": false}')
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get('/stream.mjpg')
        self.assertEqual(caught.exception.code, 503)

    def test_stream_sends_jpeg_parts(self):
        def feed():
            for i in range(20):
                if self.state.wants_frame(i * 0.2):
                    self.state.set_frame(np.full((1080, 1920, 3), 128, np.uint8), i * 0.2)
                threading.Event().wait(0.05)
        threading.Thread(target=feed, daemon=True).start()
        with urllib.request.urlopen(self.base + '/stream.mjpg', timeout=5) as r:
            self.assertEqual(r.headers.get('Content-Type'), 'multipart/x-mixed-replace; boundary=frame')
            chunk = r.read(4096)
        self.assertIn(b'--frame', chunk)
        self.assertIn(b'\xff\xd8', chunk)  # JPEG start marker

    def test_post_action(self):
        # test replay
        self.speech.last_spoken = 'กล้วย'
        self.speech.say = mock.MagicMock()
        req = urllib.request.Request(self.base + '/api/action', data=b'{"action": "replay"}', method='POST',
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)
            data = json.loads(r.read())
            self.assertTrue(data['ok'])
            self.assertEqual(data['action'], 'replay')
        self.speech.say.assert_called_with('กล้วย')

        # test speak
        req = urllib.request.Request(self.base + '/api/action', data=b'{"action": "speak", "text": "\xe0\xb8\x8a\xe0\xb9\x89\xe0\xb8\xb2\xe0\xb8\x87"}', method='POST',
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)
            data = json.loads(r.read())
            self.assertTrue(data['ok'])
        self.speech.say.assert_called_with('ช้าง')

        # test toggle_mute
        self.speech.speaker = mock.MagicMock()
        req = urllib.request.Request(self.base + '/api/action', data=b'{"action": "toggle_mute"}', method='POST',
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)
            data = json.loads(r.read())
            self.assertTrue(data['muted'])
        self.assertTrue(self.speech.muted)
        self.speech.speaker.stop.assert_called_once()

        # test set_audio
        self.speech.set_audio_config = mock.MagicMock(return_value={'engine': 'piper', 'volume': 80})
        req = urllib.request.Request(self.base + '/api/action', data=b'{"action": "set_audio", "engine": "piper", "volume": 80}', method='POST',
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)
            data = json.loads(r.read())
            self.assertTrue(data['ok'])
            self.assertEqual(data['audio_config']['engine'], 'piper')
        self.speech.set_audio_config.assert_called_with(engine='piper', volume=80)

    def test_post_action_invalid_is_400(self):
        req = urllib.request.Request(self.base + '/api/action', data=b'{"action": "set_camera", "index": 1}', method='POST',
                                     headers={'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(caught.exception.code, 400)

    def test_unknown_voice_is_400(self):
        self.speech.set_audio_config = mock.MagicMock(side_effect=ValueError('unknown voice engine: auto'))
        req = urllib.request.Request(self.base + '/api/action', data=b'{"action": "set_audio", "engine": "auto"}',
                                     method='POST', headers={'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(caught.exception.code, 400)

    def test_unknown_path_404(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get('/../panel_settings.py')
        self.assertEqual(caught.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
