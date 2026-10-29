"""Teacher-panel web server: the routes and JSON of docs/web-api.md, on a real socket."""
import json
import tempfile
import threading
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
        self.assertIn(b'BRAILLE STATION', body)

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

        # test mute / toggle_mute
        req = urllib.request.Request(self.base + '/api/action', data=b'{"action": "toggle_mute"}', method='POST',
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)
            data = json.loads(r.read())
            self.assertTrue(data['muted'])
        self.assertTrue(self.speech.muted)

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
