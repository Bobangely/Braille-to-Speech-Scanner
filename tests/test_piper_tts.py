"""ทดสอบ Piper โดยไม่ต้องติดตั้งโมเดลหรือเรียกเครือข่าย"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


class PiperSpeechTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name) / 'voice.onnx'
        self.model.write_bytes(b'test fixture')
        Path(str(self.model) + '.json').write_text('{}')

    def controller(self):
        from piper_tts import PiperSpeech
        speaker = PiperSpeech({'thai': self.model, 'english': self.model})
        self.addCleanup(speaker.stop)
        return speaker

    def test_missing_model_or_config_fails_without_download(self):
        from piper_tts import PiperSpeech
        with self.assertRaises(FileNotFoundError):
            PiperSpeech({'thai': self.model.with_name('missing.onnx')})
        Path(str(self.model) + '.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.controller()

    def test_blank_or_unconfigured_language_never_starts_worker(self):
        from piper_tts import PiperSpeech
        speaker = PiperSpeech({'thai': self.model})
        self.addCleanup(speaker.stop)
        with patch('piper_tts.subprocess.Popen') as start:
            self.assertFalse(speaker.speak('  ', 'thai'))
            with self.assertRaises(ValueError):
                speaker.speak('Hello', 'english')
            start.assert_not_called()

    def test_worker_receives_exact_text_and_local_model_and_repeat_stops_previous(self):
        speaker = self.controller()
        processes = []
        jobs = []

        def launch(*args, **kwargs):
            jobs.append(json.load(kwargs['stdin']))
            proc = Mock()
            proc.poll.return_value = None
            processes.append(proc)
            return proc

        with patch('piper_tts.subprocess.Popen', side_effect=launch), \
                patch('piper_tts.os.killpg', create=True) as kill_group:
            for lang, text in [('thai', 'สวัสดีครับ'), ('english', 'Hello; "world" & exit')]:
                self.assertTrue(speaker.speak(text, lang))
                self.assertEqual(speaker.status, 'Speaking')
            if os.name == 'nt':
                processes[0].terminate.assert_called_once()
            else:
                self.assertEqual(kill_group.call_count, 1)
            self.assertEqual(jobs[1]['text'], 'Hello; "world" & exit')
            self.assertEqual(jobs[0]['model'], str(self.model.resolve()))
            speaker.stop()
            if os.name == 'nt':
                processes[1].terminate.assert_called_once()
            else:
                self.assertEqual(kill_group.call_count, 2)
            self.assertEqual(speaker.status, 'Ready')

    def test_stop_terminates_real_worker_and_removes_temporary_audio(self):
        speaker = self.controller()
        real_popen = subprocess.Popen

        def launch(*args, **kwargs):
            return real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)

        with patch('piper_tts.subprocess.Popen', side_effect=launch):
            speaker.speak('Hello', 'english')
        proc = speaker._process
        directory = Path(speaker._workspace.name)
        self.assertIsNone(proc.poll())
        speaker.stop()
        self.assertIsNotNone(proc.poll())
        self.assertFalse(directory.exists())

    def test_launch_failure_leaves_no_worker_or_temporary_directory(self):
        speaker = self.controller()
        with patch('piper_tts.subprocess.Popen', side_effect=OSError('not executable')):
            with self.assertRaises(OSError):
                speaker.speak('Hello', 'english')
        self.assertIsNone(speaker._process)
        self.assertIsNone(speaker._workspace)

    def test_worker_failure_is_reported_not_sent_to_cloud(self):
        speaker = self.controller()

        def fail(*args, **kwargs):
            kwargs['stderr'].write(b'Missing Thai dependency')
            kwargs['stderr'].flush()
            proc = Mock()
            proc.poll.return_value = 1
            return proc

        with patch('piper_tts.subprocess.Popen', side_effect=fail):
            speaker.speak('สวัสดี', 'thai')
            self.assertEqual(speaker.status, 'Speech error')
            self.assertIn('Missing Thai dependency', speaker.error)

    def test_synthesis_uses_only_local_model_and_cpu(self):
        from piper_tts import synthesize_wav
        fake_voice = Mock()
        fake_voice.synthesize_wav.side_effect = lambda text, wav: (
            wav.setnchannels(1), wav.setsampwidth(2), wav.setframerate(22050),
            wav.writeframes(b'\0\0' * 2205))
        module = Mock()
        module.PiperVoice.load.return_value = fake_voice
        output = Path(self.temp.name) / 'speech.wav'
        with patch.dict('sys.modules', {'piper': module}), patch('socket.create_connection', side_effect=AssertionError('network')):
            metrics = synthesize_wav(self.model, 'Hello', output)
        module.PiperVoice.load.assert_called_once_with(str(self.model), use_cuda=False)
        self.assertEqual(metrics['audio_seconds'], .1)
        self.assertGreater(output.stat().st_size, 44)


class ScannerSpeechTests(unittest.TestCase):
    def scanner(self):
        from camera_reader import RealTimeBrailleScanner
        return RealTimeBrailleScanner(res_preset='480p')

    def test_only_confirmed_text_is_spoken_and_repeat_and_stop_do_not_reset_grid(self):
        scanner = self.scanner()
        scanner._speech = Mock()
        scanner._confirmed_text = 'กา'
        scanner.history.extend(['กา'] * 6)
        context = scanner._context_generation
        scanner._handle_key(ord('s'), None, None)
        scanner._handle_key(ord('s'), None, None)
        self.assertEqual(scanner._speech.speak.call_count, 2)
        scanner._speech.speak.assert_called_with('กา', 'thai')
        scanner._handle_key(ord('b'), None, None)
        scanner._speech.stop.assert_called_once()
        self.assertEqual(list(scanner.history), ['กา'] * 6)
        self.assertEqual(scanner._context_generation, context)
        scanner._confirmed_text = ''
        scanner._speech.reset_mock()
        scanner._handle_key(ord('s'), None, None)
        scanner._speech.speak.assert_not_called()

    def test_quit_and_language_switch_stop_speech(self):
        for key in ('q', 'l'):
            scanner = self.scanner()
            scanner._speech = Mock()
            scanner._handle_key(ord(key), None, None)
            scanner._speech.stop.assert_called_once()

    def test_audio_controls_fit_and_dispatch_mouse_clicks(self):
        import cv2
        scanner = self.scanner()
        scanner._speech = Mock(status='Ready')
        for size in ((640, 480), (1440, 900), (900, 1200)):
            scanner._ui.resize(*size)
            scanner._compose_dashboard(None, {})
            left, top, right, bottom = scanner._ui.control_card
            for box, key in scanner._ui.buttons:
                if key in (ord('['), ord(']')):
                    continue
                self.assertTrue(left <= box[0] < box[2] <= right)
                self.assertTrue(top <= box[1] < box[3] <= bottom)
                self.assertFalse(scanner._ui._inside(scanner._ui.preview_rect, box[0], box[1]))
            for key in ('s', 'b'):
                box, _ = next(item for item in scanner._ui.buttons if item[1] == ord(key))
                scanner._on_mouse(cv2.EVENT_LBUTTONDOWN, (box[0]+box[2])/2, (box[1]+box[3])/2, 0, None)
                self.assertEqual(scanner._ui_pending_key, ord(key))


if __name__ == '__main__':
    unittest.main()
