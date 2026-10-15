import asyncio
from array import array
import json
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch
import wave

from piper_tts import PiperSpeech


class SpeechLatencyTests(unittest.TestCase):
    def test_same_pending_read_does_not_cancel_and_restart_synthesis(self):
        speaker = PiperSpeech({}, backend='edge')
        self.addCleanup(speaker.stop)
        process = Mock()
        process.poll.return_value = None
        with patch('piper_tts.subprocess.Popen', return_value=process) as start:
            speaker.speak('เขียดเขียวเกาะขอน', 'thai')
            speaker.speak('เขียดเขียวเกาะขอน', 'thai')
        self.assertEqual(start.call_count, 1)
        process.terminate.assert_not_called()

    def test_repeat_after_stop_reuses_only_complete_matching_audio(self):
        speaker = PiperSpeech({}, backend='edge')
        self.addCleanup(speaker.stop)
        jobs = []
        def start(*args, **kwargs):
            jobs.append(json.load(kwargs['stdin']))
            process = Mock()
            process.poll.return_value = 0
            return process
        with patch('piper_tts.subprocess.Popen', side_effect=start):
            speaker.speak('ศักดิ์', 'thai')
            directory = Path(speaker._workspace.name)
            with wave.open(str(directory/'speech.wav'), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b'\0\0' * 100)
            (directory/'ready').touch()
            speaker.stop()
            self.assertFalse(directory.exists())
            speaker.speak('ศักดิ์', 'thai')
            self.assertTrue(jobs[-1]['cached'])
            self.assertGreater(Path(jobs[-1]['output']).stat().st_size, 44)
            speaker.speak('ภู่', 'thai')
            self.assertFalse(jobs[-1]['cached'])
            speaker.speak('ศักดิ์', 'thai')
            self.assertTrue(jobs[-1]['cached'])
            with patch('piper_tts.EDGE_RATE', '-10%'):
                speaker.speak('ศักดิ์', 'thai')
            self.assertFalse(jobs[-1]['cached'])
            with patch.dict('piper_tts.THAI_PAUSES_MS', {'space': 300}):
                speaker.speak('ศักดิ์', 'thai')
            self.assertFalse(jobs[-1]['cached'])
            self.assertEqual(jobs[-1]['pauses']['space'], 300)
            speaker.speak('Hello', 'english')
            self.assertFalse(jobs[-1]['cached'])

    def test_stopped_partial_audio_is_not_replayed_from_cache(self):
        speaker = PiperSpeech({}, backend='edge')
        self.addCleanup(speaker.stop)
        jobs = []
        def start(*args, **kwargs):
            jobs.append(json.load(kwargs['stdin']))
            process = Mock()
            process.poll.return_value = 0
            return process
        with patch('piper_tts.subprocess.Popen', side_effect=start):
            speaker.speak('ศักดิ์', 'thai')
            Path(jobs[-1]['output']).write_bytes(b'partial')
            speaker.stop()
            speaker.speak('ศักดิ์', 'thai')
        self.assertFalse(jobs[-1]['cached'])

    def test_stream_plays_before_download_finishes_and_preserves_pcm_order(self):
        from edge_stream import stream_edge_audio
        network_finished = []
        first_play = []
        requests = []
        devices = []
        class Communicate:
            def __init__(self, text, **kwargs):
                self.text = text
                requests.append(text)
            async def stream(self):
                yield {'type': 'audio', 'data': b'\x01\0\x02\0'}
                await asyncio.sleep(.1)
                yield {'type': 'audio', 'data': b'\x03\0\x04\0'}
                network_finished.append(True)
        def decode(source, *args, **kwargs):
            def samples():
                yield array('h')
                while data := source.read(4):
                    block = array('h')
                    block.frombytes(data)
                    yield block
            generator = samples()
            next(generator)
            return generator
        class Device:
            def __init__(self, **kwargs):
                devices.append(kwargs)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def start(self, generator):
                try:
                    while True:
                        block = generator.send(2)
                        if block and not first_play:
                            first_play.append(not network_finished)
                except StopIteration:
                    pass
        audio = types.SimpleNamespace(StreamableSource=object, stream_any=decode,
            FileFormat=types.SimpleNamespace(MP3='mp3'), PlaybackDevice=Device)
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', {
                'miniaudio': audio, 'edge_tts': types.SimpleNamespace(Communicate=Communicate)}):
            output = Path(directory)/'speech.wav'
            metrics = stream_edge_audio('ฃ เสื้อ  ภู่\nกิน. คน', 'th-TH-PremwadeeNeural', '-15%', output)
            with wave.open(str(output), 'rb') as wav:
                block = b'\x01\0\x02\0\x03\0\x04\0'
                self.assertEqual(wav.readframes(wav.getnframes()),
                                 block + b'\0\0'*3600 + block + b'\0\0'*7200 + block
                                 + b'\0\0'*10800 + block + b'\0\0'*13200 + block)
            self.assertEqual(requests, ['ขอขวด', 'เสื้อ', 'ภู่', 'กิน.', 'คน'])
            self.assertEqual(metrics['inserted_silence_seconds'], 1.45)
        self.assertEqual(len(devices), 1)
        self.assertEqual(first_play, [True])
        self.assertLess(metrics['first_audio_seconds'], metrics['network_seconds'])

    def test_decoder_failure_does_not_deadlock_a_blocked_downloader(self):
        from contextlib import nullcontext
        from edge_stream import stream_edge_audio
        class Communicate:
            def __init__(self, *args, **kwargs):
                pass
            async def stream(self):
                for _ in range(20):
                    yield {'type': 'audio', 'data': b'\0' * 65536}
        def decode(source, *args, **kwargs):
            def samples():
                yield b''
                source.read(4)
                raise RuntimeError('decoder failed')
            generator = samples()
            next(generator)
            return generator
        def play(generator):
            try:
                generator.send(2)
            except StopIteration:
                pass
        audio = types.SimpleNamespace(StreamableSource=object, stream_any=decode,
            FileFormat=types.SimpleNamespace(MP3='mp3'),
            PlaybackDevice=lambda **kwargs: nullcontext(types.SimpleNamespace(start=play)))
        finished, failures = threading.Event(), []
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', {
                'miniaudio': audio, 'edge_tts': types.SimpleNamespace(Communicate=Communicate)}):
            def run():
                try:
                    stream_edge_audio('เสื้อ', 'th-TH-PremwadeeNeural', '-15%', Path(directory)/'speech.wav')
                except Exception as exc:
                    failures.append(str(exc))
                finally:
                    finished.set()
            threading.Thread(target=run, daemon=True).start()
            self.assertTrue(finished.wait(2), 'decoder failure left the downloader blocked')
        self.assertEqual(failures, ['decoder failed'])
