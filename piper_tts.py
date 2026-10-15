"""Stoppable speech worker: local Piper/MMS or explicitly selected online Edge."""
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import wave
from edge_stream import THAI_PAUSES_MS

EDGE_VOICES = {'thai': 'th-TH-NiwatNeural', 'english': 'en-US-JennyNeural'}
EDGE_RATE = '+0%'


class PiperSpeech:
    def __init__(self, models, python_executable=None, thai_backend='piper', backend='local'):
        if backend not in ('local', 'edge'):
            raise ValueError(f'Unsupported speech backend: {backend}')
        self.backend = backend
        if thai_backend not in ('piper', 'mms'):
            raise ValueError(f'Unsupported offline Thai backend: {thai_backend}')
        self.thai_backend = thai_backend
        self.models = {lang: Path(path).expanduser().resolve() for lang, path in models.items() if path}
        for lang, path in (self.models.items() if backend == 'local' else ()):
            if lang == 'thai' and thai_backend == 'mms':
                required = ('config.json', 'model.safetensors', 'tokenizer_config.json', 'vocab.json')
                if not all((path / name).is_file() for name in required):
                    raise FileNotFoundError(f'MMS requires local model/tokenizer files: {path}')
                continue
            if not path.is_file() or not Path(str(path) + '.json').is_file():
                raise FileNotFoundError(f'Piper requires local model and model.onnx.json: {path}')
        self.python = str(Path(python_executable or sys.executable).resolve())
        if not Path(self.python).is_file():
            raise FileNotFoundError(self.python)
        self._process = None
        self._workspace = None
        self._log = None
        self.error = ''
        self._active_key = self._cache_key = None
        self._cache_audio = None
        self._pending_cleanup = []

    @property
    def status(self):
        if self._process is None:
            return 'Ready'
        code = self._process.poll()
        if code is None:
            return 'Speaking'
        if code:
            if not self.error:
                self._log.seek(0)
                self.error = self._log.read().decode('utf-8', errors='replace')[-2000:] or f'Piper exit code {code}'
                logging.getLogger(__name__).error('Speech failed: %s', self.error)
            return 'Speech error'
        return 'Ready'

    def speak(self, text, lang):
        if not text or not text.strip():
            return False
        if lang not in (EDGE_VOICES if self.backend == 'edge' else self.models):
            raise ValueError(f'No speech voice configured for {lang}')
        key = (text, lang, self.backend, self.thai_backend, str(self.models.get(lang, '')),
               EDGE_VOICES.get(lang), EDGE_RATE,
               tuple(sorted(THAI_PAUSES_MS.items())) if self.backend == 'edge' and lang == 'thai' else ())
        self._remember_audio()
        if (key == self._active_key and self._process is not None
                and self._process.poll() is None and self._cache_key != key):
            return True
        self.stop()
        self._active_key = key
        self.error = ''
        self._workspace = tempfile.TemporaryDirectory(prefix='braille-piper-', ignore_cleanup_errors=True)
        self._log = tempfile.TemporaryFile()
        job = dict(model='' if self.backend == 'edge' else str(self.models[lang]), text=text,
                   backend='edge' if self.backend == 'edge' else self.thai_backend if lang == 'thai' else 'piper',
                   output=str(Path(self._workspace.name) / 'speech.wav'))
        job['cached'] = key == self._cache_key
        if job['cached']:
            Path(job['output']).write_bytes(self._cache_audio)
        if self.backend == 'edge':
            job['voice'] = EDGE_VOICES[lang]
            job['pauses'] = dict(THAI_PAUSES_MS)
            job['rate'] = EDGE_RATE
        options = ({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt'
                   else {'start_new_session': True})
        try:
            # ส่งข้อความผ่าน stdin เพื่อไม่ให้ข้อความกลายเป็นคำสั่ง shell
            with tempfile.TemporaryFile() as source:
                source.write(json.dumps(job, ensure_ascii=False).encode('utf-8'))
                source.seek(0)
                self._process = subprocess.Popen(
                    [self.python, '-X', 'utf8', str(Path(__file__).resolve())],
                    stdin=source, stdout=self._log, stderr=self._log, **options)
        except Exception:
            self.stop()
            raise
        return True

    def _remember_audio(self):
        if self._workspace is not None and self._active_key != self._cache_key:
            directory = Path(self._workspace.name)
            if (directory/'ready').is_file():
                self._cache_audio = (directory/'speech.wav').read_bytes()
                self._cache_key = self._active_key

    def stop(self):
        self._remember_audio()
        proc = self._process
        if proc is not None and proc.poll() is None:
            try:
                if os.name == 'nt':
                    proc.terminate()
                else:
                    # หยุดทั้ง worker และ aplay ที่อยู่ในกลุ่มเดียวกัน
                    os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=.2)
                except subprocess.TimeoutExpired:
                    if os.name == 'nt':
                        proc.kill()
                    else:
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=1)
            except ProcessLookupError:
                pass
        self._process = None
        self._active_key = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._workspace is not None:
            self._pending_cleanup.append(self._workspace)
            self._workspace = None
        pending, self._pending_cleanup = self._pending_cleanup, []
        for workspace in pending:
            try:
                workspace.cleanup()
            except OSError:
                logging.getLogger(__name__).warning('Speech file cleanup deferred', exc_info=True)
            if Path(workspace.name).exists():
                self._pending_cleanup.append(workspace)
                logging.getLogger(__name__).warning('Speech files still in use; will retry cleanup: %s', workspace.name)


def synthesize_wav(model, text, output, backend='piper', voice=None):
    """Return PCM WAV and timings; Edge requires an internet connection."""
    started = time.perf_counter()
    if backend == 'edge':
        import asyncio
        import edge_tts
        import soundfile
        from edge_stream import prepare_edge_text
    elif backend == 'mms':
        import torch
        from transformers import AutoTokenizer, VitsModel
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        voice = VitsModel.from_pretrained(str(model), local_files_only=True).eval()
    elif backend == 'piper':
        from piper import PiperVoice
        # shortcut: โหลดโมเดลใหม่ต่อการกดหนึ่งครั้ง วัด latency บนบอร์ดก่อนเพิ่ม worker ค้างไว้
        voice = PiperVoice.load(str(model), use_cuda=False)
    else:
        raise ValueError(f'Unsupported offline backend: {backend}')
    loaded = time.perf_counter()
    if backend == 'edge':
        mp3 = Path(output).with_suffix('.mp3')
        try:
            edge_voice = voice or EDGE_VOICES['thai']
            asyncio.run(edge_tts.Communicate(prepare_edge_text(text, edge_voice), voice=edge_voice, rate=EDGE_RATE).save(str(mp3)))
            samples, rate = soundfile.read(mp3, dtype='int16')
            soundfile.write(output, samples, rate, subtype='PCM_16')
        finally:
            mp3.unlink(missing_ok=True)
    else:
        with wave.open(str(output), 'wb') as wav:
            if backend == 'mms':
                import numpy as np
                inputs = tokenizer(text, return_tensors='pt')
                with torch.inference_mode():
                    samples = voice(**inputs).waveform[0].cpu().numpy()
                samples = (np.clip(samples, -1, 1) * 32767).astype('<i2')
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(voice.config.sampling_rate)
                wav.writeframes(samples.tobytes())
            else:
                voice.synthesize_wav(text, wav)
    finished = time.perf_counter()
    with wave.open(str(output), 'rb') as wav:
        duration = wav.getnframes() / wav.getframerate()
    if duration <= 0:
        raise ValueError('Piper produced no audio')
    return dict(load_seconds=loaded-started, synthesis_seconds=finished-loaded,
                ready_seconds=finished-started, audio_seconds=duration)


def _worker():
    job = json.load(sys.stdin)
    output = job['output']
    backend = job.get('backend', 'piper')
    if job.get('cached'):
        metrics = dict(cached=True)
    elif backend == 'edge':
        from edge_stream import stream_edge_audio
        metrics = stream_edge_audio(job['text'], job['voice'], job.get('rate', EDGE_RATE), output,
                                    pauses=job.get('pauses'))
    else:
        metrics = synthesize_wav(job['model'], job['text'], output, backend=backend, voice=job.get('voice'))
    Path(output).with_name('ready').touch()
    print(json.dumps(metrics), flush=True)
    if backend == 'edge' and not job.get('cached'):
        return
    if sys.platform == 'win32':
        import winsound
        winsound.PlaySound(output, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
    elif sys.platform.startswith('linux'):
        player = shutil.which('aplay')
        if not player:
            raise RuntimeError('Offline playback requires ALSA aplay (alsa-utils)')
        subprocess.run([player, '-q', output], check=True)
    else:
        raise RuntimeError(f'Piper audio playback not configured for {sys.platform}')


if __name__ == '__main__':
    _worker()
