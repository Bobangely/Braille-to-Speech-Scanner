"""Piper แบบ offline: อ่านไฟล์โมเดลในเครื่องและหยุดงานผ่าน process แยก"""
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


class PiperSpeech:
    def __init__(self, models, python_executable=None):
        self.models = {lang: Path(path).expanduser().resolve() for lang, path in models.items() if path}
        for path in self.models.values():
            if not path.is_file() or not Path(str(path) + '.json').is_file():
                raise FileNotFoundError(f'Piper requires local model and model.onnx.json: {path}')
        self.python = str(Path(python_executable or sys.executable).resolve())
        if not Path(self.python).is_file():
            raise FileNotFoundError(self.python)
        self._process = None
        self._workspace = None
        self._log = None
        self.error = ''

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
                logging.getLogger(__name__).error('Piper failed: %s', self.error)
            return 'Speech error'
        return 'Ready'

    def speak(self, text, lang):
        if not text or not text.strip():
            return False
        if lang not in self.models:
            raise ValueError(f'No local Piper model configured for {lang}')
        self.stop()
        self.error = ''
        self._workspace = tempfile.TemporaryDirectory(prefix='braille-piper-')
        self._log = tempfile.TemporaryFile()
        job = dict(model=str(self.models[lang]), text=text,
                   output=str(Path(self._workspace.name) / 'speech.wav'))
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

    def stop(self):
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
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._workspace is not None:
            self._workspace.cleanup()
            self._workspace = None


def synthesize_wav(model, text, output):
    """สังเคราะห์บน CPU จากโมเดลในเครื่อง พร้อมเวลาที่วัดได้จริง"""
    started = time.perf_counter()
    from piper import PiperVoice
    # shortcut: โหลดโมเดลใหม่ต่อการกดหนึ่งครั้ง วัด latency บนบอร์ดก่อนเพิ่ม worker ค้างไว้
    voice = PiperVoice.load(str(model), use_cuda=False)
    loaded = time.perf_counter()
    with wave.open(str(output), 'wb') as wav:
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
    metrics = synthesize_wav(job['model'], job['text'], output)
    print(json.dumps(metrics), flush=True)
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
