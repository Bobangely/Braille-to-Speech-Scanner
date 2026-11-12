"""Speech audio for the finger reader: synthesize each text once to a WAV cache, play it at once.

สังเคราะห์เสียงใน Python environment ของ Piper หรือ Edge ผ่าน subprocess
เล่นเสียงด้วย winsound บน Windows และ afplay/aplay บนระบบอื่น
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import wave
from pathlib import Path

from piper_tts import EDGE_VOICES
from thai_speech import EDGE_SPELLING, TTS_SPELLING, tts_spelling

ROOT = Path(__file__).resolve().parent
log = logging.getLogger(__name__)
VOICES = ROOT / 'voices'  # เสียงที่อัดเอง ชื่อไฟล์คือข้อความที่พูด เช่น กอไก่.wav
WORD_RATE = '-25%'  # คำตอบให้ฟังชัด ส่วนชื่อเซลล์คงเดิมเพื่อให้ทันนิ้วที่เลื่อนต่อ


class CachedWavSpeaker:
    """Synthesize each text once to a WAV file on disk; play cached files at once.

    สังเคราะห์เสียงใน Python environment ที่ ``python_executable`` ชี้ไว้
    """

    def __init__(self, model: Path, cache_dir: Path, python_executable: str | None = None,
                 backend: str = 'piper', synthesize=None, play=None, rate_of=lambda text: None,
                 voices_dir: Path | None = None):
        if backend != 'edge' and not Path(model).exists():  # Edge เป็นเสียงออนไลน์ จึงไม่มีไฟล์โมเดล
            raise FileNotFoundError(f'Thai speech model not found: {model}')
        self.model, self.backend = Path(model or ''), backend
        # คำสะกดใหม่ปรับตามหูแยกตามเสียงแต่ละตัว
        table = {'piper': TTS_SPELLING, 'edge': EDGE_SPELLING}.get(backend, {})
        self.spell = lambda text: tts_spelling(text, table)
        self.cache_dir = Path(cache_dir)
        self.voices_dir = Path(voices_dir or VOICES)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.python = str(Path(python_executable).absolute()) if python_executable else sys.executable
        self._synthesize = synthesize or self._synthesize_in_tts_env
        self._play = play or _winsound_play
        self.rate_of = rate_of  # ความเร็วของแต่ละข้อความ None คือค่าปกติของเสียงนั้น
        self.volume = 100

    def ready(self, text: str) -> bool:
        return self._path(text).is_file()

    def _path(self, text: str) -> Path:
        # เสียงคนอัดชัดกว่า TTS จึงใช้ก่อน ห้ามมี / หรือ \ เพื่อไม่ให้หลุดออกนอกโฟลเดอร์
        recorded = self.voices_dir / f'{text}.wav'
        if '/' not in text and '\\' not in text and recorded.is_file():
            return recorded
        # ตั้งชื่อตามสิ่งที่สังเคราะห์จริง คำสะกดใหม่จึงได้เสียงใหม่
        rate = self.rate_of(text)
        # Edge ไม่มีไฟล์โมเดล ใช้ชื่อเสียงแทน เปลี่ยนเสียงแล้วจึงไม่เล่นเสียงเก่าจาก cache
        voice = EDGE_VOICES['thai'] if self.backend == 'edge' else self.model
        tag = f'{self.backend}|{voice}|{self.spell(text)}' + (f'|{rate}' if rate else '')
        return self.cache_dir / f'{hashlib.sha1(tag.encode("utf-8")).hexdigest()}.wav'

    def prepare_many(self, texts, lang: str = 'thai') -> set[str]:
        """สังเคราะห์ข้อความที่ยังไม่มี cache ในชุดเดียว แล้วคืนข้อความที่พร้อมเล่น"""
        texts = [t for t in texts if t.strip()]  # Piper ไม่เขียนเสียงให้ข้อความว่าง
        missing = [(t, self._path(t)) for t in dict.fromkeys(texts) if not self._path(t).is_file()]
        if missing:
            try:
                self._synthesize(self.model, [(self.spell(t), path, self.rate_of(t)) for t, path in missing])
            except Exception as error:
                log.warning('speech synthesis failed: %s', error)
            for _, path in missing:
                if path.is_file():
                    try:
                        trim_leading_silence(path)
                    except (OSError, EOFError, wave.Error) as error:  # ไม่ใช่ PCM WAV: เล่นโดยไม่ตัดช่วงเงียบ
                        log.warning('cannot trim %s: %s', path.name, error)
        return {t for t in texts if self._path(t).is_file()}

    def speak(self, text: str, lang: str = 'thai') -> bool:
        path = self._path(text)
        if not path.is_file():  # ห้ามสังเคราะห์เสียงในลูปเฟรม
            log.warning('no prepared speech for %r', text)
            return False
        if self.volume > 0:
            self._play(at_volume(path, self.volume))
        return True

    def stop(self) -> None:
        self._play(None)

    def _synthesize_in_tts_env(self, model: Path, jobs: list[tuple[str, Path, str | None]]) -> None:
        # process เดียว โหลดโมเดลครั้งเดียว: synthesize_wav เก็บเสียงที่โหลดแล้วต่อโมเดล
        payload = json.dumps(dict(model=str(model), backend=self.backend,
                                  jobs=[[text, str(path), rate] for text, path, rate in jobs]))
        done = subprocess.run([self.python, '-X', 'utf8', '-c', BATCH_SCRIPT % str(ROOT)],
                              input=payload.encode('utf-8'), capture_output=True)
        if done.returncode:
            raise RuntimeError(done.stderr.decode('utf-8', 'replace')[-500:])
        if done.stderr.strip():  # ข้อความที่ล้มเหลวทีละข้อความ ส่วนที่เหลือเก็บ cache แล้ว
            log.warning('speech synthesis skipped: %s', done.stderr.decode('utf-8', 'replace')[-500:])


class HybridSpeaker:
    """ข้อความใน ``fixed`` (หรือทุกข้อความถ้าเป็น None) ใช้เสียงออนไลน์ที่ชัดและเก็บ cache
    ที่เหลือใช้ Piper ข้อความที่ไม่มีเสียงออนไลน์ (ไม่มีเน็ต หรือบริการล่ม) จะกลับไปใช้ Piper"""

    def __init__(self, online, piper, fixed=None, engine='edge'):
        self.online, self.piper = online, piper
        self.fixed = None if fixed is None else set(fixed)
        self.engine = engine  # 'edge': ใช้เสียงออนไลน์ก่อน ถ้าไม่มีเสียงใช้ Piper, 'piper': ใช้ Piper ก่อน
        self._online_ready: set[str] = set()

    @property
    def volume(self) -> int:
        return self.piper.volume

    @volume.setter
    def volume(self, value: int) -> None:
        self.online.volume = self.piper.volume = value

    def prepare_many(self, texts, lang: str = 'thai') -> set[str]:
        texts = list(texts)
        if self.engine == 'piper':  # สังเคราะห์ Piper เฉพาะตอนเลือกใช้เสียงนี้
            return self.piper.prepare_many(texts, lang) | {t for t in texts if t in self._online_ready}
        wanted = [t for t in texts if self.fixed is None or t in self.fixed]
        try:
            self._online_ready = self._online_ready | self.online.prepare_many(wanted, lang)
        except Exception as error:  # เสียงออนไลน์พังต้องให้ Piper พูดแทน ไม่ใช่เงียบ
            log.warning('online voice failed, using Piper: %s', error)
        rest = [t for t in texts if t not in self._online_ready]
        return {t for t in texts if t in self._online_ready} | self.piper.prepare_many(rest, lang)

    def _use_online(self, text: str) -> bool:
        # ไฟล์ออนไลน์ที่เขียนเสร็จแล้วใช้ได้ทันที แม้ทั้งชุดยังไม่เสร็จ
        return text in self._online_ready or (
            (self.fixed is None or text in self.fixed) and self.online.ready(text))

    def ready(self, text: str) -> bool:
        if self.engine == 'piper':
            return self.piper.ready(text) or text in self._online_ready
        return self._use_online(text) or self.piper.ready(text)

    def speak(self, text: str, lang: str = 'thai') -> bool:
        if self.engine == 'piper':
            return self.piper.speak(text, lang) or self.online.speak(text, lang)
        return (self.online if self._use_online(text) else self.piper).speak(text, lang)

    def stop(self) -> None:
        self.piper.stop()  # ใช้ตัวเล่นร่วมกัน หยุดตัวใดตัวหนึ่งก็หยุดเสียงแล้ว


def trim_leading_silence(path: Path, keep_s: float = 0.02, threshold: float = 0.05) -> None:
    """ตัดช่วงเงียบต้นเสียง (ราว 0.25 วินาที สูงสุด 1 วินาที) ให้เล่นแล้วมีเสียงทันที"""
    import numpy as np

    with wave.open(str(path)) as source:
        params, frames = source.getparams(), source.readframes(source.getnframes())
    if params.sampwidth != 2:
        return
    samples = np.frombuffer(frames, np.int16).reshape(-1, params.nchannels)
    level = np.abs(samples.astype(np.int32)).max(axis=1)
    loud = np.nonzero(level > threshold * max(1, level.max()))[0]
    start = max(0, int(loud[0]) - round(keep_s * params.framerate)) if len(loud) else 0
    if start:
        with wave.open(str(path), 'wb') as out:
            out.setparams(params)
            out.writeframes(samples[start:].tobytes())


# รันใน environment ของ Piper/MMS/Edge: โหลดโมเดลครั้งเดียว ข้อความเดียวที่พังไม่หยุดข้อความอื่น
# Edge รอเครือข่าย จึงส่งครั้งละ 8 ข้อความ (8 คำ: 1.1 วินาที แทนที่จะเป็นราว 7 วินาที)
BATCH_SCRIPT = '''import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, %r)
from piper_tts import synthesize_wav
job = json.load(sys.stdin)
def one(item):
    text, out, rate = item
    tmp = out + ".tmp.wav"
    for attempt in range(3):  # Edge บางครั้งตอบ No audio แบบสุ่ม ลองใหม่มักผ่าน
        try:
            synthesize_wav(job["model"], text, tmp, backend=job["backend"], rate=rate)
            os.replace(tmp, out)
            return
        except Exception as error:
            if attempt == 2:
                print(repr(text), error, file=sys.stderr)
            else:
                time.sleep(1 + attempt)
with ThreadPoolExecutor(8 if job["backend"] == "edge" else 1) as pool:
    list(pool.map(one, job["jobs"]))
'''


_player: subprocess.Popen | None = None


def _process_play(path: Path | None) -> None:
    """เล่นเสียงบน macOS/Linux: ใช้ process afplay/aplay เดียว และหยุดเมื่อเสียงถัดไปเริ่ม"""
    global _player
    if _player is not None and _player.poll() is None:
        _player.kill()
    _player = None
    if path is not None:
        command = 'afplay' if sys.platform == 'darwin' else 'aplay'
        _player = subprocess.Popen([command, '-q', str(path)] if command == 'aplay' else [command, str(path)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _winsound_play(path: Path | None) -> None:
    if sys.platform != 'win32':
        return _process_play(path)
    import winsound
    if path is None:
        winsound.PlaySound(None, winsound.SND_PURGE)
    else:
        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)


def at_volume(path: Path, volume: int) -> Path:
    """``path`` ที่ปรับเป็น ``volume`` เปอร์เซ็นต์ เก็บ cache ไว้ข้างกัน (winsound ปรับความดังไม่ได้)"""
    import numpy as np

    volume = round(max(0, min(100, int(volume))) / 5) * 5
    if volume == 100:
        return path
    scaled = path.with_name(f'{path.stem}_v{volume}.wav')
    if not scaled.is_file():
        with wave.open(str(path)) as source:
            params, frames = source.getparams(), source.readframes(source.getnframes())
        samples = np.frombuffer(frames, np.int16).astype(np.float32) * (volume / 100)
        partial = scaled.with_suffix('.tmp')  # ไฟล์ที่เขียนไม่จบต้องไม่ถูกนับเป็นไฟล์สมบูรณ์
        with wave.open(str(partial), 'wb') as out:
            out.setparams(params)
            out.writeframes(np.clip(samples, -32768, 32767).astype(np.int16).tobytes())
        os.replace(partial, scaled)
    return scaled
