"""Speech audio for the finger reader: synthesize each text once to a WAV cache, play it at once.

Synthesis runs in the Piper or Edge Python environment through a subprocess;
playback uses winsound, the one Windows-only part to replace on a Linux board.
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

from thai_speech import EDGE_SPELLING, TTS_SPELLING, tts_spelling

ROOT = Path(__file__).resolve().parent
log = logging.getLogger(__name__)


class CachedWavSpeaker:
    """Synthesize each text once to a WAV file on disk; play cached files at once.

    Synthesis runs in the Piper/MMS Python environment (``python_executable``),
    the same way ``piper_tts`` does; playback uses winsound.
    """

    def __init__(self, model: Path, cache_dir: Path, python_executable: str | None = None,
                 backend: str = 'piper', synthesize=None, play=None):
        if backend != 'edge' and not Path(model).exists():  # Edge is an online voice: no model file
            raise FileNotFoundError(f'Thai speech model not found: {model}')
        self.model, self.backend = Path(model or ''), backend
        # Respellings are tuned per voice by ear.
        table = {'piper': TTS_SPELLING, 'edge': EDGE_SPELLING}.get(backend, {})
        self.spell = lambda text: tts_spelling(text, table)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.python = str(Path(python_executable).resolve()) if python_executable else sys.executable
        self._synthesize = synthesize or self._synthesize_in_tts_env
        self._play = play or _winsound_play
        self.volume = 100

    def ready(self, text: str) -> bool:
        return self._path(text).is_file()

    def _path(self, text: str) -> Path:
        # Keyed by what is synthesized, so a new respelling makes new audio.
        key = hashlib.sha1(f'{self.backend}|{self.model}|{self.spell(text)}'.encode('utf-8')).hexdigest()
        return self.cache_dir / f'{key}.wav'

    def prepare_many(self, texts, lang: str = 'thai') -> set[str]:
        """Synthesize the uncached texts in one batch; return the texts ready to play."""
        texts = [t for t in texts if t.strip()]  # Piper writes no audio for blank text
        missing = [(t, self._path(t)) for t in dict.fromkeys(texts) if not self._path(t).is_file()]
        if missing:
            try:
                self._synthesize(self.model, [(self.spell(t), path) for t, path in missing])
            except Exception as error:
                log.warning('speech synthesis failed: %s', error)
            for _, path in missing:
                if path.is_file():
                    try:
                        trim_leading_silence(path)
                    except (OSError, EOFError, wave.Error) as error:  # not a PCM WAV: play it untrimmed
                        log.warning('cannot trim %s: %s', path.name, error)
        return {t for t in texts if self._path(t).is_file()}

    def speak(self, text: str, lang: str = 'thai') -> bool:
        path = self._path(text)
        if not path.is_file():  # never synthesize inside the frame loop
            log.warning('no prepared speech for %r', text)
            return False
        if self.volume > 0:
            self._play(at_volume(path, self.volume))
        return True

    def stop(self) -> None:
        self._play(None)

    def _synthesize_in_tts_env(self, model: Path, jobs: list[tuple[str, Path]]) -> None:
        # One process, one model load: synthesize_wav caches the loaded voice per model.
        payload = json.dumps(dict(model=str(model), backend=self.backend,
                                  jobs=[[text, str(path)] for text, path in jobs]))
        done = subprocess.run([self.python, '-X', 'utf8', '-c', BATCH_SCRIPT % str(ROOT)],
                              input=payload.encode('utf-8'), capture_output=True)
        if done.returncode:
            raise RuntimeError(done.stderr.decode('utf-8', 'replace')[-500:])
        if done.stderr.strip():  # texts that failed one by one; the rest are cached
            log.warning('speech synthesis skipped: %s', done.stderr.decode('utf-8', 'replace')[-500:])


class HybridSpeaker:
    """Texts in ``fixed`` (or every text when it is None) in a clear online voice, cached;
    the rest in Piper. A text without online audio (offline, service down) falls back to Piper."""

    def __init__(self, online, piper, fixed=None, engine='edge'):
        self.online, self.piper = online, piper
        self.fixed = None if fixed is None else set(fixed)
        self.engine = engine  # 'edge': online first, Piper when it has no audio; 'piper': Piper first
        self._online_ready: set[str] = set()

    @property
    def volume(self) -> int:
        return self.piper.volume

    @volume.setter
    def volume(self, value: int) -> None:
        self.online.volume = self.piper.volume = value

    def prepare_many(self, texts, lang: str = 'thai') -> set[str]:
        texts = list(texts)
        if self.engine == 'piper':  # synthesize Piper only when it is the chosen voice
            return self.piper.prepare_many(texts, lang) | {t for t in texts if t in self._online_ready}
        wanted = [t for t in texts if self.fixed is None or t in self.fixed]
        try:
            self._online_ready = self._online_ready | self.online.prepare_many(wanted, lang)
        except Exception as error:  # a broken online voice must leave Piper to speak, not silence
            log.warning('online voice failed, using Piper: %s', error)
        rest = [t for t in texts if t not in self._online_ready]
        return {t for t in texts if t in self._online_ready} | self.piper.prepare_many(rest, lang)

    def speak(self, text: str, lang: str = 'thai') -> bool:
        if self.engine == 'piper':
            return self.piper.speak(text, lang) or self.online.speak(text, lang)
        return (self.online if text in self._online_ready else self.piper).speak(text, lang)

    def stop(self) -> None:
        self.piper.stop()  # one shared player: stopping either stops playback


def trim_leading_silence(path: Path, keep_s: float = 0.02, threshold: float = 0.05) -> None:
    """Cut the voice's silent lead-in (~0.25 s, up to 1 s) so playback starts with the sound."""
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


# Runs inside the Piper/MMS/Edge environment: one model load, one failed text never stops the rest.
# Edge waits on the network, so its texts go 8 at a time (8 words: 1.1 s instead of ~7 s).
BATCH_SCRIPT = '''import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, %r)
from piper_tts import synthesize_wav
job = json.load(sys.stdin)
def one(item):
    text, out = item
    tmp = out + ".tmp.wav"
    try:
        synthesize_wav(job["model"], text, tmp, backend=job["backend"])
        os.replace(tmp, out)
    except Exception as error:
        print(repr(text), error, file=sys.stderr)
with ThreadPoolExecutor(8 if job["backend"] == "edge" else 1) as pool:
    list(pool.map(one, job["jobs"]))
'''


def _winsound_play(path: Path | None) -> None:
    # shortcut: Windows-only playback; add an aplay process for Linux boards when needed.
    import winsound
    if path is None:
        winsound.PlaySound(None, winsound.SND_PURGE)
    else:
        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)


def at_volume(path: Path, volume: int) -> Path:
    """``path`` scaled to ``volume`` percent, cached beside it (winsound has no volume control)."""
    import numpy as np

    volume = round(max(0, min(100, int(volume))) / 5) * 5
    if volume == 100:
        return path
    scaled = path.with_name(f'{path.stem}_v{volume}.wav')
    if not scaled.is_file():
        with wave.open(str(path)) as source:
            params, frames = source.getparams(), source.readframes(source.getnframes())
        samples = np.frombuffer(frames, np.int16).astype(np.float32) * (volume / 100)
        partial = scaled.with_suffix('.tmp')  # a write cut short must never pass for a finished file
        with wave.open(str(partial), 'wb') as out:
            out.setparams(params)
            out.writeframes(np.clip(samples, -32768, 32767).astype(np.int16).tobytes())
        os.replace(partial, scaled)
    return scaled
