"""สังเคราะห์ข้อความเป็นไฟล์ WAV ด้วย Piper, MMS หรือ Edge (Edge ต้องต่ออินเทอร์เน็ต)"""
from pathlib import Path
import time
import wave

from thai_speech import expand_thai_speech_units

EDGE_VOICES = {'thai': 'th-TH-NiwatNeural', 'english': 'en-US-JennyNeural'}
EDGE_RATE = '-20%'


_LOADED_VOICES = {}  # (ชนิดเสียง, path โมเดล) -> เสียงที่โหลดแล้ว ต่อหนึ่ง process


def synthesize_wav(model, text, output, backend='piper', voice=None, rate=None):
    """สร้าง PCM WAV และคืนเวลาที่ใช้ เสียง Edge ต้องต่ออินเทอร์เน็ต"""
    started = time.perf_counter()
    if backend == 'edge':
        import asyncio
        import edge_tts
        import soundfile
    elif backend == 'mms':
        import torch
        from transformers import AutoTokenizer, VitsModel
        key = (id(VitsModel), str(model))
        if key not in _LOADED_VOICES:
            _LOADED_VOICES[key] = (AutoTokenizer.from_pretrained(str(model), local_files_only=True),
                                   VitsModel.from_pretrained(str(model), local_files_only=True).eval())
        tokenizer, voice = _LOADED_VOICES[key]
    elif backend == 'piper':
        from piper import PiperVoice
        # shortcut: โหลดโมเดลใหม่ต่อการกดหนึ่งครั้ง วัด latency บนบอร์ดก่อนเพิ่ม worker ค้างไว้
        # ภายใน process เดียวกันโหลดครั้งเดียว: ตัวอ่านตามนิ้วสังเคราะห์ทั้งหน้าใน process เดียว
        key = (id(PiperVoice), str(model))
        if key not in _LOADED_VOICES:
            _LOADED_VOICES[key] = PiperVoice.load(str(model), use_cuda=False)
        voice = _LOADED_VOICES[key]
    else:
        raise ValueError(f'Unsupported offline backend: {backend}')
    loaded = time.perf_counter()
    if backend == 'edge':
        mp3 = Path(output).with_suffix('.mp3')
        try:
            edge_voice = voice or EDGE_VOICES['thai']
            spoken = expand_thai_speech_units(text) if edge_voice.startswith('th-TH-') else text
            asyncio.run(edge_tts.Communicate(spoken, voice=edge_voice, rate=rate or EDGE_RATE).save(str(mp3)))
            samples, sample_rate = soundfile.read(mp3, dtype='int16')
            soundfile.write(output, samples, sample_rate, subtype='PCM_16')
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
