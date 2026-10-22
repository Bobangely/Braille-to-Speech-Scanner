"""Explicit model download and isolated CPU/offline speech trials; no playback."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[2]
TEXTS = [
    'สวัสดีครับ กรุณาวางกระดาษให้นิ่ง แล้วกดปุ่มอ่านอีกครั้ง',
    'ป่า ป้า ป๊า ป๋า ข้าว ข่าว เข้า เขา เสือ เสื้อ เสื่อ',
    'วันนี้มีนักเรียนสิบสองคน อ่านหนังสือที่ห้องสมุด เวลาเก้าโมงสามสิบนาที',
]
SOURCES = [
    ('VIZINTZOR/VachanaTTS', '6fa5f03a26f8ec0c0e0582e9e6812efa84d38688', 'vachana', [
        ('speaker_config.json', None),
        ('voices/th_f_1.onnx', 'e87f6d6f825680118cc412f6f9da944bdece5ebe772769a14b2d8359a07c5bf1'),
        ('voices/th_m_1.onnx', 'c63bedc664d8a2c525181360f4dfa624d2940b3636de6caff141353f72d984bd'),
    ]),
    ('facebook/mms-tts-tha', 'fcecc37a91566f4a36ba6c6c8aa39830ff6daa9d', 'mms', [
        ('config.json', None), ('special_tokens_map.json', None),
        ('tokenizer_config.json', None), ('vocab.json', None),
        ('model.safetensors', 'c00a7f642fa9e786ce0fa755393b0859ad079bbf2327a85ecbadee9286166ab9'),
    ]),
]


def download_models(destination):
    """Only this explicit command uses the network; record immutable revisions."""
    records = []
    for repo, revision, directory, files in SOURCES:
        for filename, expected in files:
            path = destination / directory / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'
            temporary = path.with_suffix(path.suffix + '.part')
            try:
                urllib.request.urlretrieve(url, temporary)
                digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
                if expected and digest != expected:
                    raise ValueError(f'SHA-256 mismatch: {filename}')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            records.append(dict(file=str(path.relative_to(destination)), url=url,
                                sha256=digest, bytes=path.stat().st_size))
    (destination / 'provenance.json').write_text(json.dumps(records, indent=2))


def run_worker(args):
    started, cpu_started = time.perf_counter(), time.process_time()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')

    def deny_network(*unused, **kwargs):
        raise RuntimeError('Network forbidden during offline synthesis')

    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny_network
    # Import cost is included in load_seconds; each engine starts a fresh process.
    if args.engine == 'piper':
        from piper import PiperVoice
        voice = PiperVoice.load(str(ROOT / 'scratch/piper-voices/th_TH-tsync2-medium.onnx'), use_cuda=False)

        def synthesize(text, output):
            with wave.open(str(output), 'wb') as wav:
                voice.synthesize_wav(text, wav)
    elif args.engine.startswith('vachana'):
        from vachanatts.voice import Voice
        name = 'th_f_1' if args.engine == 'vachana_f' else 'th_m_1'
        voice = Voice.load(args.models / f'vachana/voices/{name}.onnx',
                           args.models / 'vachana/speaker_config.json', use_cuda=False)

        def synthesize(text, output):
            with wave.open(str(output), 'wb') as wav:
                voice.synthesize_wav(text, wav)
    else:
        import torch
        from transformers import AutoTokenizer, VitsModel, set_seed
        from scipy.io.wavfile import write
        set_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(args.models / 'mms', local_files_only=True)
        voice = VitsModel.from_pretrained(args.models / 'mms', local_files_only=True).eval()

        def synthesize(text, output):
            inputs = tokenizer(text, return_tensors='pt')
            with torch.inference_mode():
                samples = voice(**inputs).waveform[0].cpu().numpy()
            # PCM16 makes sample files playable through the same UI player as Piper.
            import numpy as np
            write(output, voice.config.sampling_rate, (np.clip(samples, -1, 1) * 32767).astype(np.int16))

    loaded = time.perf_counter()
    results = []
    for index, text in enumerate(TEXTS):
        output = args.output / f'{args.engine}_{index + 1}.wav'
        before, cpu_before = time.perf_counter(), time.process_time()
        synthesize(text, output)
        finished = time.perf_counter()
        with wave.open(str(output), 'rb') as wav:
            duration = wav.getnframes() / wav.getframerate()
            assert duration > 0 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
            sample_rate = wav.getframerate()
        results.append(dict(text=text, file=output.name, synthesis_seconds=finished-before,
                            ready_seconds=finished-started, cpu_seconds=time.process_time()-cpu_before,
                            audio_seconds=duration, sample_rate=sample_rate,
                            realtime_factor=(finished-before)/duration))
    (args.output / f'{args.engine}.json').write_text(json.dumps(dict(
        engine=args.engine, load_seconds=loaded-started,
        total_cpu_seconds=time.process_time()-cpu_started, samples=results), ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=ROOT / 'scratch/thai-tts-models')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/thai_tts')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--engine', choices=['all', 'piper', 'vachana_f', 'vachana_m', 'mms'], default='all')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)
    if args.download:
        download_models(args.models)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if args.engine == 'all':
            parser.error('worker requires one engine')
        run_worker(args)
        return
    import psutil
    engines = ['piper', 'vachana_f', 'vachana_m', 'mms'] if args.engine == 'all' else [args.engine]
    for engine in engines:
        started, peak = time.perf_counter(), 0
        with (args.output / f'{engine}.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-X', 'utf8', __file__, '--worker', '--engine', engine,
                                        '--models', str(args.models), '--output', str(args.output)],
                                       stdout=log, stderr=subprocess.STDOUT)
            monitor = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    processes = [monitor] + monitor.children(recursive=True)
                    peak = max(peak, sum(item.memory_info().rss for item in processes))
                except psutil.NoSuchProcess:
                    break
                if time.perf_counter() - started > 300:
                    for item in reversed(processes):
                        try:
                            item.kill()
                        except psutil.NoSuchProcess:
                            pass
                    process.wait()
                    raise TimeoutError(f'{engine} exceeded 300 seconds; see log')
                time.sleep(.02)
        if process.wait():
            raise RuntimeError(f'{engine} failed; see {args.output / (engine + ".log")}')
        path = args.output / f'{engine}.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        report.update(peak_rss_mib=peak / 1024**2, process_wall_seconds=time.perf_counter()-started)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(engine, 'peak MiB', round(peak / 1024**2, 1), 'first WAV ready seconds', round(report['samples'][0]['ready_seconds'], 3), flush=True)


if __name__ == '__main__':
    main()
