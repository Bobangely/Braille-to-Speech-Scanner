"""
Persistent Piper TTS Daemon:
Loads the ONNX model once, maintains an on-disk WAV cache,
and processes speak / precache / stop commands via stdin/stdout.
"""
import sys
import os
import json
import wave
import hashlib
import tempfile
import time


def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    model_path = sys.argv[1]

    # Pre-import and load model once
    try:
        from piper import PiperVoice
        voice = PiperVoice.load(model_path, use_cuda=False)
    except Exception as e:
        print(json.dumps({'status': 'init_error', 'error': str(e)}), flush=True)
        sys.exit(1)

    import winsound

    cache_dir = os.path.join(tempfile.gettempdir(), 'braille_piper_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Warm up ONNX runtime with dummy synthesis so first real touch is fast
    try:
        dummy_wav = os.path.join(cache_dir, 'warmup.wav')
        with wave.open(dummy_wav, 'wb') as wav:
            voice.synthesize_wav('ก', wav)
    except Exception:
        pass

    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get('cmd')

            if cmd == 'stop':
                winsound.PlaySound(None, winsound.SND_PURGE)
                print(json.dumps({'status': 'stopped'}), flush=True)

            elif cmd == 'speak':
                text = req.get('text', '')
                if not text or not text.strip():
                    print(json.dumps({'status': 'empty'}), flush=True)
                    continue

                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                wav_path = os.path.join(cache_dir, f"{text_hash}.wav")

                cached = os.path.exists(wav_path)
                if not cached:
                    with wave.open(wav_path, 'wb') as wav:
                        voice.synthesize_wav(text, wav)

                winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                print(json.dumps({'status': 'playing', 'text': text, 'cached': cached}), flush=True)

            elif cmd == 'precache':
                words = req.get('words', [])
                count = 0
                for w in words:
                    if not w or not w.strip():
                        continue
                    w_hash = hashlib.md5(w.encode('utf-8')).hexdigest()
                    w_path = os.path.join(cache_dir, f"{w_hash}.wav")
                    if not os.path.exists(w_path):
                        try:
                            with wave.open(w_path, 'wb') as wav:
                                voice.synthesize_wav(w, wav)
                            count += 1
                        except Exception:
                            pass
                print(json.dumps({'status': 'precached', 'count': count}), flush=True)

            elif cmd == 'quit':
                break
        except Exception as exc:
            print(json.dumps({'status': 'error', 'error': str(exc)}), flush=True)


if __name__ == '__main__':
    main()
