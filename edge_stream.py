"""Play Edge audio as it arrives; keep a complete PCM WAV for read/repeat."""

import asyncio
import os
import threading
import time
import wave
from thai_speech import expand_thai_speech_units


def prepare_edge_text(text, voice):
    if voice.startswith('th-TH-'):
        return expand_thai_speech_units(text)
    return text


def stream_edge_audio(text, voice, rate, output):
    started = time.perf_counter()
    import edge_tts
    import miniaudio

    reader_fd, writer_fd = os.pipe()
    reader = os.fdopen(reader_fd, 'rb', buffering=0)
    errors = []
    completed = threading.Event()
    timings = {}

    class Source(miniaudio.StreamableSource):
        def read(self, count):
            return reader.read(count)

    def download():
        async def receive(writer):
            async for chunk in edge_tts.Communicate(prepare_edge_text(text, voice), voice=voice, rate=rate).stream():
                if chunk['type'] == 'audio':
                    timings.setdefault('first_mp3_seconds', time.perf_counter()-started)
                    writer.write(chunk['data'])
                    writer.flush()
        try:
            with os.fdopen(writer_fd, 'wb') as writer:
                asyncio.run(receive(writer))
        except Exception as exc:
            errors.append(exc)
        finally:
            timings['network_seconds'] = time.perf_counter()-started

    producer = threading.Thread(target=download, daemon=True)
    producer.start()
    decoded = None
    try:
        decoded = miniaudio.stream_any(Source(), miniaudio.FileFormat.MP3,
                                      nchannels=1, sample_rate=24000)
        with wave.open(str(output), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)

            def playback():
                requested = yield b''
                try:
                    while True:
                        frames = decoded.send(requested)
                        wav.writeframesraw(frames.tobytes())
                        timings.setdefault('first_audio_seconds', time.perf_counter()-started)
                        requested = yield frames
                except StopIteration:
                    pass
                except Exception as exc:
                    errors.append(exc)
                finally:
                    completed.set()

            stream = playback()
            next(stream)
            with miniaudio.PlaybackDevice(nchannels=1, sample_rate=24000,
                                         buffersize_msec=100, callback_periods=2) as device:
                device.start(stream)
                completed.wait()
                # Let the two queued device periods finish before closing the player.
                time.sleep(.2)
        if errors:
            raise errors[0]
        producer.join()
        if errors:
            raise errors[0]
        with wave.open(str(output), 'rb') as wav:
            duration = wav.getnframes()/wav.getframerate()
        if duration <= 0:
            raise ValueError('Edge produced no audio')
        return dict(timings, audio_seconds=duration, cached=False)
    except Exception:
        reader.close()
        producer.join(timeout=1)
        if errors and not isinstance(errors[0], BrokenPipeError):
            raise errors[0]
        raise
    finally:
        reader.close()
        if decoded is not None:
            decoded.close()
        producer.join(timeout=1)
