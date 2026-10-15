"""Play Edge audio as it arrives; keep a complete PCM WAV for read/repeat."""

import asyncio
from array import array
import os
import queue
import re
import threading
import time
import wave
from thai_speech import expand_thai_speech_units

THAI_PAUSES_MS = dict(space=30, phrase=80, line=130, sentence=200)


def plan_edge_chunks(text, voice, pauses=None):
    if not voice.startswith('th-TH-'):
        return [(text, 0)] if text.strip() else []
    durations = {**THAI_PAUSES_MS, **(pauses or {})}
    if any(type(value) is not int or not 0 <= value <= 5000 for value in durations.values()):
        raise ValueError('Pause durations must be integer milliseconds from 0 to 5000')
    chunks, start = [], 0
    for boundary in re.finditer(r'[ \t\r\n]+|[.!?ฯ]+', text):
        mark = boundary[0]
        if (mark == '.' and boundary.start() > 0 and boundary.end() < len(text)
                and text[boundary.start()-1].isdigit() and text[boundary.end()].isdigit()):
            continue
        punctuation = not mark.isspace()
        kind = ('sentence' if punctuation else 'line' if '\n' in mark or '\r' in mark
                else 'phrase' if len(mark) > 1 or '\t' in mark else 'space')
        fragment = text[start:boundary.end() if punctuation else boundary.start()]
        if fragment.strip(' \t\r\n.!?ฯ'):
            chunks.append([fragment, durations[kind]])
        elif chunks:
            if punctuation:
                chunks[-1][0] += fragment
            chunks[-1][1] = max(chunks[-1][1], durations[kind])
        start = boundary.end()
    if text[start:].strip():
        chunks.append([text[start:], 0])
    # Edge returns no audio for standalone bullets/punctuation; keep them with words.
    spoken, prefix, prefix_pause = [], '', 0
    for fragment, pause in chunks:
        if not any(char.isalnum() for char in fragment):
            prefix += fragment + ' '
            prefix_pause = max(prefix_pause, pause)
            continue
        if spoken and prefix:
            spoken[-1][1] = max(spoken[-1][1], prefix_pause)
        spoken.append([prefix + fragment, pause])
        prefix, prefix_pause = '', 0
    if spoken and prefix:
        spoken[-1][0] += ' ' + prefix.rstrip()
    chunks = spoken
    if chunks:
        chunks[-1][1] = 0
    return [tuple(chunk) for chunk in chunks]


def prepare_edge_text(text, voice):
    if voice.startswith('th-TH-'):
        return expand_thai_speech_units(text)
    return text


def stream_edge_audio(text, voice, rate, output, pauses=None):
    started = time.perf_counter()
    import edge_tts
    import miniaudio

    chunks = plan_edge_chunks(prepare_edge_text(text, voice), voice, pauses)
    if not chunks:
        raise ValueError('No speakable text')
    sources = queue.Queue(maxsize=2)
    readers, reader_lock = set(), threading.Lock()
    stopped = threading.Event()
    errors = []
    completed = threading.Event()
    timings = {}

    class Source(miniaudio.StreamableSource):
        def __init__(self, reader):
            self.reader = reader

        def read(self, count):
            return self.reader.read(count)

    def enqueue(item):
        while not stopped.is_set():
            try:
                sources.put(item, timeout=.05)
                return True
            except queue.Full:
                pass
        return False

    def download():
        async def receive():
            for text_chunk, pause in chunks:
                with reader_lock:
                    if stopped.is_set():
                        return
                    reader_fd, writer_fd = os.pipe()
                    reader = os.fdopen(reader_fd, 'rb', buffering=0)
                    readers.add(reader)
                with os.fdopen(writer_fd, 'wb') as writer:
                    if not enqueue((reader, pause)):
                        return
                    async for chunk in edge_tts.Communicate(text_chunk, voice=voice, rate=rate).stream():
                        if chunk['type'] == 'audio':
                            timings.setdefault('first_mp3_seconds', time.perf_counter()-started)
                            writer.write(chunk['data'])
                            writer.flush()
        try:
            asyncio.run(receive())
        except Exception as exc:
            errors.append(exc)
        finally:
            timings['network_seconds'] = time.perf_counter()-started
            enqueue(None)

    def decode_chunks():
        requested = yield array('h')
        while (source := sources.get()) is not None:
            reader, pause = source
            decoded_chunk = None
            try:
                decoded_chunk = miniaudio.stream_any(Source(reader), miniaudio.FileFormat.MP3,
                                                     nchannels=1, sample_rate=24000)
                while True:
                    requested = yield decoded_chunk.send(requested)
            except StopIteration:
                pass
            finally:
                reader.close()
                with reader_lock:
                    readers.discard(reader)
                if decoded_chunk is not None:
                    decoded_chunk.close()
            remaining = pause * 24
            while remaining:
                count = min(requested, remaining)
                requested = yield array('h', [0]) * count
                remaining -= count

    producer = threading.Thread(target=download, daemon=True)
    producer.start()
    decoded = None
    try:
        decoded = decode_chunks()
        next(decoded)
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
        return dict(timings, audio_seconds=duration, cached=False, chunk_count=len(chunks),
                    inserted_silence_seconds=sum(pause for _, pause in chunks)/1000)
    except Exception:
        stopped.set()
        with reader_lock:
            for reader in readers:
                reader.close()
        producer.join(timeout=1)
        if errors and not isinstance(errors[0], BrokenPipeError):
            raise errors[0]
        raise
    finally:
        stopped.set()
        with reader_lock:
            for reader in readers:
                reader.close()
        if decoded is not None:
            decoded.close()
        producer.join(timeout=1)
