"""Play Edge audio as it arrives; keep a complete PCM WAV for read/repeat."""

import asyncio
from array import array
import logging
import os
import queue
import re
import threading
import time
import unicodedata
import wave
from thai_speech import expand_thai_speech_units

THAI_PAUSES_MS = dict(space=0, phrase=0, line=10, sentence=50)
THAI_WORD_PAUSE_MS = 2


def word_pause_position(text, event, cursor):
    token = event.get('text', '')
    start = text.find(token, cursor) if token else -1
    if start < 0 or any(c.isalnum() or unicodedata.category(c).startswith('M')
                        for c in text[cursor:start]):
        raise ValueError('WordBoundary cannot be aligned with source text')
    cursor = start + len(token)
    if (cursor < len(text) and text[cursor] in ' \t' and text[cursor:].strip()
            and all(type(event.get(k)) is int and event[k] >= 0 for k in ('offset', 'duration'))):
        return cursor, round((event['offset']+event['duration'])*24000/10_000_000)
    return cursor, None


def insert_word_pauses(blocks, markers, pause_ms, stats):
    position, index = 0, 0
    silence = array('h', [0])*round(pause_ms*24)
    for block in blocks:
        output, offset = array('h'), 0
        while index < len(markers) and markers[index] <= position+len(block):
            point = markers[index]
            index += 1
            if point < position:
                stats['late_word_boundaries'] = stats.get('late_word_boundaries', 0)+1
                continue
            end = point-position
            output.extend(block[offset:end])
            output.extend(silence)
            stats['word_pauses_inserted'] = stats.get('word_pauses_inserted', 0)+1
            offset = end
        output.extend(block[offset:])
        position += len(block)
        yield output
    stats['late_word_boundaries'] = stats.get('late_word_boundaries', 0)+len(markers)-index


def pack_audio_frames(decoded):
    """Fill device requests so short trimmed blocks do not leave silent gaps."""
    pending = array('h')
    try:
        next(decoded)
        requested = yield array('h')
        while True:
            while len(pending) < requested:
                try:
                    pending.extend(decoded.send(requested-len(pending)))
                except StopIteration:
                    if pending:
                        yield pending
                    return
            frames, pending = pending[:requested], pending[requested:]
            requested = yield frames
    finally:
        decoded.close()


def trim_edge_padding(blocks):
    """Drop only digital silence at chunk edges; preserve every nonzero sample."""
    started, pending = False, 0
    for block in blocks:
        first = next((i for i, value in enumerate(block) if value), len(block))
        if first == len(block):
            if started:
                pending += len(block)
            continue
        end = len(block)
        while not block[end-1]:
            end -= 1
        samples = array('h', [0])*pending + block[first if not started else 0:end]
        pending = len(block)-end
        started = True
        yield samples


def plan_edge_chunks(text, voice, pauses=None):
    durations = {**THAI_PAUSES_MS, **(pauses or {})}
    if any(type(value) not in (int, float) or not 0 <= value <= 5000 for value in durations.values()):
        raise ValueError('Pause durations must be finite milliseconds from 0 to 5000')
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
    merged = []
    for fragment, pause in chunks:
        if merged and merged[-1][1] == 0:
            merged[-1] = (merged[-1][0] + ' ' + fragment, pause)
        else:
            merged.append((fragment, pause))
    return merged


def prepare_edge_text(text, voice):
    if voice.startswith('th-TH-'):
        return expand_thai_speech_units(text)
    return text


def stream_edge_audio(text, voice, rate, output, pauses=None, word_pause_ms=None):
    started = time.perf_counter()
    import edge_tts
    import miniaudio

    word_pause = THAI_WORD_PAUSE_MS if word_pause_ms is None else word_pause_ms
    if type(word_pause) not in (int, float) or not 0 <= word_pause <= 5000:
        raise ValueError('Word pause must be finite milliseconds from 0 to 5000')
    chunks = plan_edge_chunks(prepare_edge_text(text, voice), voice, pauses)
    if not chunks:
        raise ValueError('No speakable text')
    sources = queue.Queue(maxsize=2)
    readers, reader_lock = set(), threading.Lock()
    stopped = threading.Event()
    errors = []
    completed = threading.Event()
    timings = dict(word_pauses_inserted=0, late_word_boundaries=0, unaligned_word_boundaries=0)

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
                markers, cursor, aligned = [], 0, True
                with reader_lock:
                    if stopped.is_set():
                        return
                    reader_fd, writer_fd = os.pipe()
                    reader = os.fdopen(reader_fd, 'rb', buffering=0)
                    readers.add(reader)
                with os.fdopen(writer_fd, 'wb') as writer:
                    if not enqueue((reader, pause, markers)):
                        return
                    options = dict(boundary='WordBoundary') if word_pause else {}
                    async for chunk in edge_tts.Communicate(text_chunk, voice=voice, rate=rate, **options).stream():
                        if chunk['type'] == 'audio':
                            timings.setdefault('first_mp3_seconds', time.perf_counter()-started)
                            writer.write(chunk['data'])
                            writer.flush()
                        elif chunk['type'] == 'WordBoundary' and word_pause and aligned:
                            try:
                                cursor, point = word_pause_position(text_chunk, chunk, cursor)
                                if point is not None:
                                    if markers and point < markers[-1]:
                                        raise ValueError('WordBoundary timestamps are out of order')
                                    markers.append(point)
                            except ValueError:
                                aligned = False
                                timings['unaligned_word_boundaries'] += 1
                                logging.getLogger(__name__).warning('Skipping unaligned WordBoundary pauses', exc_info=True)
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
            reader, pause, markers = source
            decoded_chunk = None
            try:
                decoded_chunk = miniaudio.stream_any(Source(reader), miniaudio.FileFormat.MP3,
                                                     nchannels=1, sample_rate=24000)
                def blocks():
                    try:
                        while True:
                            yield decoded_chunk.send(2048)
                    except StopIteration:
                        return
                audio = insert_word_pauses(blocks(), markers, word_pause, timings) if word_pause else blocks()
                audio = trim_edge_padding(audio)
                for block in audio:
                    offset = 0
                    while offset < len(block):
                        count = min(requested, len(block)-offset)
                        requested = yield block[offset:offset+count]
                        offset += count
            except StopIteration:
                pass
            finally:
                reader.close()
                with reader_lock:
                    readers.discard(reader)
                if decoded_chunk is not None:
                    decoded_chunk.close()
            remaining = round(pause * 24)
            while remaining:
                count = min(requested, remaining)
                requested = yield array('h', [0]) * count
                remaining -= count

    producer = threading.Thread(target=download, daemon=True)
    producer.start()
    decoded = None
    try:
        decoded = pack_audio_frames(decode_chunks())
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
                    inserted_silence_seconds=(sum(pause for _, pause in chunks)
                                             + timings['word_pauses_inserted']*word_pause)/1000)
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
