"""Speech for finger events: the newest event cuts off whatever is playing.

``speaker`` provides ``speak(text, lang)``, ``stop()`` and
``prepare_many(texts, lang) -> set`` (synthesize ahead of time and return the
texts that are ready to play at once). Nothing is synthesized while reading.
"""

from __future__ import annotations

import logging
import threading

from finger_state import EnterCell, FinishWord
from page_layout import PageMap
from thai_speech import symbol_name

UNKNOWN = 'ไม่ทราบ'
log = logging.getLogger(__name__)


class SpeechQueue:
    def __init__(self, speaker, lang: str = 'thai', background: bool = False,
                 mode: str = 'spell', spell_max_cells: int = 2):
        self.speaker, self.lang, self.background = speaker, lang, background
        # 'spell' names every cell; 'word' names cells only of words up to spell_max_cells long.
        self.mode, self.spell_max_cells = mode, spell_max_cells
        self.last_spoken: str | None = None
        self.muted: bool = False
        self._texts: list[str] = []
        self._prepared: set[str] = set()
        self._worker: threading.Thread | None = None
        self._synthesizing = threading.Lock()  # one batch at a time: two would write the same cache files
        self._words: tuple[str, ...] = ()
        self._word_of: dict[tuple[int, int], int] = {}
        self._word_cells: dict[int, int] = {}

    @property
    def engine(self) -> str:
        return getattr(self.speaker, 'engine', 'edge')  # a plain speaker has one voice

    @property
    def volume(self) -> int:
        return getattr(self.speaker, 'volume', 100)

    def set_audio_config(self, engine: str | None = None, volume: int | None = None) -> dict:
        """Switch voice ('edge'/'piper') and volume (0-100); a new voice prepares the current page."""
        if engine not in (None, 'edge', 'piper'):
            raise ValueError(f'unknown voice engine: {engine}')
        if engine is not None and engine != self.engine:
            self.speaker.engine = engine
            self._start(self._texts)
        if volume is not None:
            self.speaker.volume = max(0, min(100, int(volume)))
        return {'engine': self.engine, 'volume': self.volume}

    def prepare_page(self, page: PageMap) -> None:
        """Prepare every text this page can produce: words and cell names.

        With ``background`` it returns at once: cell names are cached from startup,
        and each word becomes speakable when its audio is ready.
        """
        self._words = page.words
        self._word_of = {(line.line_id, c.index): c.word_index
                         for line in page.lines for c in line.cells if c.word_index is not None}
        self._word_cells = {}
        for word in self._word_of.values():
            self._word_cells[word] = self._word_cells.get(word, 0) + 1
        names = {c.name for line in page.lines for c in line.cells if c.name}
        self._texts = list(dict.fromkeys([*page.words, *sorted(names), UNKNOWN]))
        self._start(self._texts)

    def _start(self, texts: list[str]) -> None:
        if self.background:
            self._worker = threading.Thread(target=self._prepare, args=(texts,), daemon=True)
            self._worker.start()
        else:
            self._prepare(texts)

    def wait(self) -> None:
        if self._worker is not None:
            self._worker.join()

    def _prepare(self, texts: list[str]) -> None:
        try:
            with self._synthesizing:
                ready = self.speaker.prepare_many(texts, self.lang)
        except Exception as error:  # a broken voice must not stop the reader
            log.warning('cannot prepare page speech: %s', error)
            return
        missing = [t for t in texts if t not in ready]
        if missing:
            log.warning('speech not prepared for %r', missing)
        self._prepared |= set(ready)

    def ready(self, text: str) -> bool:
        return text in self._prepared

    def say(self, text: str) -> None:
        """Speak a message, cutting off whatever is playing; playback errors are logged."""
        try:
            self.speaker.stop()
            self.last_spoken = text
            if not self.muted:
                self.speaker.speak(text, self.lang)
        except Exception:
            log.exception('cannot play %r', text)

    def _quiet_cell(self, event: EnterCell) -> bool:
        """In 'word' mode a long word's cells stay quiet, unless the word itself cannot be spoken yet."""
        word = self._word_of.get((event.line_id, event.cell_index))
        return (self.mode == 'word' and word is not None
                and self._word_cells[word] > self.spell_max_cells and self.ready(self._words[word]))

    def handle(self, events: list[EnterCell | FinishWord]) -> None:
        if not events:
            return
        # Newest event wins, but a quiet cell must not hide the word finished in the same frame.
        audible = [e for e in events if not (isinstance(e, EnterCell) and self._quiet_cell(e))]
        if not audible:
            return
        event = audible[-1]
        if isinstance(event, EnterCell):
            text = event.name or UNKNOWN
        elif symbol_name(event.text) != event.text:
            return  # a one-symbol word: its name was spoken when the finger entered it
        elif self.ready(event.text):
            text = event.text
        else:
            log.info('word %r not ready yet; its cells were named', event.text)
            return
        self.say(text)
