"""Speech for finger events: the newest event cuts off whatever is playing.

``speaker`` provides ``speak(text, lang)``, ``stop()`` and
``prepare_many(texts, lang) -> set`` (synthesize ahead of time and return the
texts that are ready to play at once). Nothing is synthesized while reading.
"""

from __future__ import annotations

import logging

from finger_state import EnterCell, FinishWord
from page_layout import PageMap
from thai_speech import symbol_name

UNKNOWN = 'ไม่ทราบ'
log = logging.getLogger(__name__)


def spelled(text: str) -> str:
    """A word spoken symbol by symbol, used when its own audio is missing."""
    return ' '.join(symbol_name(char) for char in text)


class SpeechQueue:
    def __init__(self, speaker, lang: str = 'thai'):
        self.speaker, self.lang = speaker, lang
        self._prepared: set[str] = set()

    def prepare_page(self, page: PageMap) -> None:
        """Prepare every text this page can produce: words, spellings, cell names."""
        names = {c.name for line in page.lines for c in line.cells if c.name}
        texts = list(dict.fromkeys([*page.words, *map(spelled, page.words), *sorted(names), UNKNOWN]))
        try:
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
            self.speaker.speak(text, self.lang)
        except Exception:
            log.exception('cannot play %r', text)

    def handle(self, events: list[EnterCell | FinishWord]) -> None:
        if not events:
            return
        event = events[-1]
        if isinstance(event, EnterCell):
            text = event.name or UNKNOWN
        elif symbol_name(event.text) != event.text:
            return  # a one-symbol word: its name was spoken when the finger entered it
        elif self.ready(event.text):
            text = event.text
        else:
            text = spelled(event.text)
            log.warning('word %r not prepared; spelling it', event.text)
        self.say(text)
