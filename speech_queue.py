"""Speech for finger events: the newest event cuts off whatever is playing.

``speaker`` provides ``speak(text, lang)``, ``stop()`` and ``prepare(text, lang)``
(synthesize ahead of time so playback starts at once).
"""

from __future__ import annotations

import logging

from finger_state import EnterCell, FinishWord
from page_layout import PageMap
from thai_speech import symbol_name

UNKNOWN = 'ไม่ทราบ'
log = logging.getLogger(__name__)


class SpeechQueue:
    def __init__(self, speaker, lang: str = 'thai'):
        self.speaker, self.lang = speaker, lang
        self._prepared: set[str] = set()

    def prepare_page(self, page: PageMap) -> None:
        for word in page.words:
            try:
                self.speaker.prepare(word, self.lang)
            except Exception as error:  # a missing word voice must not stop the reader
                log.warning('cannot prepare speech for %r: %s', word, error)
            else:
                self._prepared.add(word)

    def ready(self, text: str) -> bool:
        return text in self._prepared

    def say(self, text: str) -> None:
        """Speak a status message, cutting off whatever is playing."""
        self.speaker.stop()
        self.speaker.speak(text, self.lang)

    def handle(self, events: list[EnterCell | FinishWord]) -> None:
        if not events:
            return
        event = events[-1]
        if isinstance(event, EnterCell):
            text = event.name or UNKNOWN
        elif self.ready(event.text):
            text = event.text
        else:
            text = ' '.join(symbol_name(char) for char in event.text)
            log.warning('word %r not prepared; spelling it', event.text)
        self.say(text)
