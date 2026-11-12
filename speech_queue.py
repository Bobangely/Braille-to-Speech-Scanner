"""เสียงตามเหตุการณ์ของนิ้ว: เหตุการณ์ใหม่สุดจะตัดเสียงที่กำลังเล่น

``speaker`` ต้องมี ``speak(text, lang)``, ``stop()`` และ
``prepare_many(texts, lang) -> set`` (สังเคราะห์ล่วงหน้าและคืนข้อความที่เล่นได้ทันที)
ไม่มีการสังเคราะห์เสียงขณะอ่าน
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
        # 'spell' พูดชื่อทุกเซลล์, 'word' พูดชื่อเซลล์เฉพาะคำที่ยาวไม่เกิน spell_max_cells
        self.mode, self.spell_max_cells = mode, spell_max_cells
        self.last_spoken: str | None = None
        self.muted: bool = False
        self._texts: list[str] = []
        self._prepared: set[str] = set()
        self._worker: threading.Thread | None = None
        self._synthesizing = threading.Lock()  # สังเคราะห์ทีละชุด เพราะสองชุดจะเขียนไฟล์ cache เดียวกัน
        self._words: tuple[str, ...] = ()
        self._word_of: dict[tuple[int, int], int] = {}
        self._word_cells: dict[int, int] = {}

    @property
    def engine(self) -> str:
        # HybridSpeaker สลับเสียงได้ ส่วน CachedWavSpeaker ตัวเดียวมีเสียงเดียว
        return getattr(self.speaker, 'engine', None) or self.speaker.backend

    @property
    def volume(self) -> int:
        return getattr(self.speaker, 'volume', 100)

    def set_audio_config(self, engine: str | None = None, volume: int | None = None) -> dict:
        """เปลี่ยนเสียง ('edge'/'piper') และความดัง (0-100) เสียงใหม่จะเตรียมหน้าปัจจุบันให้"""
        if engine not in (None, 'edge', 'piper'):
            raise ValueError(f'unknown voice engine: {engine}')
        if engine is not None and engine != self.engine:
            if not hasattr(self.speaker, 'engine'):
                raise ValueError(f'only {self.engine} is running: start with --symbol-voice edge to switch')
            self.speaker.engine = engine
            self._start(self._texts)
        if volume is not None:
            self.speaker.volume = max(0, min(100, int(volume)))
        return {'engine': self.engine, 'volume': self.volume}

    def prepare_page(self, page: PageMap) -> None:
        """เตรียมทุกข้อความที่หน้านี้อาจพูด: คำและชื่อเซลล์

        ถ้าใช้ ``background`` จะคืนทันที ชื่อเซลล์มี cache ตั้งแต่เริ่มโปรแกรม
        และแต่ละคำพูดได้เมื่อเสียงพร้อม
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
        except Exception as error:  # เสียงพังต้องไม่ทำให้การอ่านหยุด
            log.warning('cannot prepare page speech: %s', error)
            return
        missing = [t for t in texts if t not in ready]
        if missing:
            log.warning('speech not prepared for %r', missing)
        self._prepared |= set(ready)

    def ready(self, text: str) -> bool:
        # คำที่เขียนไฟล์เสร็จแล้วพูดได้เลย ไม่ต้องรอทั้งหน้า (เสียง Edge บางเสียงช้า)
        file_ready = getattr(self.speaker, 'ready', None)
        return text in self._prepared or bool(file_ready and file_ready(text))

    def say(self, text: str) -> None:
        """พูดข้อความโดยตัดเสียงที่กำลังเล่น ข้อผิดพลาดตอนเล่นจะถูกบันทึก log"""
        try:
            self.speaker.stop()
            self.last_spoken = text
            if not self.muted:
                self.speaker.speak(text, self.lang)
        except Exception:
            log.exception('cannot play %r', text)

    def _quiet_cell(self, event: EnterCell) -> bool:
        """ในโหมด 'word' เซลล์ของคำยาวจะเงียบ ยกเว้นคำนั้นยังพูดไม่ได้"""
        word = self._word_of.get((event.line_id, event.cell_index))
        return (self.mode == 'word' and word is not None
                and self._word_cells[word] > self.spell_max_cells and self.ready(self._words[word]))

    def handle(self, events: list[EnterCell | FinishWord]) -> None:
        if not events:
            return
        # เหตุการณ์ใหม่สุดชนะ แต่เซลล์ที่เงียบต้องไม่บังคำที่จบในเฟรมเดียวกัน
        audible = [e for e in events if not (isinstance(e, EnterCell) and self._quiet_cell(e))]
        if not audible:
            return
        event = audible[-1]
        if isinstance(event, EnterCell):
            text = event.name or UNKNOWN
        elif symbol_name(event.text) != event.text:
            return  # คำที่มีสัญลักษณ์เดียว: พูดชื่อไปแล้วตอนนิ้วเข้าเซลล์
        elif self.ready(event.text):
            text = event.text
        else:
            log.info('word %r not ready yet; its cells were named', event.text)
            return
        self.say(text)
