# หน้าเว็บครูและโหมดเสียง Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** โหมดเสียง "พูดเฉพาะคำ", ฟังก์ชันตั้งค่าที่ server ของทีมเรียกได้ และหน้าเว็บครู (มีโหมดจำลอง)

**Architecture:** `SpeechQueue` ตัดสินว่าจะพูดอะไรจาก `mode` และความยาวคำ `panel_settings.py` ตรวจ/บันทึก/ใช้ค่าตั้ง `web/index.html` เป็นไฟล์เดียวที่คุยกับ server ผ่าน `docs/web-api.md` web server เพื่อนในทีมทำ ไม่อยู่ในแผนนี้

**Tech Stack:** Python 3.12 stdlib + `unittest`, HTML/CSS/JS ล้วน ไม่มี dependency ใหม่

**Spec:** `docs/superpowers/specs/2026-10-28-teacher-panel-design.md` · สัญญา API: `docs/web-api.md`

## Global Constraints

- ไม่เพิ่ม dependency ใหม่ทั้ง Python และหน้าเว็บ (ไม่โหลด CDN/ฟอนต์จากอินเทอร์เน็ต)
- ค่าเริ่มต้น: `mode='spell'`, `spell_max_cells=2`, `camera=True`; `spell_max_cells` รับ int 1–5
- ไม่สังเคราะห์เสียงระหว่างอ่าน: `say()` พูดได้เฉพาะข้อความที่เตรียมไว้แล้ว
- เทสต์เดิมทั้งหมดต้องผ่านโดยไม่แก้: `.\.venv\Scripts\python.exe -m unittest discover -s tests`
- ข้อความ error ที่ผู้ใช้เห็นเป็นภาษาไทย
- commit message ภาษาไทย ไม่ใส่บรรทัด Co-Authored-By
- สีหลักหน้าเว็บ `#0050DC` โทนสว่าง แบบ A สองคอลัมน์ เรียงแถวเดียวเมื่อกว้าง < 720 px

## Review Focus

- เปลี่ยนโหมดระหว่างนิ้วอยู่กลางคำ: event ถัดไปใช้โหมดใหม่ ไม่ crash (เทสต์ใน Task 1)
- ไฟล์ settings เสีย/ว่าง/เป็น JSON แต่ไม่ใช่ object: เปิดโปรแกรมได้ด้วยค่าเริ่มต้น (Task 2)
- body ที่ `True`/`False` ส่งมาใน `spell_max_cells` หรือ `"3"` เป็น string: ต้องปฏิเสธ (Task 2)
- เซลล์อ่านไม่ออก (`name=None`) ในโหมด `word`: ยังพูด "ไม่ทราบ" เมื่อเซลล์นั้นถูกพูด (Task 1)
- server ล่มขณะเปิดหน้าเว็บ: แถบแดง และกลับมาเองเมื่อ server กลับมา (Task 4)

---

### Task 1: โหมดเสียงใน `SpeechQueue`

**Files:**
- Modify: `speech_queue.py`
- Test: `tests/test_speech_modes.py` (ใหม่) ใช้ `RecordingSpeaker` แบบเดียวกับ `tests/test_speech_queue.py`

**Interfaces:**
- Produces: `SpeechQueue(speaker, lang='thai', background=False, mode='spell', spell_max_cells=2)`; attribute `mode: str`, `spell_max_cells: int` (ตั้งตรงได้ระหว่างทำงาน), `last_spoken: str | None` (ตั้งใน `say()` ก่อนเล่น)

- [ ] **Step 1: เขียนเทสต์ที่ล้ม** หน้าเทสต์: บรรทัดเดียว คำ 0 = 1 เซลล์ (`'ก'`), คำ 1 = 2 เซลล์ (`'กา'`), คำ 2 = 5 เซลล์ (`'กระดาน'`) ชื่อเซลล์ต่างกันทุกตัว

```python
def test_spell_mode_unchanged(self):           # mode='spell': เซลล์ของคำ 5 เซลล์ถูกพูดชื่อ
def test_word_mode_long_word_cells_silent(self):  # mode='word': EnterCell ของคำ 2 → speaker.calls == []
def test_word_mode_long_word_spoken_at_end(self): # FinishWord(2,'กระดาน') → calls[-1] == ('speak','กระดาน')
def test_word_mode_short_word_spelled(self):   # EnterCell ของคำ 1 (2 เซลล์) → พูดชื่อ
def test_word_mode_threshold_one(self):        # spell_max_cells=1: คำ 1 เงียบตอน EnterCell
def test_word_mode_unready_word_spells(self):  # prepare_many คืน set ที่ไม่มี 'กระดาน' → EnterCell พูดชื่อ
def test_word_mode_unknown_cell(self):         # คำสั้นมีเซลล์ name=None → ('speak','ไม่ทราบ')
def test_mode_switch_mid_word(self):           # spell→EnterCell พูด, ตั้ง queue.mode='word', EnterCell ถัดไปเงียบ
def test_last_spoken(self):                    # หลังพูด 'กระดาน' → queue.last_spoken == 'กระดาน'
def test_cell_without_page_spelled(self):      # mode='word' ไม่มี prepare_page → EnterCell พูดชื่อ (ไม่ crash)
```

- [ ] **Step 2:** `.\.venv\Scripts\python.exe -m unittest tests.test_speech_modes -v` → FAIL (`unexpected keyword argument 'mode'`)
- [ ] **Step 3: implement ใน `speech_queue.py`**
  - `prepare_page()` สร้าง `self._word_of: dict[(line_id, cell_index), word_index]` และ `self._word_cells: dict[word_index, int]` จาก `page.lines` (ข้ามเซลล์ `word_index is None`) และเก็บ `self._words = page.words`
  - `_quiet_cell(event) -> bool`: True เมื่อ `mode == 'word'` และเซลล์อยู่ในคำที่ `_word_cells[w] > spell_max_cells` และ `self.ready(self._words[w])`; หาไม่เจอในแผนที่ → False
  - `handle()`: `EnterCell` ที่ `_quiet_cell` → return; ส่วนอื่นคงเดิม (กฎ `FinishWord` ตารางใน spec เหมือนโหมด spell ทุกกรณี)
- [ ] **Step 4:** `.\.venv\Scripts\python.exe -m unittest discover -s tests` → ผ่านทั้งหมด รวม `test_speech_queue`, `test_finger_state`, `test_finger_reader` เดิม
- [ ] **Step 5: commit** `git add speech_queue.py tests/test_speech_modes.py` → `git commit -m "เพิ่มโหมดพูดเฉพาะคำ สะกดเฉพาะคำสั้น"`

### Task 2: `panel_settings.py`

**Files:**
- Create: `panel_settings.py`
- Test: `tests/test_panel_settings.py`

**Interfaces:**
- Consumes: `SpeechQueue.mode`, `SpeechQueue.spell_max_cells` จาก Task 1
- Produces:
  - `DEFAULTS = {'mode': 'spell', 'spell_max_cells': 2, 'camera': True}`
  - `DEFAULT_PATH = Path(__file__).resolve().parent / 'output' / 'panel_settings.json'`
  - `validate(partial: dict, current: dict) -> dict` (คืน dict ใหม่ ไม่แก้ `current`; ผิด → `ValueError` ข้อความไทย)
  - `load(path: Path) -> dict` · `save(path: Path, settings: dict) -> None` · `apply_settings(speech, path: Path, partial: dict) -> dict`
  - `apply_to(speech, settings: dict) -> None` (ตั้ง `mode`, `spell_max_cells` ไม่บันทึกไฟล์)

- [ ] **Step 1: เขียนเทสต์ที่ล้ม**

```python
def test_validate_partial_keeps_others(self):  # validate({'mode':'word'}, DEFAULTS) == {**DEFAULTS, 'mode':'word'}
def test_validate_rejects(self):               # แต่ละค่า → assertRaises(ValueError):
    # {'mode':'loud'}, {'spell_max_cells':0}, {'spell_max_cells':6}, {'spell_max_cells':True},
    # {'spell_max_cells':'3'}, {'camera':'yes'}, {'volume':5}, not-a-dict เช่น ['mode']
def test_validate_all_or_nothing(self):        # {'mode':'word','spell_max_cells':9} → ValueError และ current ไม่เปลี่ยน
def test_load_missing_file(self):              # path ไม่มี → DEFAULTS
def test_load_broken_file(self):               # ไฟล์ '{' และไฟล์ '[1]' → DEFAULTS (ไม่ raise)
def test_load_fills_missing_keys(self):        # ไฟล์ {"mode":"word"} → {**DEFAULTS,'mode':'word'}
def test_save_then_load(self):                 # save แล้ว load ได้ค่าเดิม; ไม่มีไฟล์ .tmp ค้าง
def test_apply_settings(self):                 # speech ปลอม: หลัง apply {'mode':'word','spell_max_cells':3}
    # speech.mode=='word', speech.spell_max_cells==3, ไฟล์บันทึกแล้ว, คืน settings ทั้งชุด
def test_apply_invalid_changes_nothing(self):  # ValueError, speech และไฟล์ไม่เปลี่ยน
```
  ใช้ `tempfile.TemporaryDirectory()` สำหรับ path

- [ ] **Step 2:** `.\.venv\Scripts\python.exe -m unittest tests.test_panel_settings -v` → FAIL (`No module named 'panel_settings'`)
- [ ] **Step 3: implement** `load` ใช้ค่าจากไฟล์ผ่าน `validate(data, DEFAULTS)`; ไฟล์เสียหรือ validate ไม่ผ่าน → log warning คืน `dict(DEFAULTS)` · `save` เขียน `path.with_suffix('.tmp')` แล้ว `os.replace`, `ensure_ascii=False`, สร้างโฟลเดอร์แม่ · `apply_settings` = `apply_to(speech, validate(partial, load(path)))` แล้ว `save` · ตรวจ int ด้วย `type(v) is int`
- [ ] **Step 4:** `.\.venv\Scripts\python.exe -m unittest discover -s tests` → ผ่านทั้งหมด
- [ ] **Step 5: commit** `git add panel_settings.py tests/test_panel_settings.py` → `git commit -m "เพิ่มฟังก์ชันตรวจ บันทึก และใช้ค่าตั้งของหน้าเว็บครู"`

### Task 3: ต่อค่าตั้งเข้ากับ `finger_reader.py`

**Files:**
- Modify: `finger_reader.py` (`main()` ส่วน `argparse` บรรทัด ~367–398)

**Interfaces:**
- Consumes: `panel_settings.load`, `apply_to`, `apply_settings`, `DEFAULT_PATH`
- Produces: CLI `--speech-mode {spell,word}`, `--spell-max-cells N`, `--settings PATH` (ค่าเริ่มต้น `DEFAULT_PATH`); ตัวแปร `speech` (SpeechQueue) และ `args.settings` ใน `main()` ให้ server ของทีมใช้

- [ ] **Step 1: implement** สร้าง `SpeechQueue` เป็นตัวแปร `speech` ก่อนส่งให้ `FingerReader`; `apply_to(speech, load(args.settings))`; ถ้ามี `--speech-mode`/`--spell-max-cells` → `apply_settings(speech, args.settings, {...เฉพาะที่สั่ง})` (ค่าผิดให้ `parser.error(str(e))`)
- [ ] **Step 2: ตรวจ** `.\.venv\Scripts\python.exe finger_reader.py --help` แสดงสามตัวเลือก และ `--spell-max-cells 9` (พร้อม `--tts-thai-model x`) จบด้วย error ภาษาไทยก่อนเปิดกล้อง
- [ ] **Step 3:** `.\.venv\Scripts\python.exe -m unittest discover -s tests` → ผ่านทั้งหมด
- [ ] **Step 4: commit** `git add finger_reader.py` → `git commit -m "ตั้งโหมดเสียงจาก CLI และไฟล์ค่าตั้ง"`

### Task 4: หน้าเว็บ `web/index.html`

**Files:**
- Create: `web/index.html`

**Interfaces:**
- Consumes: `docs/web-api.md` (`/api/state`, `/api/settings`, `/stream.mjpg`) — ชื่อ field ต้องตรงทุกตัว

- [ ] **Step 1: implement** โครงตาม spec ส่วนที่ 3 (แถบบน + ชิป · ซ้ายกล้อง+สวิตช์ · ขวาใต้นิ้ว · ขวาโหมดเสียง · ล่างข้อความทั้งหน้า)
  - ปุ่มโหมดเป็น `<button aria-pressed>` สองปุ่ม; [−]/[+] จำกัด 1–5; ส่ง POST ทันที; ล้มเหลว → คืนค่าเดิม + แสดง `error`
  - poll `/api/state` ด้วย `setTimeout` ต่อเนื่อง 300 ms (ไม่ใช้ `setInterval` กัน request ซ้อน); fetch ล้ม → แถบแดง "ขาดการเชื่อมต่อ"; สำเร็จครั้งถัดไป → ซ่อนแถบ
  - สวิตช์กล้องปิด → `img.src = ''` + POST `{camera:false}`; เปิด → `img.src = '/stream.mjpg?' + Date.now()`
  - demo: `location.protocol === 'file:' || location.search.includes('demo')` → ข้อมูลปลอม 2 บรรทัด นิ้วเลื่อนเซลล์ทุก 0.8 วินาที, settings เปลี่ยนในหน้า, กล้องแสดงกล่องเทา "ภาพกล้อง (จำลอง)"
  - ตัวอักษรตัดพื้นหลังตาม WCAG AA (ข้อความเทาไม่จางกว่า `#5b6472` บนขาว)
- [ ] **Step 2: ตรวจในเบราว์เซอร์** เปิด `web/index.html` ใน browser pane โหมดจำลอง: ไม่มี error ใน console, ไฮไลต์คำเลื่อน, กดโหมดแล้ว `aria-pressed` สลับ, [+] หยุดที่ 5 · ย่อเป็น 375 px แล้วเรียงแถวเดียวไม่มี scroll แนวนอน · ถ่ายภาพหน้าจอทั้งสองขนาดให้ผู้ใช้ตรวจ
- [ ] **Step 3: ตรวจโหมดขาดการเชื่อมต่อ** เปิดผ่าน `python -m http.server` ในโฟลเดอร์ `web` (ไม่มี `/api/state`) → แถบแดงขึ้น ไม่มี JS exception
- [ ] **Step 4: commit** `git add web/index.html` → `git commit -m "เพิ่มหน้าเว็บครู แบบสองคอลัมน์ พร้อมโหมดจำลอง"`
