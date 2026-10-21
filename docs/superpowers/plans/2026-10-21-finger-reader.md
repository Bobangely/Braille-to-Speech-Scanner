# Finger Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** โปรแกรม `finger_reader.py` ออกเสียงชื่อสัญลักษณ์ของเซลล์เบรลล์นูนใต้ปลายนิ้ว และออกเสียงคำเต็มเมื่อนิ้วออกจากคำไปทางขวา

**Architecture:** สองช่วง: (1) จดจำหน้า — calibrate ด้วย ArUco 4 มุม อ่านบรรทัดสีครั้งเดียวเป็น `PageMap`; (2) อ่านตามนิ้ว — ทุกเฟรมแปลงปลายนิ้ว (MediaPipe) เป็น mm แล้วให้ `FingerState` (logic ล้วน) ส่งเหตุการณ์ไปที่ `SpeechQueue`

**Tech Stack:** Python 3.12, OpenCV (`cv2.aruco`), numpy, mediapipe 1.1.1, Piper/MMS ผ่าน `piper_tts.PiperSpeech`, unittest

**Spec:** `docs/superpowers/specs/2026-10-21-finger-reader-design.md`

## Global Constraints

- test ใช้ `unittest` รันด้วย `.venv/Scripts/python.exe -m unittest discover -s tests -t .` (โปรเจกต์ไม่มี pytest)
- หน่วยพิกัดหน้าเป็น mm ต้นทางมุมซ้ายบนของหน้า A5 แนวนอน 210 × 148 mm, x ขวา y ลง (ตาม `page_calibration.py`)
- marker ArUco `DICT_4X4_50` ID 0–3 = TL, TR, BR, BL ขนาด 15 mm ห่างขอบหน้า 10 mm
- ค่าเริ่มต้น: dwell 0.15 s, hysteresis 0.5 mm, แผนที่นิ่ง 3 เฟรมติดกัน, marker ขยับเกิน 3 mm = หน้าเปลี่ยน, marker หายเกิน 2 s = หน้าเปลี่ยน, ภาพหน้าที่แก้เอียง 8 px/mm
- ห้ามเดาเซลล์ที่อ่านไม่ออก: พูด "ไม่ทราบ"
- ข้อความ commit ภาษาไทย ลงท้าย `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- ห้ามแก้พฤติกรรมของ `camera_reader.py` (โหมดสแกนเดิม)

## Review Focus

1. นิ้วสั่นอยู่บนเส้นแบ่งเซลล์ — ต้องไม่พูดสลับไปมา (Task 1 `test_jitter_on_boundary_speaks_once`)
2. นิ้วออกจากบรรทัดนูนในแนวตั้ง (ขึ้นไปบนบรรทัดสี) แล้วกลับมาเซลล์เดิม — ต้องพูดชื่อซ้ำ ไม่พูดคำเต็ม (Task 1 `test_leave_vertically_then_return_repeats_symbol_only`)
3. นิ้วเลื่อนไปคู่บรรทัดถัดไปกลางคำ — ต้องไม่ส่ง `finish_word` ของคำเก่า (Task 1 `test_jump_to_other_line_does_not_finish_word`)
4. มือบัง marker 1–3 ตัวระหว่างอ่าน — ใช้ calibration ล่าสุด ไม่รีเซ็ตแผนที่ (Task 3 `test_partial_occlusion_keeps_last_calibration`)
5. เปิดหน้าใหม่โดยมือยังอยู่ในภาพ — ห้ามจดจำหน้าจนกว่ามือออก (Task 7 `test_page_learning_waits_for_no_hand`)

---

### Task 1: ชนิดข้อมูลหน้า และ `FingerState`

**Files:**
- Create: `page_layout.py`
- Create: `finger_state.py`
- Test: `tests/test_finger_state.py`

**Interfaces:**
- Produces (`page_layout.py`, frozen dataclasses):
  - `CellSlot(index: int, x1_mm: float, x2_mm: float, dots: frozenset[int], name: str | None, word_index: int | None)` — `name=None` = อ่านไม่ออก, `word_index=None` = เซลล์ว่าง
  - `LinePair(line_id: int, raised_y1_mm: float, raised_y2_mm: float, cells: tuple[CellSlot, ...])`
  - `PageMap(lines: tuple[LinePair, ...], words: tuple[str, ...])` — `words[word_index]` คือข้อความคำ
- Produces (`finger_state.py`):
  - `EnterCell(line_id: int, cell_index: int, name: str | None)`, `FinishWord(word_index: int, text: str)` (frozen dataclasses)
  - `FingerState(page: PageMap, dwell_s: float = 0.15, hysteresis_mm: float = 0.5)`
  - `FingerState.update(point_mm: tuple[float, float] | None, now_s: float) -> list[EnterCell | FinishWord]`
  - `FingerState.reset() -> None`

- [ ] **Step 1: เขียน test ที่ fail** — fixture: 1 คู่บรรทัด `raised_y 20–30 mm`, เซลล์กว้าง 6 mm เริ่ม x=10: `ไ`(word 0) `ก`(0) `่`(0) ว่าง `ม`(1); `words=('ไก่', 'ม')` ชื่อ `สระไอ, กอไก่, ไม้เอก, None-for-empty, มอม้า`

```python
def test_reading_kai_then_space(self):   # ไล่นิ้ว x=13,19,25,31 ที่ y=25 ทีละ 0.2 s
    self.assertEqual(events, [EnterCell(0,0,'สระไอ'), EnterCell(0,1,'กอไก่'),
                              EnterCell(0,2,'ไม้เอก'), FinishWord(0,'ไก่')])
def test_dwell_below_threshold_is_silent(self)        # อยู่เซลล์ 0.1 s แล้วออกนอกบรรทัด → []
def test_backwards_speaks_symbol_without_word(self)   # 0→1→0 → EnterCell x3, ไม่มี FinishWord
def test_skipped_cells_speak_only_current(self)       # กระโดด x=13→25 → EnterCell(0,0), EnterCell(0,2) เท่านั้น
def test_jitter_on_boundary_speaks_once(self)         # สลับ x=15.8/16.2 ทุก 0.05 s นาน 1 s → EnterCell ของ cell 0 ครั้งเดียว
def test_lost_finger_returns_outside(self)            # point None แล้วกลับเซลล์เดิม → EnterCell ซ้ำ
def test_leave_vertically_then_return_repeats_symbol_only(self)  # y=35 แล้วกลับ → EnterCell ซ้ำ ไม่มี FinishWord
def test_jump_to_other_line_does_not_finish_word(self) # fixture คู่บรรทัดที่สอง raised_y 50–60; ไปกลางคำ → ไม่มี FinishWord(0)
def test_past_line_end_finishes_last_word(self)       # หลัง ม ไป x=70 → FinishWord(1,'ม')
def test_unreadable_cell_reports_none_name(self)      # name=None → EnterCell(...,None)
```

- [ ] **Step 2: รัน `.venv/Scripts/python.exe -m unittest tests.test_finger_state -v`** — Expected: FAIL `ModuleNotFoundError: finger_state`
- [ ] **Step 3: เขียน `page_layout.py` และ `finger_state.py`** — สถานะ `outside / candidate(cell, since) / active(cell)`; hysteresis: ขณะ active เซลล์เดิมขยายขอบออก 0.5 mm ก่อนเทียบ; `FinishWord` ส่งเมื่อ active เปลี่ยนจากเซลล์ของคำ w ไปเซลล์ว่างทางขวาหรือ x > ขอบขวาเซลล์สุดท้ายของบรรทัดเดียวกัน และคำ w ยังไม่ถูกส่งตั้งแต่ครั้งล่าสุดที่นิ้วเข้าคำ w
- [ ] **Step 4: รัน test** — Expected: 10 tests OK
- [ ] **Step 5: commit** `git add page_layout.py finger_state.py tests/test_finger_state.py` ข้อความ `เพิ่ม state machine ของปลายนิ้ว`

### Task 2: ชื่อสัญลักษณ์สำหรับเสียง

**Files:**
- Modify: `thai_speech.py` (เพิ่มฟังก์ชันท้ายไฟล์)
- Test: `tests/test_symbol_names.py`

**Interfaces:**
- Produces: `symbol_name(text: str) -> str` — พยัญชนะใช้ `expand_thai_speech_units` เดิม; สระ วรรณยุกต์ เครื่องหมาย ใช้ตาราง `SYMBOL_NAMES: dict[str, str]`

- [ ] **Step 1: test ที่ fail**

```python
def test_kai_symbols(self):
    self.assertEqual([symbol_name(c) for c in ('ไ', 'ก', '่')], ['สระไอ', 'กอไก่', 'ไม้เอก'])
def test_all_vowels_and_tones_have_names(self)  # ทุก key ใน SYMBOL_NAMES ไม่ว่าง และ ่ ้ ๊ ๋ ์ ะ ั า ำ ิ ี ึ ื ุ ู เ แ โ ใ ไ ็ อยู่ครบ
def test_unknown_symbol_returns_text_itself(self)  # symbol_name('?') == '?'
```

- [ ] **Step 2: รันแล้ว FAIL `ImportError: symbol_name`**
- [ ] **Step 3: เพิ่ม `SYMBOL_NAMES` และ `symbol_name`** — ชื่อ: `่` ไม้เอก, `้` ไม้โท, `๊` ไม้ตรี, `๋` ไม้จัตวา, `์` ไม้ทัณฑฆาต, `็` ไม้ไต่คู้, `ะ` สระอะ, `ั` ไม้หันอากาศ, `า` สระอา, `ำ` สระอำ, `ิ` สระอิ, `ี` สระอี, `ึ` สระอึ, `ื` สระอือ, `ุ` สระอุ, `ู` สระอู, `เ` สระเอ, `แ` สระแอ, `โ` สระโอ, `ใ` สระใอ, `ไ` สระไอ; ค่า `char` หลายตัวอักษรที่ decoder คืนสำหรับสระประสม: `เ◌อ` สระเออ, `เ◌า` สระเอา, `เ◌ีย` สระเอีย, `เ◌ือ` สระเอือ, `◌ัว` สระอัว
- [ ] **Step 4: รัน test OK และรัน `tests.test_thai_speech_names` ต้องยังผ่าน**
- [ ] **Step 5: commit** `เพิ่มชื่อสัญลักษณ์สระและวรรณยุกต์`

### Task 3: `PageFormat` A5 แนวนอน และติดตาม calibration

**Files:**
- Modify: `page_calibration.py` (ค่า default ของ `PageFormat`, เพิ่ม `CalibrationTracker`)
- Create: `tools/export_markers.py`
- Test: `tests/test_page_calibration.py`

**Interfaces:**
- Produces: `PageFormat(width_mm=210, height_mm=148, margin_mm=10, marker_size_mm=15, line_pairs=3, ...)` — ลบ `line_origin_mm`/`cells_per_line` ออกจากการใช้ใน `page_map` (ยังใช้ใน `render_page` สำหรับ test ได้ โดยเพิ่ม `line_origins_mm: tuple[tuple[float, float], ...]`)
- Produces: `CalibrationTracker(fmt, move_mm=3.0, lost_s=2.0)` กับ `.update(image, now_s) -> tuple[PageCalibration | None, bool]` คืน (calibration ที่ใช้ได้ล่าสุด, `page_changed`)
- Produces: `tools/export_markers.py --out DIR --px 600` เขียน `marker_{id}_{TL|TR|BR|BL}.png`

- [ ] **Step 1: test ที่ fail**

```python
def test_default_format_is_a5_landscape(self)          # 210×148, marker 15 mm ห่าง 10 mm, validate() ผ่าน
def test_partial_occlusion_keeps_last_calibration(self) # ภาพครบแล้วทาสีขาวทับ marker TR → calibration เดิม, page_changed False
def test_marker_shift_over_3mm_flags_page_change(self)  # เลื่อนภาพ 4 mm (32 px ที่ 8 px/mm render) → page_changed True
def test_markers_lost_over_2s_flags_page_change(self)   # ภาพว่าง now 0→2.1 s → page_changed True; ที่ 1.9 s → False
def test_export_markers_writes_four_files(self)        # tmpdir มี 4 ไฟล์ และ detect_markers อ่าน ID ได้ตรงชื่อ
```

- [ ] **Step 2: รัน `tests.test_page_calibration` — FAIL**
- [ ] **Step 3: แก้ default, เพิ่ม `CalibrationTracker` และ `tools/export_markers.py`** — การขยับวัดจากระยะ max ของ marker centre (mm) ระหว่าง calibration ใหม่ที่ ok กับตัวที่ใช้อยู่
- [ ] **Step 4: รันทั้ง `tests.test_page_calibration` OK**
- [ ] **Step 5: commit** `ปรับหน้าเป็น A5 แนวนอนและติดตามการปรับเทียบ`

### Task 4: `page_map` อ่านบรรทัดสี

**Files:**
- Create: `page_map.py`
- Test: `tests/test_page_map.py`

**Interfaces:**
- Consumes: `PageCalibration`, `warp_page` (Task 3); `ColoredCellStream.find_dots` (`colored_braille.py`); `decode_cells_verbose(cells, 'thai')` (`decoder.py`); `symbol_name` (Task 2); `PageMap/LinePair/CellSlot` (Task 1)
- Produces: `read_page(image, calibration, fmt) -> PageMap`; `PageLearner(required=3)` กับ `.update(page_map: PageMap) -> PageMap | None` คืนแผนที่เมื่อเท่ากัน 3 ครั้งติด

วิธีอ่าน: warp เป็น 8 px/mm → `find_dots` → จัดแถวด้วย y (ระยะจุด `fmt.dot_spacing_mm`) → จัดเซลล์ด้วย x โดยใช้ `fmt.cell_pitch_mm` และจุดเริ่มบรรทัด = x ต่ำสุดของบรรทัด ปัดเป็นช่องที่ใกล้ที่สุด → ช่วงแนวตั้งของบรรทัดนูน = ตั้งแต่ขอบล่างบรรทัดสี + 1 mm ถึงก่อนบรรทัดสีถัดไป − 1 mm (บรรทัดสุดท้าย: + `fmt.raised_band_mm`, ค่าเริ่ม 12) → decode ทีละบรรทัด แบ่งคำที่เซลล์ว่าง

- [ ] **Step 1: test ที่ fail**

```python
def test_synthetic_page_reads_all_cells(self)      # render_page 3 คู่บรรทัด ตรง: dots และ words ตรงกับต้นฉบับ
def test_tilted_page_reads_all_cells(self)         # warpPerspective เอียง 8° ก่อน calibrate → ผลเท่าเดิม
def test_missing_dot_marks_cell_unreadable(self)   # ลบจุดที่ทำให้ pattern ไม่มีใน decoder → name None
def test_raised_band_sits_between_colored_lines(self)
def test_learner_needs_three_equal_maps(self)      # A,A,B,A,A,A → คืน None ×5 แล้ว A
```

- [ ] **Step 2: รัน — FAIL**
- [ ] **Step 3: เขียน `page_map.py`**
- [ ] **Step 4: รัน OK**
- [ ] **Step 5: ตรวจกับหน้าจริงจากทีม** (ขั้นนี้รอข้อมูลทีม): `.venv/Scripts/python.exe tools/diagnostics/page_map_accuracy.py sample.png --truth sample.txt` ต้องได้ ≥90% ในแสงทั่ว; ปรับ `dot_spacing_mm`/`cell_pitch_mm` ใน `PageFormat` ตามค่าที่วัด
- [ ] **Step 6: commit** `เพิ่มการอ่านแผนที่หน้าจากบรรทัดสี` (รวม `tools/diagnostics/page_map_accuracy.py` ที่รายงาน % ถูกต่อเซลล์)

### Task 5: `fingertip` ย้ายจาก spike

**Files:**
- Create: `fingertip.py`
- Modify: `tools/diagnostics/hand_landmarker_spike.py` (import จาก `fingertip`)
- Modify: `tests/test_hand_landmarker_spike.py` (import path)

**Interfaces:**
- Produces: `fingertip_pixels`, `ActiveFingertipLock` (ย้ายตามเดิม ไม่เปลี่ยน signature); `FingertipTracker(model_path, dwell_ms=250, radius_px=20)` กับ `.update(frame_bgr, now_ms) -> tuple[int, int] | None` และ `.hand_present: bool`

- [ ] **Step 1: ย้ายโค้ด แก้ import ใน spike และ test**
- [ ] **Step 2: เพิ่ม test `test_tracker_returns_none_without_hands`** (mock landmarker คืนรายการว่าง → `None`, `hand_present False`) และ `test_tracker_returns_locked_tip_pixels`
- [ ] **Step 3: รัน `tests.test_hand_landmarker_spike tests.test_fingertip` OK**
- [ ] **Step 4: commit** `แยกโมดูลตรวจปลายนิ้ว`

### Task 6: `SpeechQueue`

**Files:**
- Create: `speech_queue.py`
- Test: `tests/test_speech_queue.py`

**Interfaces:**
- Consumes: `EnterCell`, `FinishWord` (Task 1); `symbol_name` (Task 2)
- Produces: `SpeechQueue(speaker, lang='thai')` โดย `speaker` มี `speak(text, lang) -> bool`, `stop() -> None`, `prepare(text, lang) -> None`; เมธอด `.prepare_page(page: PageMap)`, `.handle(events) -> None`, `.ready(text) -> bool`

พฤติกรรม: `handle` ใช้เหตุการณ์ล่าสุดในรายการ → `stop()` แล้ว `speak()`; `EnterCell` name None → "ไม่ทราบ"; `FinishWord` ที่ `ready(text)` False → พูดชื่อสัญลักษณ์ของคำต่อกัน และ log warning

- [ ] **Step 1: test ที่ fail (speaker ปลอมบันทึกการเรียก)**

```python
def test_new_event_stops_previous_audio(self)     # calls == ['stop', ('speak','สระไอ'), 'stop', ('speak','กอไก่')]
def test_unknown_cell_says_mai_sap(self)          # ('speak','ไม่ทราบ')
def test_unprepared_word_is_spelled(self)         # ('speak','สระไอ กอไก่ ไม้เอก')
def test_prepare_page_prepares_every_word(self)   # prepare เรียกด้วย 'ไก่','ม'
```

- [ ] **Step 2–4: FAIL → เขียน → OK**
- [ ] **Step 5: commit** `เพิ่มคิวเสียงสำหรับอ่านตามนิ้ว`

### Task 7: `finger_reader.py` ต่อทุกส่วน

**Files:**
- Create: `finger_reader.py`
- Test: `tests/test_finger_reader.py`

**Interfaces:**
- Consumes: ทุก Produces ของ Task 1–6; `PiperSpeech` (`piper_tts.py`) ห่อเป็น speaker: `prepare` เรียก `synthesize_wav` ลง cache ของ `PiperSpeech`
- Produces: `FingerReader(camera, tracker, calibrator, speech, fmt)` กับ `.step(frame, now_s) -> str` คืนสถานะ `waiting_page | learning | reading`; CLI `finger_reader.py --camera 0 --log-latency`

- [ ] **Step 1: test ที่ fail (ทุกชิ้นเป็นของปลอม)**

```python
def test_page_learning_waits_for_no_hand(self)     # hand_present True → สถานะ learning ค้าง, speech 'ยกมือออกก่อน' ครั้งเดียว
def test_no_markers_asks_to_place_page(self)       # 'วางหน้าให้เห็นมุมครบ'
def test_learned_page_announces_lines(self)        # 'พร้อม 3 บรรทัด' และสถานะ reading
def test_page_change_returns_to_learning(self)     # calibrator page_changed True → learning, FingerState reset
def test_fingertip_mapped_to_mm_drives_events(self)# tracker คืน px → calibration.to_page → EnterCell ถึง speech
def test_missing_tts_model_fails_at_startup(self)  # speaker status ไม่ Ready → raise ตอนสร้าง
```

- [ ] **Step 2–4: FAIL → เขียน → OK** (`--log-latency` เขียน CSV: frame_ts, tip_ts, enter_ts, active_ts, speak_ts)
- [ ] **Step 5: รัน test ทั้งชุด** — Expected: OK ทุกตัว รวม test เดิม
- [ ] **Step 6: commit** `เพิ่มโปรแกรมอ่านเบรลล์ตามปลายนิ้ว`
- [ ] **Step 7: ทดสอบกล้องจริง** — ไล่นิ้ว 20 ครั้ง รายงาน median/p90 ของ (speak_ts − frame_ts); ให้คนอ่าน `ไก่` และ 3 ประโยค บันทึกผลใน `docs/superpowers/specs/` ไม่ใช่ในโค้ด

### Task 8: template Word (รอค่าสเลท)

**Files:**
- Create: `tools/make_book_template.py` (สร้าง `.docx` A4 แนวตั้ง 2 หน้าต่อแผ่น ด้านหน้า marker 8 ตัว + ย่อหน้าบรรทัดสี, ด้านหลังขีดนำกลับซ้ายขวา)
- Test: `tests/test_book_template.py`

**Interfaces:**
- Consumes: `PageFormat` (Task 3), ค่าที่ทีมวัด: `cell_pitch_mm`, `line_pitch_mm`, `cells_per_row`
- Produces: CLI `--cell-pitch-mm 6.2 --line-pitch-mm 10 --out book.docx`; คำนวณ Expanded = pitch/0.3528 − 16.2 pt และ Line spacing Exactly = line_pitch/0.3528 pt

- [ ] **Step 1: test** — `test_expanded_spacing_formula` (6.2 → 1.37 pt ±0.01), `test_marker_positions_cm` (ครึ่งบน TL (1.0,1.0) TR (18.5,1.0) BR (18.5,12.35) BL (1.0,12.35); ครึ่งล่าง +14.85 cm), `test_back_guides_are_mirrored`
- [ ] **Step 2–4: FAIL → เขียน → OK** (ถ้าต้องเพิ่ม dependency `python-docx` ให้ถามผู้ใช้ก่อน)
- [ ] **Step 5: commit** `เพิ่มตัวสร้าง template หนังสือสองบรรทัด`
