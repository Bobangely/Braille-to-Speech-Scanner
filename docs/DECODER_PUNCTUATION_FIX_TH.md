# สาเหตุของ other: 1 และการแก้ตัวแปลเครื่องหมาย

## ข้อสรุปจากภาพจริง

อาการในภาพนี้เกิดจาก **mapping เครื่องหมายสองเซลล์ไม่ครบ** ระบบทำ detection และ decoding จบแล้ว แต่ผลมีคำเตือน ทำให้ไม่ผ่านการยืนยันข้อความ ไม่ได้รอคิวหรือวนรอเซลล์ถัดไปค้างอยู่

Diagnostic `1789489556525254800` เฟรม 6699 มี 22 เซลล์จากบรรทัดที่มีปัญหา:

- C12 = `[4,5,6]`
- C13 = `[2,5]`
- ก่อนแก้: `คำที่มีใช้�ู ฐาน ฐิติ`
- คำเตือน C12: `unknown_or_incomplete_symbol`
- หลังแก้: **`คำที่มีใช้: ฐาน ฐิติ`**

คู่นี้เป็นเครื่องหมาย colon แบบมี prefix ของเบรลล์ไทย ตรวจจาก Liblouis `tables/th-g1.utb` บรรทัด 84–89 ซึ่งกำหนด comma, colon, semicolon, period, exclamation และ question mark แบบ prefix 456 เก็บสำเนาที่ตรวจไว้ใน `scratch/decoder_punctuation_review/th-g1.utb`

แหล่งต้นฉบับ:

```text
https://raw.githubusercontent.com/liblouis/liblouis/master/tables/th-g1.utb
```

การเพิ่มนี้รองรับเครื่องหมายกลุ่มดังกล่าว ไม่ได้เปลี่ยนระบบเป็นตัวอ่านอักษรย่อภาษาไทยทั้งชุด

## เหตุใดหน้าจอจึงดูเหมือนค้าง

1. `tokenize_thai()` หาคู่ 456 + 25 ไม่พบ จึงสร้าง `�` พร้อม warning ให้ C12 ส่วน C13 ถูกอ่านแยกเป็นสระ “ู”
2. `AsyncBrailleWorker._worker_loop()` ประกอบข้อความเสร็จ แต่ตั้ง `status='uncertain'` เพราะมี warning และยังส่งผลพร้อม `result_id` ต่อไปตามปกติ
3. `RealTimeBrailleScanner._render_frame()` ล้าง history เมื่อผล uncertain ข้อความเดียวกันจึงไม่เคยสะสมครบ 6 ผล แม้ Grid จะดูนิ่ง
4. UI รวม warning ที่ไม่ใช่ `empty_crop` / `ambiguous_row_grid` เป็น `other: 1` โดยไม่บอกเซลล์หรือสาเหตุ

`tokenize_thai()` เพิ่ม index ทุกครั้งแม้พบรหัสไม่รู้จัก จึงไม่ได้วนรอไม่สิ้นสุด และ worker ไม่มีเงื่อนไขหยุด thread เมื่อเกิด warning ชนิดนี้ การ restart กล้องหรือข้ามคำเตือนทั้งหมดไม่แก้ mapping ที่ขาด

## ไฟล์ที่แก้

| ไฟล์ | การเปลี่ยนแปลง |
|---|---|
| `config_thai.py` | เพิ่ม `THAI_READING_MULTI_CELL` ครอบคลุมคู่ 456 + 2/23/25/256/235/236 → `, ; : . ! ?` แยกจาก vocabulary ของ synthetic เดิม |
| `thai_decoder.py` | ให้ `tokenize_thai()` ใช้ตารางอ่านใหม่ ยังคงตรวจ adjacency, line break และ crop warning ก่อนรวมเซลล์ |
| `yolo_cell_stream.py` | ให้ `pair_markers()` ใช้ตารางเดียวกับ tokenizer เซลล์ต่อท้ายเครื่องหมายถูกระบุเป็น continuation โดยไม่เปลี่ยน dot pattern |
| `camera_reader.py` | แสดง warning พร้อมตำแหน่ง เช่น `C12 [456]: unknown/incomplete symbol` แทน `other: 1`; ถ้าประกอบข้อความไม่ครบแต่ไม่มี cell warning ให้บอก `text assembly incomplete` |
| `tests/test_punctuation_recovery.py` | เพิ่ม 8 tests สำหรับวรรคตอน การจับคู่ การไม่ข้ามบรรทัด/ช่องว่าง และการฟื้นจาก uncertain จน UI ยืนยันครบ |
| `tests/fixtures/punctuation_cells.json` | เก็บตำแหน่งและรูปแบบจุดของตัวอย่าง 22 เซลล์สำหรับ regression โดยไม่ผูก tests กับโฟลเดอร์ภาพในเครื่อง |

ไม่บังคับผล uncertain ให้เป็น ok และไม่เติมจุดที่หาย เช่น `[5,6]` จะไม่ถูกเดาเป็น `[4,5,6]` ข้อความยังต้องยืนยันครบ 6 ผล inference ใหม่ตามเดิม

## ผลทดสอบ

- Unit/regression **146 tests ผ่าน**
- YOLO regression ด้วย weights จริง **99/99 ผ่าน**
- Replay diagnostic 10 เฟรม: แปลจากเซลล์ที่บันทึกไว้สำเร็จ **10/10** หลังแก้
- อ่านใหม่จากภาพดิบแบบเริ่ม reader ใหม่ต่อภาพ: **9/10** สำเร็จ จำนวนเซลล์และ dot pattern ก่อน/หลังตรงกันทุกภาพ และภาพที่อ่านได้ก่อนแก้ไม่มีข้อความถดถอย
- ภาพที่ยังไม่ผ่านคือเฟรม 10528 ซึ่ง cold detection ได้ `[5,6]` ที่ C23 จึงยังแจ้ง unknown/incomplete ตามจริง เป็นปัญหาตรวจจุดไม่ครบ ไม่ใช่ mapping colon ที่สมบูรณ์
- ทดสอบ worker จริงให้เผยแพร่ผล uncertain 3 ครั้ง แล้วส่งผลที่ถูกต้อง 6 ครั้งต่อเนื่อง: thread เดิมทำงานต่อ, history เพิ่มเฉพาะผลใหม่, การวาดผลเดิมซ้ำไม่เพิ่มคะแนนยืนยัน และข้อความแสดงเมื่อครบ 6 ผล
- Replay ภาพดิบของตัวอย่าง 22 เซลล์ผ่าน detector → worker → decoder → tracker → UI จริง 6 ครั้ง: สถานะ ok ทั้ง 6 ครั้ง, history 1–6, thread เดิม, แสดง **คำที่มีใช้: ฐาน ฐิติ** ภาพผลอยู่ที่ `scratch/decoder_punctuation_review/confirmed_scan.png`
- `git diff --check` ผ่าน

การทดสอบสุดท้ายเป็นการอ่านภาพที่บันทึกไว้ซ้ำเพื่อทดสอบการยืนยันผล ไม่ใช่การเปิดกล้องสดหรือการวัด FPS หลังแก้ ตัวเลข FPS บนภาพตัวอย่างมาจาก loop replay จึงใช้บอก FPS ของกล้องไม่ได้ ยังต้องเปิดโปรแกรมกล้องใหม่เพื่อโหลดโค้ด และลองสแกนหน้าหนังสือจริงอีกครั้ง

คำสั่งตรวจซ้ำ:

```powershell
.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -q
.venv\Scripts\python.exe -B -X utf8 tools/diagnostics/benchmark_yolo_stream.py --output scratch/decoder_punctuation_review/yolo_regression.json
.venv\Scripts\python.exe camera_reader.py --lang thai --res 4k --sharp 0 --dot-color blue
```

รายงานก่อน/หลังและ trace การยืนยันผลอยู่ใน `scratch/decoder_punctuation_review/` ไม่มีการแก้ weights หรือข้อมูล annotation ที่เตรียมไว้
