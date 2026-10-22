# English Braille audit — diag_063

ตรวจบน branch เดิม `feature/single-bat-entrypoint` วันที่ 2026-10-15

## ข้อสรุปและหลักฐาน

- เป็น bug ที่ English mapping ไม่ใช่ปุ่มสลับภาษาหรือ config เลือกภาษาไม่ถึง decoder
- `config.py` ระบุ Grade 1 ตั้งแต่เดิม จึงรักษา Grade 1 ไม่เพิ่ม contractions/Grade 2
- C8 บรรทัดแรกมี `dots`, `overview_dots`, `read_pattern` เป็น `[2]` ตรงกัน, `crop_status=read`, `crop_disagreement=false`, `read_source=color` ภาพ contact sheet เห็นจุดเดียวตำแหน่ง 2 ซึ่งเป็น comma
- mapping เดิมไม่มี `{2}` และใส่ comma ที่ `{6}` แต่ decoder จับ `{6}` เป็น capital indicator ก่อน lookup จึงอ่าน comma ไม่ได้และรายงาน `unknown_pattern`
- mapping เครื่องหมาย `!` เดิมเป็น `{2,6}` แทน `{2,3,5}` และไม่มี `; : ' -` แก้เฉพาะตาราง English Grade 1 และ reverse mapping ที่สร้างจากตารางเดียวกัน
- บรรทัดสองมี `a` หนึ่งเซลล์และ `l` หนึ่งเซลล์ก่อนช่องว่างจริงใน diagnostic ผล `al` จึงไม่ได้เกิดจาก parser ยุบ `ll` หลักฐานไม่พอแยกว่าต้นฉบับพิมพ์ขาดหรือขั้น detection ตกเซลล์ ห้ามเติมอักษรจากการเดาประโยค

## เส้นทางที่ audit

1. `scanner_ui.py` สร้างปุ่มภาษา key `l` จาก state `lang`; mouse hit ส่ง key เดียวกับ keyboard เข้า `_handle_key` ใน `camera_reader.py`
2. `_handle_key` สลับ `self.lang` และ reset confirmation; render ส่ง `lang` เข้า worker และใส่ไว้ใน context ตรวจผลลัพธ์เก่าคนละภาษาออกก่อนแสดง
3. `AsyncBrailleWorker.submit_frame` เก็บภาษาไว้กับ frame; worker ส่ง `frame_lang` เข้า detector, `decode_cells` และ `decode_cells_verbose` แล้ว publish `result_lang`
4. `YOLOBrailleDetector.detect` ส่งภาษาไป `ColoredCellStream.detect`; diag นี้ใช้ COLOR BLUE, ROI off ไม่ใช่การอนุมานจุดด้วยโมเดล YOLO แม้ชื่อ detector mode ยังเป็น yolo
5. color components → `plan_cells` → grid tracker → `read_cell` → `pair_markers(lang)`; Thai-specific marker merging มีเงื่อนไขภาษา จึงไม่ควรเปลี่ยนเพื่อแก้ English mapping
6. `read_cell` สร้าง pattern 1–6 ตาม slot; `_english_tokens` ตรวจ geometry/crop warning ก่อน lookup และใช้ boundary จากช่องว่าง/geometry/line_id
7. decoder รองรับ capital indicator เดี่ยว, numeric indicator, a–j เป็นเลข และ comma/period ใน number mode; verbose warning ส่งเข้า `reading_lines` และ `LineConfirmation` บรรทัดที่มี warning ไม่ถูกยืนยัน
8. UI ใช้ line confirmation แสดงเฉพาะบรรทัดที่ยืนยันแล้ว ส่วนบรรทัดแรกขึ้น review ดังภาพ การมี 55 cells/147 dots ไม่ใช่หลักประกันว่าทุก pattern แปลงได้

## ก่อนและหลัง

รัน inference_input.png ด้วย COLOR BLUE และ replay cells ใน manifest:

| สิ่งที่ตรวจ | ก่อน | หลัง |
| --- | --- | --- |
| cells / dots / lines | 55 / 147 / 3 | 55 / 147 / 3 |
| pattern ทั้ง 55 cells | baseline | เหมือนเดิมทั้งหมด |
| บรรทัดแรก | One day[2] Jupiter called | One day, Jupiter called |
| บรรทัดสอง | in al the animals in | เหมือนเดิม |
| บรรทัดสาม | heaven for a meeting | เหมือนเดิม |
| warning | C8 unknown_pattern | ไม่มี |
| line status | uncertain / ok / ok | ok / ok / ok |

ทดสอบ English ใหม่ 7 tests กับ mapping จาก HEAD ก่อนแก้: 8 failures จาก subtests, 0 errors หลังแก้ผ่านทั้งหมด
baseline ชุดเดิมที่เลือกก่อนแก้ผ่าน 50 tests; หลังแก้ชุดที่เกี่ยวข้องรวม English/UI/worker/confirmation และ Thai parser/composition/raw encoding/multicell ผ่าน 94 tests
fixture คัดเฉพาะ dots, line_id, reading_x, cell_pitch, adjacency_limit จาก diag_063 เพื่อเก็บ regression ได้โดยไม่พึ่ง output ที่อาจถูกลบ
fixture ทดสอบว่า 3 บรรทัดยืนยันได้เมื่อได้ผลเหมือนกันครบจำนวนที่กำหนด ไม่ใช่หลักฐานการยืนยันจากกล้องสด

## ไฟล์ที่แก้

- `config.py`: ตาราง punctuation ภาษาอังกฤษเท่านั้น
- `tests/test_english_braille.py`: alphabet, punctuation/reverse mapping, capital/number, spaces/line breaks, incomplete/unknown cells, diagnostic replay, คลิกปุ่มภาษาจริงผ่าน worker ไป decoder แล้วกลับไทย พร้อม reset history
- `tests/fixtures/english_diag_063.json`: cell geometry และ patterns จาก diagnostic เดิม
- เอกสารนี้

## ทดสอบด้วยภาพเดิม

จากโฟลเดอร์โปรเจกต์ รัน `.venv/Scripts/python.exe -X utf8 main.py output/diag_063/inference_input.png --lang english --dot-color blue --roi off --no-display --save`
ดูภาพผล `output/inference_input_yolo.png` เทียบต้นฉบับ ควรได้ comma หลัง day และยังคง al ตามเซลล์ที่มี
สำหรับกล้องสดเลือก Blue / English [L] ใช้ภาพเดิมและถือให้ grid นิ่งจนได้ผลซ้ำครบ threshold จึงคาดหวังการยืนยัน 3/3; งานนี้ไม่ได้ทดสอบฮาร์ดแวร์สด

## ข้อจำกัดที่ยังเหลือ

- ไม่อ้างว่าความแม่นยำ detection ดีขึ้น: patterns และจำนวนจุดไม่เปลี่ยน และไม่มี ground truth ที่รับรองทุกเซลล์
- ไม่เดาหรือแก้ `al` เป็น `all`; ต้องมีภาพต้นฉบับ/ground truth ชัดเจนเพื่อ audit การตกเซลล์ต่อ
- Grade 2, capital word/passage แบบหลาย indicator และ indicator ค้างท้ายไม่อยู่ในขอบเขตการแก้นี้; decoder เดิมรองรับตัวพิมพ์ใหญ่ถัดไปหนึ่งอักษร ยังไม่ควรอ้างว่ารองรับ UEB ทั้งมาตรฐาน
- pattern ที่จุดหายแล้วบังเอิญตรงตัวอักษรอื่นยังตรวจจาก mapping อย่างเดียวไม่ได้; เซลล์ที่มี crop/grid warning ยังคงเป็น replacement character และต้อง review
- ไม่แก้ Thai parser, detection, geometry, threshold หรือใส่การแก้ประโยคเฉพาะภาพ ไม่รันทุก test ใน repository และไม่ทดสอบกล้อง/เสียงจริง
