# แก้ spacing ของจุดพิมพ์ sparse ตาม Diagnostic 060–061

## ต้นเหตุที่ยืนยันแล้ว

`diag_061` ตรงกับ failure case: 84 cells, 199 dots, 5 lines, unknown 15 ตำแหน่ง และบรรทัดที่ผ่านเกณฑ์อ่าน 2/5

เส้นทางจริงคือ `ColoredCellStream.find_dots()` → `plan_cells()` → `read_cell()` → `pair_markers()` → Thai decoder → `reading_lines()`/`LineConfirmation`

- color components 199 ก้อน และ dot centers 199 ตำแหน่งไม่ซ้ำ
- source-dot ownership 199 ครั้งไม่ซ้ำ ไม่พบ dot หายระหว่าง detection กับ assignment
- geometry เดิมผ่าน six-slot/overlap validation; ไม่มี shared dot หรือ Grid overlap ที่ validator ตรวจพบ
- ปัญหาแรกเกิดที่ spacing ก่อน column grouping ไม่ใช่ threshold สีหรือกฎภาษาไทย
- บรรทัดแรกสองบรรทัดมี pattern sparse ที่จุดใกล้สุดอยู่แนวทแยง: nearest Euclidean pitch ประมาณ 57.54 และ 56.63 px แต่ระยะแถวที่วัดจากภาพประมาณ 42 px
- คอลัมน์ของ cell แรกอยู่ประมาณ x=490 และ x=534; จุดของ cell ถัดไปอยู่ประมาณ x=604 การใช้ spacing ที่โตทำให้ `_column_groups()` เลือกคู่หลังแทนคู่จริง แล้ว fit ความกว้าง Grid ต่อจากคู่ที่ผิด
- บรรทัด 3–5 และภาพพยัญชนะ `diag_060` มีคู่แนวตรงเป็นหลัก จึงได้ spacing ประมาณ 41–43 px

Grid ที่ครบและไม่ซ้อนยังอาจมี phase/grouping ผิดได้ จึงไม่ใช้เพียงผล validation เป็นหลักฐานว่าอ่านถูก

## การเปลี่ยน

เปลี่ยนเฉพาะ `yolo_cell_stream.py`

- `_spacing(..., axis_angle=None)` คง Euclidean behavior เดิมเมื่อไม่ระบุมุม เพื่อไม่เปลี่ยนการกรองสีและผู้เรียกเดิม
- เมื่อระบุมุม วัดคู่จุดในแกน Grid ด้วย `max(abs(dx), abs(dy))`; คู่ทแยงหนึ่งแถว/หนึ่งคอลัมน์ไม่ขยาย spacing เป็นความยาวทแยง
- การตัด duplicate centers ภายในขนาด dot ยังคงใช้ระยะ Euclidean จริง
- `plan_cells()` ใช้ระยะตามแกนที่จัดแนวแล้ว ทั้งกรณีบรรทัดเดียวและแถบข้อความที่แยกชัดเจน โดยแต่ละแถบยังใช้มุม/scale ของตนเอง
- คงการ fit ระยะคอลัมน์จากคู่ที่ตรวจพบ ไม่บังคับให้ความกว้างคอลัมน์เท่าระยะแถว
- ไม่เพิ่ม threshold แบบสุ่ม ไม่เปลี่ยนสี, parser, decoder, confirmation หรือ tracking rules

## ผลก่อน–หลัง

| รายการ | ก่อน | หลัง |
|---|---:|---:|
| diag_061 cells | 84 | 76 |
| dots | 199 | 199 |
| lines | 5 | 5 |
| unknown/incomplete tokens | 15 | 1 |
| บรรทัดที่ผ่านเกณฑ์อ่าน | 2/5 | 4/5 |
| diag_060 | 60 cells / 187 dots / 4 lines | เหมือนเดิมทุก pattern/ข้อความ |

- replay แต่ละภาพ 6 ครั้งใน camera-context tracking ให้ pattern เหมือนกันทุกครั้ง
- จำลองผล inference 6 ครั้งที่มี result_id ต่างกัน: confirmation 4/4 สำหรับ diag_060 และ 4/5 สำหรับ diag_061
- เปรียบเทียบกับ source ก่อนแก้โดย replay ภาพทั้ง 61 Diagnostics: 60 ชุดให้จำนวน cell/dot/line, pattern, warning และข้อความเดิม ชุดที่เปลี่ยนมีเพียง diag_061
- median detection แบบ offline ของ diag_061 หลัง warm-up และวัด 7 ครั้ง: 63.15 → 64.40 ms เป็นการวัดครั้งเดียวบนเครื่องนี้ ไม่ใช่ FPS กล้องสด

## Tests

- ก่อนแก้: full suite เดิม 369 tests ผ่าน
- เขียน test ก่อนแก้จริงและยืนยันว่า fail ทั้งพิกัด Diagnostic และ sparse diagonal synthetic page
- หลังแก้: tests เฉพาะส่วน 45 tests ผ่าน; full suite 371 tests ผ่าน ไม่มี failure/error/skip
- test ใหม่ใช้ fixture พิกัดจากภาพจริง พร้อมผล pattern ที่ตรวจช่วงสองบรรทัดแรก และคงผลบรรทัดที่อ่านได้
- ภาพจำลองใช้ 1, 2, 3, 5 lines, scale 0.35/0.7/1, มุม -3/0/3 องศา และ BLUE/RED
- ตรวจ Grid ครบ 6 slots, ownership หนึ่ง dot ต่อหนึ่ง cell, ไม่มี overlap และการประมวลผลซ้ำคงที่
- tests เดิมที่ row/column spacing ต่างกันยังผ่าน
- fixture เก็บเฉพาะ coordinates/area ไม่พึ่ง output/ และไม่แก้ Diagnostic ต้นทาง
- `git diff --check` ผ่าน

## สิ่งที่ยังเหลือ

- diag_061 บรรทัด 5 ยังมีหนึ่ง unknown ที่ cell 72 pattern 456 (หลังจำนวน cell เปลี่ยน) ไม่ได้แก้ด้วยการเดาหรือเพิ่ม mapping
- ช่วงนั้นมีคอลัมน์ sparse และ anchor คู่ครบเพียงหนึ่งคู่ ทำให้การตัดสินว่าเป็นคอลัมน์ซ้าย/ขวายังต้องตรวจแยกจาก bug spacing นี้
- ยังต้องเทียบ raw encoding/ตำแหน่งจากไฟล์กระดาษต้นฉบับของบรรทัด 5 ก่อนแก้ phase อย่างมั่นใจ
- ยังไม่ได้ทดสอบกล้องสด การเปลี่ยนหน้า แสง/โฟกัสเพิ่มเติม หรือ accuracy จาก ground truth ครบทุก cell
- ผล 4/5 เป็น replay ของภาพเดิม ไม่ใช่การรับรองว่าแก้ทั้งหน้าครบแล้ว
