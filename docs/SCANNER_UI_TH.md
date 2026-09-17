# หน้าจอกล้องแบบใหม่

Branch: `feature/ปรับแต่ง-ui-yolov8`

## สิ่งที่เปลี่ยน

- Preview เป็นพื้นที่หลัก วางเฉพาะกรอบเซลล์และจุดบนภาพ ส่วนผลอ่านและสถานะอยู่ในแผงแยก
- แยกแผงผลอ่านที่ยืนยันแล้ว แถบความคืบหน้าการยืนยัน และ Controls
- แสดง Preview FPS, Read rate, Resolution จริง, Mode และ Zoom เป็นช่องแยก
- ใช้พื้นหลังเข้ม สีเน้นสีเดียว และสถานะที่มีทั้งข้อความกับสี:
  Ready / Scanning / Detected / Check image / Error
- หน้าต่างแนวนอนใช้แผงข้าง แนวตั้งย้ายผลอ่านและ Controls ลงด้านล่าง
- ผลภาษาไทยใช้ฟอนต์ Pillow เดิม ตัดบรรทัดตามความกว้างโดยไม่แยกเครื่องหมาย
  สระ/วรรณยุกต์ออกจากตัวอักษรฐาน ข้อความยาวเลื่อนได้ด้วยล้อเมาส์ในแผงผลอ่าน
  หรือปุ่มลูกศรบนแผง / `[` และ `]`
- หน้าหลักแสดงกรอบบาง ชื่อ C1/C2 เหนือกรอบ จุดครบ 1–6 และ pattern ใต้กรอบ
  จุด active ใช้ marker สีสว่าง จุด inactive ใช้วงขอบสีเทาบนพื้นเข้ม
  ตำแหน่ง inactive คือช่องในกริดที่ยังไม่มีจุด active ไม่ใช่จุดที่ตรวจพบ
- กด `H` หรือ Details เพื่อดูกรอบจุดดิบ confidence (ถ้ามี) และตัวอักษรที่ถอดรหัสแล้ว
  โดยชื่อเซลล์และจุด 1–6 ยังคงแสดง กด `D` ยังส่งออก diagnostic เต็มเหมือนเดิม
- ปุ่มบนจอคลิกได้และเรียกผ่าน `_handle_key()` เดิม ไม่สร้างเส้นทางควบคุมกล้องซ้ำ

## การใช้งานที่คงเดิม

`C` น้ำเงิน/แดง, `L` ไทย/อังกฤษ, `V/F` ความละเอียด, `E` ความคม,
`Z/X` หรือ `+/-` ซูม, `R/0` รีเซ็ต, `P` ภาพผลและภาพดิบ,
`D` diagnostic, `Q/Esc` ออก ยังใช้งานได้ รวม alias เดิม `I/O`
ล้อเมาส์บน Preview ใช้ซูม คลิกเพื่อ pan เมื่อซูมแล้ว คลิกกลาง/ดับเบิลคลิกรีเซ็ต
มีการแปลงพิกัดผ่านกรอบ Preview และขอบดำก่อนแมปกลับไปยังภาพกล้อง

ภาพส่งเข้า detector ยังเป็น native camera/ROI pixels ชุดเดิม UI resize หรือ
Details ไม่ส่งเฟรมเดิมไป inference ซ้ำและไม่ล้างประวัติยืนยันข้อความ
การอ่านผลยังต้องตรงกันตามจำนวนที่กำหนดด้วย `--stability` (เริ่มต้น 6)
กล้องและ worker ใช้ lifecycle/error recovery เดิม

`YOLOBrailleDetector.annotate_with_text()` เพิ่มเฉพาะ options การแสดงผล
`details` และ `footer` ที่มี default เป็น True และ `cell_hud` ที่มี default เป็น False
จึงรักษาภาพ export/CLI แบบเดิม ส่วนกล้องเปิด `cell_hud` เป็น True
ไม่เปลี่ยน detection, color threshold, tracking, dot grouping หรือ decoder

## ไฟล์

| ไฟล์ | หน้าที่ |
| --- | --- |
| `scanner_ui.py` | Layout, cached UI, ปุ่ม, ข้อความไทย, scroll, status |
| `camera_reader.py` | ประกอบหน้าจอและส่ง mouse/keyboard ไปคำสั่งเดิม |
| `live_preview.py` | เปิด numbered cell HUD และ invalidate cache เมื่อสลับ Details |
| `yolo_detector.py` | `_draw_cell_hud()` และ options ของ `annotate_with_text()` สำหรับวาด overlay เท่านั้น |
| `tests/test_scanner_ui.py` | Regression ของ UI, พิกัด และภาพเข้า inference |
| `tests/test_cell_hud.py` | เลขจุด/ชื่อเซลล์/pattern, geometry เอียง, หลายเซลล์, resize และขอบภาพ |

ไม่เพิ่ม dependency ใช้ OpenCV + Pillow ที่มีอยู่แล้ว

## Resize และต้นทุนการวาด

ใช้ [OpenCV HighGUI](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html)
อ่านขนาดพื้นที่แสดงผลและรับ mouse/keyboard events ภาพ Preview รักษาสัดส่วน
ด้วยขอบดำ UI canvas จำกัดไม่เกิน 1920×1200 พิกเซลและให้ HighGUI ขยาย
บนหน้าจอ 4K เพื่อจำกัดงานวาด ไม่ลดความละเอียดภาพเข้า detection
ฟอนต์จึงถูกขยายบนจอ 4K ไม่ใช่การวาด UI เต็ม 3840×2160 ทุกเฟรม

UI ส่วนที่คงเดิมใช้ cache ส่วนค่า FPS ปรับการแสดงผลทุก 0.25 วินาที
กริดใช้ cache/tracking เดิม และไม่อัปเดต tracking ซ้ำจากการสลับ Details

ผลรอบแรกที่เปลี่ยน UI เดิมเป็น dashboard (ก่อนเพิ่ม numbered HUD)
ด้วย replay ภาพกล้อง 4K เดียวกัน (40 รอบ สลับลำดับรัน):

| การวาด | ก่อน median / p95 | หลัง median / p95 |
| --- | --- | --- |
| ใช้ overlay เดิม | 6.005 / 7.168 ms | 6.449 / 7.414 ms |
| สร้าง overlay จากผล inference ใหม่ | 29.132 / 34.239 ms | 20.911 / 24.991 ms |

กรณีแรกเพิ่มต้นทุนประมาณ 0.44 ms จากการประกอบ dashboard กรณีที่สองลดงาน
วาดข้อความ/จุดจำนวนมากบน Preview ตัวเลขนี้ไม่รวม camera capture/model inference
และไม่ใช่ FPS กล้องสดหรือ FPS บนบอร์ด ต้องตรวจบนฮาร์ดแวร์ใช้งานจริงต่อ

## ผลตรวจ dashboard รอบแรก

- ผ่าน unit/regression ทั้งชุด **162 tests** รวม UI tests ใหม่ 8 รายการ
- ตรวจสัดส่วน Preview/ขอบเขตปุ่มที่ 640×480, 800×600, 1440×900, 1920×1080,
  3840×2160 และแนวตั้ง 900×1200
- ตรวจภาพ renderer จริงที่ desktop / หน้าต่างเล็ก / แนวตั้ง / 4K
- ตรวจภาษาไทย ข้อความยาว การเลื่อนผล ปุ่มคลิก พิกัดซูมและ pan
- ตรวจว่า UI ไม่แก้ source pixels ไม่ส่ง inference ซ้ำ และไม่ล้าง confirmation
- เปรียบเทียบ AST ของ detector (ยกเว้น annotate), camera capture และ worker
  กับ snapshot ก่อนแก้: ไม่มีการเปลี่ยน logic
- `camera_reader.py --help` และ `git diff --check` ผ่าน

ภาพตัวอย่างและผลวัดอยู่ใน `scratch/ui_review/` (ignored)
ตัวเลข FPS บนภาพตัวอย่างเป็นค่าทดสอบสำหรับจัด layout ไม่ใช่ค่าที่วัดจากกล้องสด
ยังไม่ได้ทดสอบเมาส์/resize ผ่านหน้าต่างกล้องจริงบนจอผู้ใช้

## LIVE Preview HUD: คืนข้อมูลสำคัญจากหน้าจอเดิม

สาเหตุที่ข้อมูลหายคือ dashboard เรียก `annotate_with_text(details=False)`
ซึ่งเดิมข้ามชื่อเซลล์ ช่องจุดว่าง และ pattern ทั้งหมด จึงเพิ่มตัวเลือกวาด HUD
ที่ใช้ `grid.slots` และ `cell.dots` เดิม ไม่คำนวณหรือเดาตำแหน่งจุดจากกรอบใหม่

- ใช้หมายเลข `track_id` เดิม ถ้าไม่มีจึงใช้ลำดับเซลล์ของผลอ่าน
  แสดงทุกเซลล์ทางกายภาพ รวมเซลล์ที่สองของอักษรไทยแบบสองเซลล์
- วาดเลขที่ตำแหน่ง slot จริง จึงรองรับ geometry ที่เอียงตามข้อมูลเดิม
- Pattern ยาวตัดเป็นสองบรรทัดตามระยะห่างจากเซลล์ข้างเคียง โดยคงทุก dot ไว้
  หากชิดขอบล่าง ย้าย pattern ทั้งชุดไปด้านบนเมื่อมีพื้นที่
- ปรับ marker และขนาดข้อความตามขนาดเซลล์บน Preview ใช้ cache เดิม
  เมื่อหน้าต่างแสดงภาพทั้งหน้า เซลล์อาจเล็กจนอ่านเลขยาก ใช้ Zoom เดิมเพื่อขยาย
- ไม่เปลี่ยน source pixels, cell order, tracking, YOLO, color filtering หรือ decoding

ภาพตัวอย่างจาก raw diagnostic และผลวัดของรอบ HUD อยู่ใน `scratch/hud_review/`
เป็นการทดสอบ replay ไม่ใช่การเปิดกล้องสด ตัวเลขบนภาพใช้ตรวจ layout เท่านั้น

ผลทดสอบหลังเพิ่ม HUD: **169 tests ผ่านทั้งหมด** รวม HUD tests ใหม่ 7 รายการ
ทดสอบแยกกลุ่ม HUD/UI/Live Preview/Grid Stability ผ่าน 24 tests
ตรวจ AST เทียบ snapshot ก่อนเพิ่ม HUD: เปลี่ยนเฉพาะฟังก์ชันวาดและข้อความใน UI
camera lifecycle/worker และทุก detection method เหมือนเดิม `git diff --check` ผ่าน

ผลวัดรอบสุดท้ายด้วย raw ภาพ 4K, 35 cells / 2 lines, 40 รอบสลับก่อน–หลัง
วัด `_render_frame()` รวมการวาด UI และ motion overlay เดิม ไม่รวม capture/inference:

| การวาด | ก่อนเพิ่ม HUD median / p95 | หลังเพิ่ม HUD median / p95 |
| --- | --- | --- |
| ใช้ overlay เดิม | 6.811 / 8.067 ms | 6.862 / 8.431 ms |
| สร้าง overlay จากผล inference ใหม่ | 26.867 / 36.854 ms | 28.734 / 41.729 ms |

ต้นทุน median เพิ่มประมาณ 0.05 ms เมื่อใช้ cache และ 1.87 ms เมื่อวาด overlay ใหม่
มีต้นทุนเพิ่มจากเลขจุดและ pattern แต่ไม่วาดใหม่ทุก display frame
ยังยืนยัน FPS กล้องสดไม่ได้ ต้องตรวจบนกล้องและฮาร์ดแวร์ที่ใช้งานจริง
