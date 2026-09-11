# ผลวิเคราะห์อักษรไทยอ่านไม่ออกและเครื่องหมาย �

วันที่ 10 กันยายน 2026

## ข้อสรุป

เครื่องหมายรูปสี่เหลี่ยมข้าวหลามตัดที่เห็นในภาพตรงกับ `�` (U+FFFD)
ซึ่ง decoder ของโปรเจกต์สร้างขึ้นเมื่ออ่านเซลล์ไม่ได้หรือระบุตำแหน่งแถวไม่แน่นอน
ตัว `?` (U+003F) เป็นคนละอักขระและยังเป็นเครื่องหมายคำถามภาษาอังกฤษที่ถูกต้องได้

ยืนยันจากค่าก่อนวาดหน้าจอ, การทดสอบ UTF-8, ฟอนต์จริงบนเครื่อง และเส้นทางเรียก Pillow แล้ว
การตรวจครั้งนี้พบข้อจำกัดที่การตรวจจุดในภาพนูนจางและการอ่าน crop
**ยังไม่ได้แก้ให้โมเดลอ่านหนังสือที่เบลอได้แม่นยำขึ้น** การแก้โค้ดครั้งนี้เพิ่มหลักฐานรายเฟรม/รายเซลล์
เพื่อระบุจุดเสียให้ชัดเจน และรักษาผลอ่านที่ไม่แน่นอนไว้ตามจริง

## ไฟล์และเส้นทางที่อ่านก่อนแก้

ตรวจ implementation/call sites ร่วมกันใน camera_reader.py, main.py, yolo_detector.py,
yolo_cell_stream.py, dot_fusion.py, decoder.py, thai_decoder.py, config.py, config_thai.py,
live_preview.py, tts.py, tools/diagnostics/export_yolo_stream.py,
เครื่องมือสร้างและเทรน dataset, รายงาน/metadata โมเดล และ tests ที่เกี่ยวข้อง
พร้อมเทียบ pipeline เดิมจาก Git และ archive/legacy_cv/detector.py

```text
Camera BGR frame
  → crop ROI (ยังเป็นพิกเซลต้นฉบับ)
  → optional sharpening
  → YOLO overview / tiles หาจุด
  → plan_cells ระบุกริดและเสนอเซลล์
  → crop_cell สร้างภาพ 320×320
  → YOLO อ่านจุดใน crop
  → read_cell จับคู่จุดเข้ากับ 6 ตำแหน่ง
  → pair_markers / Thai decoder / ประกอบสระและวรรณยุกต์
  → Python Unicode string
  → LivePreview ปรับพิกัดเพื่อแสดงผล
  → Pillow วาดป้าย C1/C2/... และข้อความไทย
  → แปลงสี RGB↔BGR เพื่อแสดงภาพ
```

การแปลง RGB/BGR เป็นการแปลงพิกเซล ไม่ได้แปลง encoding ของข้อความ
การตั้ง stdout เป็น UTF-8 ใช้กับ console ส่วนป้ายภาษาไทยส่ง string เข้า Pillow โดยตรง

## หลักฐานของเครื่องหมายที่เกิดขึ้น

| อินพุตให้ decoder | ข้อความที่ได้ก่อนวาด | Unicode | เหตุผล |
|---|---|---|---|
| crop อ่านไม่ได้ ไม่มีจุด | � | U+FFFD | empty_crop |
| มีจุด แต่แถวของกริดกำกวม | � | U+FFFD | ambiguous_row_grid |
| จุด 1245 ที่กริดแน่นอน | ก | U+0E01 | อ่านได้ |
| สองเซลล์ 6 + 13456 | ญ | U+0E0D | รวม marker กับเซลล์ถัดไป |
| เครื่องหมายคำถามอังกฤษ | ? | U+003F | เป็นเครื่องหมายที่มีความหมาย |

เส้นทางที่สร้าง U+FFFD ได้แก่ thai_decoder.tokenize_thai() และ decode_thai()
โดยกรณี crop ว่างมาจาก yolo_cell_stream.read_cell(): ไม่มีจุดที่รับเข้าตำแหน่งกริดได้
การมีกรอบสีฟ้าไม่ได้แปลว่าขั้นอ่าน crop อ่านสำเร็จ เพราะกรอบมาจากการเสนอเซลล์ในขั้นก่อนหน้า

ฟอนต์ที่โหลดจริงคือ C:/Windows/Fonts/tahomabd.ttf ตรวจตาราง glyph แล้วพบ ก ญ ศ ภ ธ,
U+003F และ U+FFFD ครบ การ round-trip UTF-8 รักษาค่าเดิม
ป้ายที่ส่งเข้า Pillow ใน regression test เป็น C1: � อยู่แล้ว ไม่มีการเปลี่ยนเป็น ? ระหว่างวาด

ภาพ screenshot ไม่ได้มีผล JSON ของ session เดิม จึงไม่สามารถจับคู่ทุก C-number ในภาพกับ
frame ID และค่าภายในเดิมย้อนหลังได้ ข้อสรุปข้างต้นมาจากเส้นทางโค้ดและการจำลองที่ตรวจค่าได้
ไม่ใช่การเดาสาเหตุจากลักษณะ glyph เพียงอย่างเดียว

## ทดสอบส่วนที่ไม่มี overlay ในภาพแนบ

ใช้ ROI พิกัด (270, 635) ถึง (1410, 765) จาก screenshot ขนาด 1675×818
ส่วนนี้มีจุดนูนจาง แต่ไม่มีกรอบ/ตัวอักษร overlay ทับบริเวณที่นำมาวิเคราะห์
อย่างไรก็ตาม ยังเป็นภาพจาก screenshot ไม่ใช่ภาพดิบจาก sensor และไม่มีคำตอบอ้างอิงที่กำกับไว้

| การทดลองด้วยโมเดลใช้งาน | Candidate จุดในภาพรวม | เซลล์ที่เสนอ | จุดที่อ่านได้จาก crop | ผล |
|---|---:|---:|---:|---|
| ROI เดิม | 0 | 0 | 0 | ไม่มีข้อความ |
| เพิ่ม contrast แบบ CLAHE เพื่อวิเคราะห์เท่านั้น | 5 | 3 | 0 | ��� / empty_crop 3 เซลล์ |

ตรวจรายงานของทั้งสาม crop แล้ว YOLO ภายใน crop คืน **0 detections**
จึงแยกได้ว่ากรณีทดลองนี้ล้มที่การตรวจจุดใน crop ไม่ใช่ตรวจพบจุดแล้วถูกตัดทิ้งเพราะอยู่นอกกริด
จำนวน candidate 5 จุดไม่ได้ยืนยันว่าเป็นจุดจริงทั้งหมด เพราะยังไม่มี ground truth

ผลนี้ชี้ว่าการเพิ่ม contrast อย่างเดียวไม่ทำให้ pipeline อ่านได้ครบ
ยังไม่มีการเปิด CLAHE ใน production หรือปรับ threshold เพื่อบังคับให้ได้ข้อความ

## เกี่ยวกับการตัด CV หรือไม่

เกี่ยวข้องในแง่ที่เส้นทางตรวจจับเปลี่ยน:

- ก่อนตัด ใน Git commit 0cefa72 โหมด hybrid เรียก CV ตรวจจุดร่วมกับ YOLO และใช้ CV fallback ได้
  ส่วน YOLO แบบเดิมบางเส้นทางใช้ตัวจัดกลุ่มเซลล์ของ CV
- ปัจจุบัน YOLO stream ใช้ YOLO ทั้งภาพรวมและ crop รายเซลล์
  จุดที่อ่านไม่ได้ใน crop จะแสดงความไม่แน่นอน ไม่มีข้อมูลจาก CV มาเสริม
- OpenCV library ยังใช้สำหรับรับภาพ, crop/warp, resize, sharpening และวาดกริด
  สิ่งที่ตัดออกคือ pipeline ตรวจจุดแบบ CV/สีและ fallback

จึงมีเหตุผลที่อาการอ่านไม่ออกจะเห็นชัดขึ้นหลังเปลี่ยน pipeline แต่ข้อมูลที่มี
ยังพิสูจน์ไม่ได้ว่า CV เดิมจะอ่านหนังสือหน้านี้ถูกต้อง การเปิด CV กลับหรือเปลี่ยน mapping ไทยทันที
ไม่ได้มีหลักฐานรองรับว่าแก้ข้อจำกัดของ YOLO ในภาพนี้

## โมเดลและ Dataset ที่ใช้งานจริง

โมเดลเริ่มต้นคือ models/braille_yolo.pt; SHA-256:

```text
41d14e6550301cd6ee0b831f18b7ab75a023c5f50109fc33af67b6d3159929be
```

metadata ใน checkpoint ระบุ datasets/braille_dots/data.yaml, imgsz 640, epochs 30
ตรวจ candidate ที่มีอยู่เพิ่มเติมกับ ROI เดียวกัน:

| โมเดล | ข้อมูลใน checkpoint | Candidate ใน ROI เดิม |
|---|---|---:|
| models/braille_yolo.pt | braille_dots, 640, 30 epochs | 0 |
| runs/detect/runs/detect/train/weights/best.pt | braille_dots, 416, 15 epochs | 0 |
| runs/detect/synthetic_smoke_20260908/weights/best.pt | dataset_generator_check, 320, 1 epoch | 0 |

training_report.json ของ synthetic_smoke ระบุ production_weights_replaced=false
จึงไม่ควรถือว่าการสร้าง dataset ใหม่หมายถึงกล้องใช้โมเดลที่เทรนชุดนั้นแล้ว

generator สังเคราะห์ปัจจุบันวาดจุดสีทึบบนพื้นสว่างและ blur เบา (sigma 0.1–0.65)
ยังไม่มีการจำลองจุดนูนสีเดียวกับกระดาษพร้อมแสง/เงาแบบหนังสือในภาพ
นี่เป็นช่องว่างของข้อมูลที่ต้องประเมินเพิ่ม ไม่ใช่หลักฐานว่ามีสาเหตุจาก training เพียงอย่างเดียว

## สิ่งที่แก้ในรอบนี้

- yolo_cell_stream.py: เพิ่ม crop_diagnostics แยกจำนวน detections, matched_slots,
  outside_grid และ duplicate_slots โดยไม่เปลี่ยนการเลือกจุด
- camera_reader.py: แสดงจำนวน empty crops/row-grid warnings และเพิ่มปุ่ม D
  เก็บภาพ ROI ก่อน sharpening กับภาพที่ส่งเข้า YOLO คู่กับผล/ภาษา/frame ID ชุดเดียวกัน
  เก็บเฉพาะผลล่าสุด และคืน reference ของภาพเมื่อ stop
- tools/diagnostics/export_yolo_stream.py: บันทึกภาพ, crop แต่ละเซลล์, เหตุผล,
  codepoints, UTF-8 hex, โมเดล/พารามิเตอร์/ฟอนต์ และ hash ของ input
- yolo_detector.py: --dump-stream ใส่ข้อมูลโมเดลและฟอนต์ในรายงานด้วย
- tests/test_recognition_trace.py และ tests/test_camera_recovery.py:
  ตรวจ Unicode, rendering, การจับคู่เฟรม และกรณี diagnostic export ล้มเหลว

ยังคง U+FFFD และเครื่องหมายคำถามที่ถูกต้องไว้ ไม่มีการเดาตัวอักษร, เปลี่ยน mapping,
สลับ production weights หรือเทรนโมเดลใหม่ในรอบตรวจนี้

## วิธีตรวจครั้งถัดไป

เปิดกล้องด้วยคำสั่งเดิม แล้วกด **D** ขณะที่พบอาการ โปรแกรมบันทึกลง output/diagnostic_<เวลา>/

- camera_roi.png: ภาพ ROI ก่อน sharpening
- inference_input.png: ภาพเฟรมเดียวกับที่ YOLO อ่าน
- line_..._cell_....png และ contact_sheet.png: crop ที่ใช้ตรวจแต่ละเซลล์
- manifest.json: ผลอ่าน, เหตุผล, Unicode, frame ID, ภาษา, preprocessing และโมเดลที่ใช้จริง

ภาพ preview อาจใหม่กว่าผล AI ปุ่ม D จึงใช้ภาพจากผล worker โดยตรง ไม่ใช้ภาพสดมาจับคู่กับผลเก่า
ไฟล์ที่บันทึกทั้งหมดอยู่ในเครื่อง

## ผลตรวจและงานที่ยังเหลือ

- ชุดทดสอบทั้งหมด **72 tests ผ่าน**
- โมเดลจริงกับ synthetic reference fixtures **99/99 ผ่าน** เป็น regression ไม่ใช่ความแม่นยำหนังสือจริง
- ทดสอบ export ด้วยโมเดลจริงผ่าน และเปิดตรวจภาพ ROI/contact sheet แล้ว
- การรันทดสอบที่ถูก usage limit ปฏิเสธก่อนหน้าได้รันซ้ำสำเร็จแล้ว
- ยังต้องเก็บภาพดิบหนังสือจริงพร้อมคำตอบที่ถูกต้อง แยกชุดฝึก/ทดสอบตามหน้าและ session
  ให้ครอบคลุมจุดนูนไม่แต้มสี, แสง/เงา, ระยะถ่าย, ความเบลอ และสิ่งรบกวนจากขอบ/ห่วงสมุด
  รวมทั้งตัวอย่าง crop 320×320 ตามรูปแบบอินพุตของขั้นอ่านเซลล์
- ควรวัดทั้งการตรวจจุดและอัตราอ่านเซลล์/คำถูกต้อง ก่อนเลือกโมเดลแทน production

หลักฐานดิบอยู่ใน scratch/recognition_diagnosis_20260910/
ได้แก่ summary.json, model_comparison.json และ manifest.json ของแต่ละการทดลอง

