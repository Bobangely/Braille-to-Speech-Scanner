# Braille-to-Speech Scanner

ต้นแบบอ่านจุดเบรลล์ที่แต้มสีด้วยกล้อง: **YOLO → แยกบรรทัดและ crop ทีละเซลล์ → YOLO อ่าน crop → จับคู่ marker → ประกอบข้อความไทย/อังกฤษ**

branch ปัจจุบัน `codex/yolo-cell-stream` แยกจาก `feature/yolo-detection` เน้น YOLO; weights ที่มีตรวจจุด และใช้เรขาคณิตเสนอกรอบเซลล์

รอบพัฒนาปัจจุบันเน้น algorithm และ dataset บนคอมพิวเตอร์ ยังไม่รับรองประสิทธิภาพบน Redxa Dragon Q6A

## เริ่มใช้งาน

เปิดกล้องด้วย `Start_Scanner_YOLO.bat` หรือรันจากโฟลเดอร์โปรเจกต์:

```powershell
.venv\Scripts\python.exe camera_reader.py --detector yolo --yolo-pipeline stream --lang thai --res fhd --sharp 0
```

ตรวจภาพเดี่ยว:

```powershell
.venv\Scripts\python.exe yolo_detector.py path/to/image.png --mode yolo --lang thai --save --dump-stream
```

โหมด YOLO stream ไม่ใช้การตรวจสี; `--color` ใช้กับ CV/Hybrid ซึ่งยังเปิดเปรียบเทียบได้อย่างชัดเจน ภาพผลลัพธ์และรายงาน crop บันทึกใน `output/` ปุ่ม `P` ในกล้องเก็บภาพดิบ `*_raw.png` ควบคู่ภาพผลลัพธ์

`--tile-size 0` ปิดการตรวจส่วนย่อยเพิ่มเติมหาก AI ช้า หน้าจอใช้ preview กว้างไม่เกิน 1280 พิกเซล ส่วน AI ยังรับภาพต้นฉบับ/ภาพครอปที่ไม่ขยายพิกเซล

## ภาษาไทย: ตารางใหม่กับภาพเก่า

- `--lang thai` ใช้ตารางที่ตรวจเทียบ Liblouis และ World Braille Usage
- ภาพไทยสังเคราะห์เดิมใน `sample_images/` ใช้รหัสสระ/วรรณยุกต์เก่าบางตัว ให้ทดสอบด้วย `--lang thai-legacy` ใน `yolo_detector.py`
- `generate_test.py` สร้างภาพตามตารางใหม่ใน `sample_images/standard/` โดยไม่ทับชุดภาพเก่า
- อย่านำคะแนนจากภาพเก่ามาอ้างเป็นความแม่นยำของเบรลล์มาตรฐานหรือกล้องจริง

## โครงสร้างที่ใช้งาน

| ตำแหน่ง | หน้าที่ |
|---|---|
| `camera_reader.py`, `live_preview.py` | กล้อง งาน AI เบื้องหลัง และหน้าจอ |
| `yolo_detector.py`, `yolo_cell_stream.py` | YOLO, แยกบรรทัด/crop, อ่านเซลล์ และจับคู่ marker |
| `detector.py`, `dot_fusion.py` | CV/Hybrid เดิม และเครื่องมือ tiles/ลบจุดซ้ำ |
| `config_thai.py`, `thai_decoder.py` | รหัสไทยมาตรฐานและการประกอบข้อความ |
| `decoder.py`, `config.py`, `config_thai_legacy.py` | API decoder, อังกฤษ และความเข้ากันได้กับภาพเก่า |
| `main.py`, `tts.py` | อ่านภาพด้วย CV และส่วนสังเคราะห์เสียง |
| `tests/` | ชุดทดสอบอัตโนมัติ |
| `tools/training/` | สร้างข้อมูลและฝึก YOLO |
| `tools/diagnostics/` | ตรวจกล้อง/จุด และวัดเวลาวาดหน้าจอ |
| `models/`, `datasets/`, `sample_images/` | โมเดล ข้อมูลฝึก และภาพต้นฉบับ |
| `output/` | รายงานและภาพผลลัพธ์ |
| `archive/` | README และสคริปต์ตรวจแบบเก่าที่เก็บไว้อ้างอิง |

## ทดสอบ

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe tools\diagnostics\benchmark_yolo_stream.py
.venv\Scripts\python.exe benchmark_detection.py --dataset standard --mode hybrid --output output/standard_thai_benchmark.json
.venv\Scripts\python.exe benchmark_detection.py --dataset legacy --mode hybrid --scales 1 .25 .15 --output output/legacy_benchmark.json
.venv\Scripts\python.exe tools\diagnostics\benchmark_preview.py
```

YOLO stream ผ่าน 99/99 ภาพสังเคราะห์ (33 กรณี × 3 ขนาด) และ 36 unit tests ยังไม่ใช่ผลความแม่นยำจากกล้องจริงหรือความเร็วบนบอร์ด

[การใช้ YOLO cell stream และผลทดสอบ](docs/YOLO_CELL_STREAM.md) · [รายละเอียดภาษาไทยและกล้องรอบก่อน](docs/THAI_AND_CAMERA.md) · [การปรับจุดเล็กรอบก่อน](DETECTION_VALIDATION.md)

โปรเจกต์ใช้ environment เดิม `.venv` โมเดล YOLO ที่ฝึกแล้วอยู่ใน `models/braille_yolo.pt` และยังไม่ได้เทรนใหม่ในรอบนี้ ระบบเสียงในหน้ากล้องยังถูกปิดไว้ตามโค้ดเดิม
