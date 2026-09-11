# Thai Braille Reader — Painted Dots

รีวิวก่อน merge: [บั๊กที่แก้ การเก็บโค้ด ผลตรวจ และข้อจำกัด](docs/PRE_MERGE_REVIEW_TH.md)

เส้นทางกล้องและภาพเดี่ยวอ่าน **เฉพาะจุดที่แต้มสี** โดยค่าเริ่มต้นเป็นสีน้ำเงิน: ตรวจสี → กรอง noise → จัดกริดเซลล์ → ตำแหน่งจุด 1–6 → รวม marker → ข้อความไทย/อังกฤษ

จุดนูนที่ไม่มีสีไม่ถูกนำมาสร้างเซลล์หรือ decode ตัวอ่านสีไม่โหลด/รัน YOLO ส่วนเส้นทาง YOLO เดิมยังเก็บไว้สำหรับประเมินและฝึกโมเดล โดยเรียก `YOLOBrailleDetector(dot_color=None)` อย่างชัดเจน ไม่ใช่ค่าเริ่มต้นของกล้อง

## ตัวอ่านเฉพาะสี

[สาเหตุที่แก้ อัลกอริทึม ข้อจำกัด และผลทดสอบ](docs/COLORED_BRAILLE_ONLY_TH.md)

เลือกสีด้วย `--dot-color blue`, `red`, `green` หรือ `auto` สำหรับหลายสี สีดำ/เทาไม่มี chromatic evidence เพียงพอที่จะแยกจากเงาของกระดาษในเส้นทางนี้

## งาน Dataset / YOLO

สร้าง dataset เบรลล์ไทยสังเคราะห์ **3,000 ภาพ**: train 2,400 / validation 300 / test 300 อยู่ใน `datasets/thai_synthetic/seed_20260908/` พร้อม label จุดและข้อมูลเซลล์/บรรทัด ภาพฝึกไม่มีกรอบ debug ทับ

**[รีวิวภาษาไทย สิ่งที่ทำ ผลตรวจ และวิธีฝึกต่อ](docs/SYNTHETIC_DATASET.md)**

ข้อมูลที่สร้างในเครื่องถูกแยกจาก source code ด้วย `.gitignore` หากเริ่มจาก clone ใหม่ ให้สร้างด้วยคำสั่งนี้:

```powershell
.venv\Scripts\python.exe tools\training\generate_yolo_training.py
```

ตัวสร้างไม่เขียนทับ dataset เดิม การเพิ่มข้อมูลใช้ `--extend-from` และ seed ใหม่ โดยคง validation/test เดิมไว้ ดูตัวอย่างในเอกสารรีวิว

## ฝึก YOLO

```powershell
.venv\Scripts\python.exe tools\training\train_yolo.py --data datasets/thai_synthetic/seed_20260908/data.yaml --epochs 30 --batch 8 --imgsz 640 --device cpu --name thai_synthetic_round1
```

เก็บโมเดลทดลองใน `runs/detect/<ชื่อรอบ>/weights/` โดยไม่เขียนทับ production ใช้ `--weights` สำหรับฝึกรอบใหม่หรือ `--resume` สำหรับงานที่หยุดกลางคัน ประเมิน candidate ด้วย `tools/diagnostics/benchmark_yolo_stream.py --model <best.pt>` ตัวอ่านสีไม่ได้ใช้ weights เหล่านี้

## ใช้งานกล้องและภาพเดี่ยว

เปิด `Start_Scanner_YOLO.bat` หรือ:

```powershell
.venv\Scripts\python.exe camera_reader.py --lang thai --res fhd --sharp 0 --dot-color blue
.venv\Scripts\python.exe yolo_detector.py path/to/image.png --lang thai --dot-color blue --save --dump-stream
```

กด `P` บันทึก snapshot หรือ `D` บันทึกภาพที่ตัวอ่านใช้จริงพร้อม trace รายเซลล์ กล้องยังรองรับ zoom, sharpness และสลับความละเอียด `--dump-stream` สร้างภาพ crop สำหรับตรวจสอบและ JSON โดยระบุว่าสีเป็นแหล่งของจุด

## ตรวจสอบ

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe tools\training\validate_dataset.py datasets/thai_synthetic/seed_20260908
.venv\Scripts\python.exe tools\diagnostics\benchmark_yolo_stream.py
```

ตรวจอัตโนมัติทั้งตัวอ่านสีและเส้นทางเดิม ส่วน benchmark YOLO เป็น regression ของโมเดลและภาพสังเคราะห์ ไม่ใช่ตัวเลขความแม่นยำของหนังสือจริงหรือของตัวอ่านสี

ยังต้องทดสอบกับภาพดิบจากกล้องจริงในสภาพแสงและการแต้มสีที่หลากหลาย พิกัด/รูปแบบสีใน tests ไม่ได้ใช้เป็นค่าตายตัวใน algorithm

[การทำงานของ cell stream และข้อจำกัด](docs/YOLO_CELL_STREAM.md) · [ประวัติโค้ด CV ที่เลิกใช้](archive/legacy_cv/README.md)
