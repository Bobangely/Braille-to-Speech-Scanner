# Thai Braille Reader — YOLO

รีวิวก่อน merge: [บั๊กที่แก้ การเก็บโค้ด ผลตรวจ และข้อจำกัด](docs/PRE_MERGE_REVIEW_TH.md)

branch `codex/yolo-cell-stream` ใช้ **YOLO เป็นตัวตรวจจับเดียว**: ตรวจจุด → crop เซลล์ → YOLO อ่าน crop → จับ marker → ประกอบข้อความไทย/อังกฤษ

ถอด CV/Hybrid และ fallback ตรวจสีออกจากเส้นทางใช้งานแล้ว `cv2` ยังใช้เปิดกล้องและจัดการภาพ โค้ด CV เก่าอยู่ใน `archive/legacy_cv/` เพื่ออ้างอิงเท่านั้น

## งานรอบปัจจุบัน

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

เก็บโมเดลทดลองใน `runs/detect/<ชื่อรอบ>/weights/` ไม่แทนที่ `models/braille_yolo.pt` อัตโนมัติ ใช้ `--weights` เพื่อฝึกต่อเป็นรอบใหม่ และ `--resume <last.pt>` เฉพาะรอบที่ถูกขัดจังหวะ

## ใช้งานกล้องและภาพเดี่ยว

เปิด `Start_Scanner_YOLO.bat` หรือ:

```powershell
.venv\Scripts\python.exe camera_reader.py --lang thai --res fhd --sharp 0
.venv\Scripts\python.exe yolo_detector.py path/to/image.png --lang thai --save --dump-stream
```

ระบุ `--model <ไฟล์ best.pt>` เพื่อทดลองโมเดลใหม่ ปุ่ม `P` ในกล้องเก็บภาพดิบ `*_raw.png` ควบคู่ภาพผลลัพธ์สำหรับตรวจข้อมูลภายหลัง

## ตรวจสอบ

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe tools\training\validate_dataset.py datasets/thai_synthetic/seed_20260908
.venv\Scripts\python.exe tools\diagnostics\benchmark_yolo_stream.py
```

ผ่าน 46 tests และ YOLO weights หลักยังอ่านชุดสังเคราะห์มาตรฐานได้ 99/99 ภาพหลังรีวิวก่อน merge ทดลองฝึกสั้น 1 epoch บนชุดเล็กแล้ว ส่วนชุดหลักยังไม่ได้ฝึกเต็มรอบ

คะแนนนี้ยังไม่ยืนยันความแม่นยำกับหนังสือจริงหรือ FPS บน Redxa ระบบเสียงในหน้ากล้องยังปิดไว้ตามโค้ดเดิม

[การทำงานของ cell stream และข้อจำกัด](docs/YOLO_CELL_STREAM.md) · [ประวัติโค้ด CV ที่เลิกใช้](archive/legacy_cv/README.md)
