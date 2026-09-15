# Thai Braille Reader — Painted Dots

รีวิวก่อน merge: [บั๊กที่แก้ การเก็บโค้ด ผลตรวจ และข้อจำกัด](docs/PRE_MERGE_REVIEW_TH.md)

เส้นทางกล้องและภาพเดี่ยวอ่าน **เฉพาะจุดที่แต้มสี** โดยค่าเริ่มต้นเป็นสีน้ำเงิน: ตรวจสี → กรอง noise → จัดกริดเซลล์ → ตำแหน่งจุด 1–6 → รวม marker → ข้อความไทย/อังกฤษ

จุดนูนที่ไม่มีสีไม่ถูกนำมาสร้างเซลล์หรือ decode ค่าเริ่มต้น `--roi off` ไม่โหลด/รัน YOLO เปิด `--roi yolo` เพื่อทดลองใช้โมเดลตรวจจุดเสนอกรอบบริเวณข้อความก่อนอ่านสี ส่วนเส้นทาง YOLO เดิมยังเก็บไว้สำหรับประเมินและฝึกโมเดล โดยเรียก `YOLOBrailleDetector(dot_color=None)` อย่างชัดเจน ไม่ใช่ค่าเริ่มต้นของกล้อง

## ตัวอ่านเฉพาะสี

[สาเหตุที่แก้ อัลกอริทึม ข้อจำกัด และผลทดสอบ](docs/COLORED_BRAILLE_ONLY_TH.md)

รองรับเฉพาะ `--dot-color blue` และ `red` กด `C` ในหน้ากล้องเพื่อสลับสีระหว่างใช้งาน ระบบล้างผลสีเดิมและรอผลเฟรมใหม่ โดยไม่เปิดกล้องหรือโหลดโมเดลซ้ำ สีอื่นและจุดที่ไม่มีสีไม่ถูกนำไปอ่าน

กล้องติดตามตำแหน่งกรอบด้วย optical flow เมื่อขยับเล็กน้อย หากติดตามไม่ได้หรือผลเกินอายุ 1 วินาที จะล้างข้อความยืนยันและรอเฟรมใหม่ กล้องยังทำงานต่อ ค่า sharpening เริ่มต้นเป็น `0` เพื่ออ่านขอบสีจากภาพเดิม

สำหรับกล้องบนขาตั้ง ใช้ `--scan-roi X1 Y1 X2 Y2` กำหนดพื้นที่อ่านคงที่เป็นสัดส่วน 0–1 ของภาพกล้อง ตัดพื้นที่ก่อนตรวจสี/YOLO และจำกัด zoom ภายในกรอบนั้น ค่าเริ่มต้นยังอ่านทั้งภาพ ตัวอ่านแยกก้อนสีที่มีแกนจุดสองก้อนก่อนสร้าง Grid และตรวจมุมแต่ละแถบข้อความเมื่อมุมรวมจัดแถวไม่ได้

[รีวิวกล้องบนขาตั้ง วิธีเลือก ROI ผลทดสอบ และวิธีเก็บภาพหนังสือเพื่อฝึก YOLO](docs/MOUNTED_CAMERA_REVIEW_TH.md)

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

เก็บโมเดลทดลองใน `runs/detect/<ชื่อรอบ>/weights/` โดยไม่เขียนทับ production ใช้ `--weights` สำหรับฝึกรอบใหม่หรือ `--resume` สำหรับงานที่หยุดกลางคัน ประเมิน candidate ด้วย `tools/diagnostics/benchmark_yolo_stream.py --model <best.pt>` โหมด `--roi yolo` เลือกโมเดลเสนอ ROI ด้วย `--model <best.pt>` ได้ ส่วน `--roi off` ไม่ใช้ weights

## ใช้งานกล้องและภาพเดี่ยว

เปิด `Start_Scanner_YOLO.bat` หรือ:

```powershell
.venv\Scripts\python.exe camera_reader.py --lang thai --res fhd --sharp 0 --dot-color blue
.venv\Scripts\python.exe yolo_detector.py path/to/image.png --lang thai --dot-color blue --save --dump-stream
```

กด `C` สลับน้ำเงิน/แดง, `P` บันทึก snapshot, `D` บันทึกภาพที่ตัวอ่านใช้จริงพร้อม trace รายเซลล์ และ `Q`/Esc ออก กล้องยังรองรับ zoom, sharpness และสลับความละเอียด `--dump-stream` สร้างภาพ crop สำหรับตรวจสอบและ JSON พร้อมสีของเฟรมนั้นและสถานะ ROI

ทดลองให้ YOLO เสนอ ROI ก่อนอ่านสี:

```powershell
.venv\Scripts\python.exe camera_reader.py --lang thai --res fhd --dot-color blue --roi yolo
```

โมเดลปัจจุบันตรวจ `braille_dot` จึงใช้กรอบรวมจุดพร้อมขอบเผื่อ ไม่ใช่โมเดลตรวจกรอบบรรทัด/ตัวอักษรโดยตรง ติดตาม ROI ระหว่างรอบ YOLO และค้นหาใหม่ทุกประมาณ 0.35 วินาทีหรือเมื่อ tracking หลุด ภายใน ROI อ่านเฉพาะจุดสีที่เลือก หาก YOLO หา ROI ไม่พบจะรอเฟรมใหม่ โหมดนี้ยังต้องเทียบกับ `--roi off` บนหนังสือจริง

## ตรวจสอบ

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe tools\training\validate_dataset.py datasets/thai_synthetic/seed_20260908
.venv\Scripts\python.exe tools\diagnostics\benchmark_yolo_stream.py
```

ผลตรวจวันที่ 13 กันยายน 2026: ผ่าน **99 tests**, ทดลอง ROI ด้วยโมเดลจริงบนภาพสังเคราะห์น้ำเงิน/แดงตรงและเอียง 3° ผ่าน **4/4 กรณี** และ benchmark เส้นทาง YOLO เดิม **99/99** ผลเหล่านี้ไม่ใช่ตัวเลขความแม่นยำของหนังสือจริงหรือ FPS กล้อง/บอร์ด

ยังต้องทดสอบกับภาพดิบจากกล้องจริงในสภาพแสงและการแต้มสีที่หลากหลาย พิกัด/รูปแบบสีใน tests ไม่ได้ใช้เป็นค่าตายตัวใน algorithm

[การทำงานของ cell stream และข้อจำกัด](docs/YOLO_CELL_STREAM.md) · [ประวัติโค้ด CV ที่เลิกใช้](archive/legacy_cv/README.md)
