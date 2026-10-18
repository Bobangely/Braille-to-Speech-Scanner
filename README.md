# Thai Braille Reader — Painted Dots

## ติดตั้งและเปิดจาก source

ชุดนี้รองรับการรันจากโฟลเดอร์ source ไม่ใช่แพ็กเกจ pip/wheel และไม่มีคำสั่งติดตั้ง `braille-reader` ให้ใช้ Python 3.12 แบบ 64-bit แล้วเปิด terminal ในโฟลเดอร์ที่มี README นี้:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\main.bat
```

หรือใช้ `uv sync --locked` เพื่อเตรียม dependencies แล้วเรียก `uv run camera_reader.py` จากโฟลเดอร์โปรเจกต์ บน Linux ใช้ `python3.12` และ `.venv/bin/python` แทนคำสั่ง Windows; ต้องมี desktop/GUI กล้อง และอุปกรณ์เสียงที่ OS รองรับ ยังไม่รับรอง Linux ARM64 จนกว่าจะทดสอบบนบอร์ดจริง

requirements หลักไม่รวมโมเดลเสียงหรือ dependencies เสียงเสริม: Edge ต้องติดตั้ง `requirements-edge-tts.txt` ใน environment เสียงและใช้อินเทอร์เน็ต ส่วน Piper/MMS ต้องติดตั้ง requirements และโมเดล local ตามคู่มือเสียงด้านล่าง ไม่ใช้ `uv build` หรือ `pip install .` เพื่อแจกโปรแกรมรุ่นนี้

กล้องเริ่มด้วย `--res auto`: ตรวจเฟรมจริงของ 4K / 2K / 1080p แล้วเลือกโหมดสูงสุดที่ใช้ได้ ปุ่ม `V` วนเฉพาะโหมดที่ตรวจผ่าน หากเปิดไม่ได้จะลองโหมดต่ำกว่า; CLI ที่ระบุ preset/ขนาดเองยังใช้ได้เป็น override

ถอด Sharpness และปุ่ม `E` แล้ว ส่ง pixel ต้นฉบับเข้า Detection โดยตรง คง `--sharp 0` เพื่อ compatibility; ค่าอื่นไม่รองรับ ไม่ได้เปลี่ยนกฎ Parser ไทย/อังกฤษ

รีวิวก่อน merge: [บั๊กที่แก้ การเก็บโค้ด ผลตรวจ และข้อจำกัด](docs/PRE_MERGE_REVIEW_TH.md)

หน้าจอกล้องใหม่: [Preview, ปุ่มควบคุม, การ resize และผลทดสอบ UI](docs/SCANNER_UI_TH.md) — แสดงชื่อเซลล์ จุด 1–6 และ pattern บน Preview โดยตรง กด `H` เพื่อเปิด/ปิดข้อมูลตรวจสอบเพิ่มเติม ปุ่ม Sharpness เดิมถูกยกเลิกแล้ว

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

ดับเบิลคลิก `main.bat` เพื่อเปิดเมนูเดียว: scanner, scanner พร้อม Piper offline หรือ Edge online,
ตรวจกล้อง, ทดสอบภาพ/ภาพตัวอย่างเดิม และติดตั้ง dependencies ลง `.venv` แบบเลือกเอง
ลากไฟล์ภาพมาวางบน `main.bat` ได้ หรือเรียกคำสั่งโดยตรง:

```powershell
.\main.bat scan
.\main.bat scan --res 1080p --camera 0
.\main.bat piper
.\main.bat edge
.\main.bat camera-test
.\main.bat image "path/to/image.png" --lang thai --no-show
.\main.bat setup
```

โหมด `scan` ไม่เปิดเสียง; เลือก `piper` เพื่อเปิดปุ่ม S อ่าน/อ่านซ้ำ และ B หยุดเสียง
โดยต้องมี environment/โมเดลในเครื่องก่อน ไม่มีการดาวน์โหลดอัตโนมัติหรือ Cloud fallback
ดู [Piper offline](docs/PIPER_OFFLINE_TTS_TH.md), [Edge online ไทย/อังกฤษและวิธีติดตั้งหลัง pull](docs/EDGE_TTS_TH.md) และ [ผล audit launcher](docs/MAIN_LAUNCHER_AUDIT_TH.md)

`main.bat` หาไฟล์โปรเจกต์จากตำแหน่งของตัวเอง จึงเรียกจากโฟลเดอร์อื่นได้
path ภาพแบบ relative อ้างอิงโฟลเดอร์ที่เรียกคำสั่ง ส่วน scanner options ส่งต่อให้ CLI เดิม
หาก dependencies ไม่ครบ จะแจ้งให้รัน `main.bat setup` โดยไม่ติดตั้งเอง

คำสั่ง Python เดิมยังใช้ได้สำหรับงานพัฒนา:

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

## Lexicon ช่วยจัดอันดับภาษาไทย

ติดตั้ง dependencies จาก `requirements.txt` หรือ `uv sync --locked` เพื่อใช้ชุดคำทั่วไป
ที่มากับ `pythainlp==5.3.3` ระบบโหลดชุดคำครั้งเดียวและไม่ดาวน์โหลดข้อมูลระหว่างอ่านกล้อง
หากชุดคำใช้ไม่ได้ จะคงผลจาก structural parser พร้อมบันทึกคำเตือน

- ตรวจโครงสร้างและ source cells ของทุก segmentation ก่อนให้คะแนน
- จับคำที่ขอบพยางค์ภายในช่วงเดียวกัน ไม่จับข้ามช่องว่างหรือบรรทัด
- คำที่ไม่มีในชุดคำยังเป็น candidate ได้ รวมถึงชื่อเฉพาะและคำใหม่
- ใช้คำเป็นหลักฐานรองหลังตรวจ raw patterns, กฎเบรลล์ และการประกอบอักษร ไม่ทำให้คู่แข่งผิดเพียงเพราะไม่พบคำ
- ยืนยันด้วย lexicon ได้เมื่อค้น candidate ครบ มีเพียงทางเดียวครอบคลุมคำครบ เป็นมาตรฐาน มีหลักฐาน raw cells ไร้คำเตือน ผ่านเกณฑ์ confidence ที่กำหนด และมีอย่างน้อยสองคำพร้อมคำที่ครอบคลุมหลายพยางค์ ทางเลือกอื่นยังเก็บเป็น valid/plausible ใน trace
- หากมีหลายทางที่ครอบคลุมคำครบ หลักฐานไม่เพียงพอ หรือค้น candidate ไม่ครบภายในขอบเขต 128 ทาง
  จะคง `NEEDS_REVIEW` และไม่ยืนยันข้อความนั้น

`decode_thai_details(cells, lexicon=ชุดคำ)` ใช้ชุดคำของผู้เรียกได้;
ส่ง `lexicon=set()` เพื่อปิดหลักฐาน lexicon โดยคง parser เดิม
ข้อมูล debug เดิมอยู่ใน `raw_patterns`, `parser_trace`, `semantic_tokens` และ
`structural_syllables`; ทางเลือกทั้งหมดกับคะแนนอยู่ใน `segmentation_runs` และ
`lexicon_ranking` ส่วน `syllables` แสดงทางที่เลือกสำหรับ renderer

## มาตรฐาน raw Braille และ compatibility ภาษาไทย

เส้นทาง scanner ใช้กฎ raw ของ Liblouis Thai Grade 1 เป็นหลัก โดยแยกจาก semantic
compatibility ของโปรเจกต์ ลำดับเลือกคือ `standard > compatibility > lexicon`
และเก็บ candidate ที่ถูกปฏิเสธไว้ใน `segmentation_runs` พร้อมเหตุผลของแต่ละ mode

กฎ raw ที่ตรวจสอบและเพิ่มในรอบนี้อยู่ใน `thai_raw_rules.py`: `15` แทนรูปเขียน `ัว`
จึงห้ามนำตัวสะกดจากพยางค์ถัดไปมาทำให้ `ัว` กลายเป็น `ว` โดยไม่มีหลักฐานจาก raw
ส่วนอัวที่ลดรูปแล้วใช้ `ว` (`2456`) ยังคงอ่านผ่าน parser เดิม ไม่ห้ามตัวสะกด `ร`
และไม่ใช้รายชื่อคำมาตัดสิน ดูแหล่งตารางที่ตรึง revision ไว้ใน `RAW_RULES_SOURCE`

การเปลี่ยนนี้เป็น **scanner subset ที่ตรวจสอบกฎอัวแล้ว** ไม่ใช่การกลับตาราง Liblouis
ทุกกฎแบบสมบูรณ์ กฎ segmentation/สระอื่นที่มีอยู่ยังคงเดิมเพื่อรักษา regression ของ P1

ค่าเริ่มต้นไม่รับ raw รูปปิดที่อาศัย semantic compatibility เพียงอย่างเดียว หากเป็นข้อมูล
legacy ที่ใช้ `15` เป็นสระอัวแบบปิด ให้เปิด mode อย่างชัดเจน:

```python
from thai_decoder import decode_thai_details

result = decode_thai_details(cells, compatibility=True)
```

`decode_cells()` และ `decode_cells_verbose()` รับ keyword เดียวกัน ส่วน `compose_tokens()`
เป็น semantic adapter จึงคง compatibility สำหรับ token เดิมที่ไม่มี raw encoding
เมื่อทั้งสอง mode ให้คำตอบที่ต่างกันและยัง valid ระบบแสดง standard ก่อน แต่คืน
`NEEDS_REVIEW`; lexicon จัดอันดับได้เฉพาะภายในระดับ mode เดียวกัน ไม่สามารถชุบ
candidate ที่ผิดโครงสร้างหรือทำให้ compatibility แซง standard ได้

debug มี `standard_valid`, `standard_error`, `compatibility_valid`,
`compatibility_error`, `accepted_mode` และ `acceptance_reason` โดยไม่แก้ raw cells,
patterns, source cell IDs หรือกฎ confirmation/Vision/Grid

## ตรวจสอบ

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe tools\training\validate_dataset.py datasets/thai_synthetic/seed_20260908
.venv\Scripts\python.exe tools\diagnostics\benchmark_yolo_stream.py
```

ผลตรวจวันที่ 13 กันยายน 2026: ผ่าน **99 tests**, ทดลอง ROI ด้วยโมเดลจริงบนภาพสังเคราะห์น้ำเงิน/แดงตรงและเอียง 3° ผ่าน **4/4 กรณี** และ benchmark เส้นทาง YOLO เดิม **99/99** ผลเหล่านี้ไม่ใช่ตัวเลขความแม่นยำของหนังสือจริงหรือ FPS กล้อง/บอร์ด

ยังต้องทดสอบกับภาพดิบจากกล้องจริงในสภาพแสงและการแต้มสีที่หลากหลาย พิกัด/รูปแบบสีใน tests ไม่ได้ใช้เป็นค่าตายตัวใน algorithm

[การทำงานของ cell stream และข้อจำกัด](docs/YOLO_CELL_STREAM.md) · [ประวัติโค้ด CV ที่เลิกใช้](archive/legacy_cv/README.md)
