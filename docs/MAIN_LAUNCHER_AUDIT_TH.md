# รวม launcher เป็น main.bat — 15 ตุลาคม 2026

Branch: `feature/single-bat-entrypoint` จากงาน Piper `648049e`; ไม่แก้ branch `main`

## หน้าที่ที่ตรวจและย้าย

| ไฟล์เดิม | หน้าที่/dependency เดิม | ทางเรียกใหม่ |
|---|---|---|
| `Start_Scanner_YOLO.bat` | เลือก Python, cd project root, เปิด `camera_reader.py` ด้วย camera 0 / YOLO stream / res auto / sharp 0 / Thai | `main.bat scan` หรือเมนู 1; defaults เดิมคงไว้และส่ง options เพิ่มได้ |
| `Test_Camera.bat` | หากไม่มี `.venv` เรียก installer แล้วเปิด `tools/diagnostics/check_camera.py` | `main.bat camera-test` หรือเมนู 3 |
| `Test_YOLO.bat` | หากไม่มี `.venv` เรียก installer; มีเมนู 4 ภาพเดิม, custom path, drag/drop; ภาษา thai / thai-legacy / english | `main.bat image`, `main.bat image "path"`, หรือลากภาพวางบน main.bat; เมนู 4 |
| `install_dependencies.bat` | สร้าง `.venv`, upgrade pip, install requirements; เดิมมี fallback ไป global pip และรายการ launcher ที่ไม่มีอยู่แล้ว | `main.bat setup` หรือเมนู 5; ติดตั้งเฉพาะ `.venv`, ไม่ fallback global, คืน error code เมื่อผิดพลาด |

ลบเฉพาะ `.bat` 4 ไฟล์ข้างต้นหลังทดสอบหน้าที่แทนที่แล้ว ไม่ลบ source, model, config, dataset หรือ environment
ยังคง `.bat` อีก 6 ไฟล์ภายใน environment/dependencies เพราะเป็น activation/build files ของแพ็กเกจ ไม่ใช่ launcher ของแอป

## โครงสร้างใหม่

- `main.bat`: entry point เดียวของผู้ใช้; `pushd` ตำแหน่งของตัวเองแล้ว `popd` คืน cwd, ปิด delayed expansion,
  quote paths, เลือก `.venv\Scripts\python.exe` ก่อน `py -3` / `python` ที่เป็น Python >=3.12
- `tools/launch.py`: standard-library only; menu, dependency checks, script/image checks และส่ง argv เป็น list โดยไม่ใช้ shell
- `main.bat piper`: เปิด scanner + โมเดล offline ที่เตรียมแล้วใน `scratch/piper-env` และ `scratch/piper-voices`
  ตรวจไฟล์ `.onnx`/`.onnx.json` และแพ็กเกจ Piper/ONNX/TLTK/requests ก่อนเปิดแอป ไม่มีการดาวน์โหลดหรือ Cloud fallback
- ไม่ส่ง arguments: เปิดเมนู; error ไม่ทำให้หน้าต่าง double-click ปิดจนอ่านข้อความไม่ทัน
- path ภาพแบบ relative อ้างอิง cwd ของผู้เรียก; path ภายในแอปและ scanner options คงอ้างอิง project root ตาม launcher เดิม
- การเลือกภาพตัวอย่างยังคงภาษา legacy/อังกฤษเดิม; presets ไม่เพิ่ม dump-stream เอง ส่วน custom/drag-drop เก็บ crop/debug ตามเดิม
- Python API และคำสั่งเปิดโปรเจกต์เดิมยังใช้งานได้ ไม่เปลี่ยน detection, Grid, Mapping, Parser หรือ TTS backend

## References / shortcuts ที่ตรวจ

- ค้น `.bat`, `.lnk`, ไฟล์ hidden, โค้ด, scripts, `.vscode` และเอกสารในโปรเจกต์
- ตรวจ Desktop, OneDrive Desktop และ Start Menu ของผู้ใช้ปัจจุบันผ่าน WScript.Shell โดยอ่าน shortcut เท่านั้น
- พบ `braille-reader.lnk` ใน Start Menu ชี้เข้าโฟลเดอร์โปรเจกต์ ไม่ได้เรียก `.bat` จึงเก็บไว้โดยไม่แก้
- ไม่พบ shortcut ในพื้นที่ที่ตรวจซึ่งเรียก `.bat` ที่ลบโดยตรง; ไม่ได้สแกน shortcuts ทุกไดรฟ์หรือผู้ใช้อื่น
- แก้คู่มือใช้งานจริงเป็น `main.bat`; ชื่อเก่าที่อยู่ในเอกสาร audit นี้, รายงาน cleanup เดิมและ archive เป็นประวัติที่ระบุชัดว่าเลิกใช้แล้ว
- ถ้ามี shortcut ภายนอกพื้นที่ที่ตรวจซึ่งชี้ launcher เก่า ต้องเปลี่ยน Target เป็น path ของ `main.bat`

## วิธีเปิด

```powershell
.\main.bat
.\main.bat scan --res 1080p
.\main.bat piper --res 1080p
.\main.bat camera-test
.\main.bat image
.\main.bat image "C:\Pictures\ภาพ A & B.png" --lang thai --no-show
.\main.bat setup
.\main.bat --help
```

`setup` เป็นการเลือกติดตั้งอย่างชัดเจนและอาจใช้ internet; เปิด scanner/image ปกติจะไม่ติดตั้งเอง
ถ้า `.venv` มีอยู่แต่ไม่สมบูรณ์จะให้ซ่อมเอง ไม่ลบหรือเขียนทับ environment
เมนู Piper ใช้เฉพาะเสียงที่เตรียมแล้วและยังอยู่ภายใต้ข้อจำกัด non-commercial ของเสียงไทย ดู `PIPER_OFFLINE_TTS_TH.md`

## การตรวจสอบ

Regression tests รัน `cmd.exe` จริงใน project จำลองที่ชื่อมีช่องว่าง, `&` และ `!`:
เมนูเปิดปกติ, cwd อื่น, Unicode/space/ampersand image path, relative image path, args overrides,
camera diagnostic dispatch, sample menu legacy/English, fallback Python เมื่อไม่มี `.venv`, dependency/script/image หาย,
Piper files หาย/ครบ, app exit code, pip error และไม่มี Python/helper

การติดตั้งใน tests ใช้โมดูล pip จำลอง: ไม่ติดตั้งแพ็กเกจจริงและไม่แตะ environment ของผู้ใช้
การทดสอบกล้องใน fixtures ตรวจการส่งคำสั่ง ไม่อ้างว่าได้ตรวจฮาร์ดแวร์กล้อง

ผลรันจริงบน Windows:
- tests ของ launcher ผ่าน 13 tests; suite รวมผ่าน **410 tests** ด้วย `python -X utf8 -m unittest discover -s tests -q`
- เรียก `main.bat scan --help` และ `main.bat piper --help` จาก `%TEMP%` สำเร็จ exit 0
- เรียก `main.bat image <ภาพสังเคราะห์ใน scratch> --lang english --no-show` จาก `%TEMP%` สำเร็จ exit 0
  ตัวอ่านทำงานและสร้าง annotated image / stream manifest ตามคำสั่งเดิม ผลนี้ไม่ได้รับรองความแม่นยำบนหนังสือจริง
- ตรวจหลังลบพบ `.bat` ในตัวโปรเจกต์ 1 ไฟล์ (`main.bat`) และใน environment/dependencies 6 ไฟล์เดิม
- ไม่ติดตั้ง dependencies จริงซ้ำ ไม่เปิดกล้องจริง และไม่ทดสอบการ double-click ด้วยเมาส์ในรอบนี้;
  การเปิดเมนูไม่มี arguments ทดสอบผ่าน cmd พร้อม input จำลอง
