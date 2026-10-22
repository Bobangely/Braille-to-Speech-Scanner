# ทดลอง Piper offline TTS — 15 ตุลาคม 2026

สถานะ: ทดลองบน Windows สำเร็จ ยังไม่ยืนยันบน QCS6490 จริง
Branch: `feature/piper-offline-tts` จาก `codex/fix-printed-blue-grid` (`f66ba6c`)
ไม่ได้เปลี่ยน `main.py`, branch `main`, ตัวตรวจจุด, Grid, Mapping หรือ Thai Parser

## ขอบเขตและแผนที่ดำเนินการ

1. ตรวจ environment, จุดเรียก TTS, ปุ่ม, dependency/voice license และ baseline tests
2. เพิ่ม backend แยกที่อ่านโมเดลในเครื่อง และเปิดใช้ผ่าน flags ของ `camera_reader.py`
3. อ่านเฉพาะ `_confirmed_text` หลังยืนยันผลแล้ว ใช้การเตรียมข้อความเสียงไทยเดิม
4. เพิ่มอ่าน/อ่านซ้ำและหยุดผ่านเส้นทาง mouse/keyboard เดิม ไม่เรียก Cloud fallback
5. ทดสอบ regression, สังเคราะห์จริง, เล่น/หยุดจริง และวัดทรัพยากรโดยแยกผลเครื่องพัฒนาจากบอร์ด

## วิธีทำงาน

- ไม่ส่ง flags TTS: scanner ทำงานเหมือนเดิม ไม่ import Piper และไม่เพิ่มปุ่มเสียง
- ส่งโมเดลไทย/อังกฤษ: เปิดปุ่ม `Read / repeat [S]` และ `Stop voice [B]`
- `[S]` อ่านข้อความที่ยืนยันแล้วในภาษาที่เลือกด้วย `[L]`; ไม่มีข้อความจะไม่เริ่มงาน
- กด `[S]` ซ้ำหยุดงานเดิมและอ่านใหม่; `[B]`, เปลี่ยนภาษา หรือปิด scanner หยุดงาน
- Worker เป็น process แยก จึงไม่รอสังเคราะห์/เล่นเสียงในวงกล้อง
- Windows ใช้ `winsound`; Linux ใช้ ALSA `aplay` ผ่าน process group เพื่อหยุดลูก process ด้วย
- ต้องมี `.onnx` และ `.onnx.json` ในเครื่องแล้ว; ไม่มีการดาวน์โหลดขณะใช้งาน
- ข้อผิดพลาดแสดง `Speech error` พร้อมรายละเอียดใน console โดยไม่ fallback ไป gTTS
- โหลดโมเดลใหม่ทุกครั้งเพื่อลดความซับซ้อนในการยกเลิกงาน ช่วงรอรวมโหลดและสังเคราะห์ทั้งข้อความ
- ยังไม่แยกเสียงไทย/อังกฤษอัตโนมัติภายในประโยคเดียว ให้เลือกภาษาก่อนอ่าน
- API ของ `tts.py` และพฤติกรรม CLI เดิมไม่เปลี่ยน: การทดลอง offline นี้เปิดผ่าน `camera_reader.py` เท่านั้น

## Environment และ license ที่ตรวจ

เครื่องพัฒนา: Windows 11 x64, Python 3.12.13, Ryzen 7 7800X3D (8 cores / 16 threads), RAM ประมาณ 31 GiB
ติดตั้งแยกที่ `scratch/piper-env`; ไม่เปลี่ยน `.venv` หรือ `uv.lock` ของแอป

| ส่วน | รุ่น/แหล่ง | เงื่อนไข |
|---|---|---|
| Piper | `piper-tts[th]==1.8.0` | GPL-3.0; มี Windows x64 และ Linux ARM64 wheel |
| ONNX Runtime | ทดสอบ 1.31.0, CPUExecutionProvider | MIT; ARM64 wheel รุ่นที่ตรวจต้องเป็น glibc >=2.28 |
| TLTK | 1.10 (ตามข้อกำหนด Piper `<1.11`) | BSD-3-Clause; ใช้ร่วมกับ pandas, nltk, sklearn, gensim และ unicode-rbnf |
| requests | 2.34.2 ในการทดลอง | Apache-2.0; TLTK import แต่ไม่ประกาศ dependency จึงเพิ่มใน requirements-piper.txt |
| ไทย | `th_TH-tsync2-medium` ประมาณ 63.2 MB | MODEL_CARD ระบุข้อมูลฝึก CC BY-NC-SA 3.0, **non-commercial**; ใช้เฉพาะทดลอง ไม่อนุมัติสำหรับผลิตภัณฑ์เชิงพาณิชย์ |
| อังกฤษ | `en_US-ljspeech-medium` ประมาณ 63.5 MB | MODEL_CARD ระบุข้อมูลฝึก LJ Speech เป็น public domain; เก็บ model card และตรวจสิทธิ์น้ำหนักก่อนแจกจ่าย |

License ของโค้ดไม่ใช่ license ของทุกเสียง ต้องปฏิบัติตาม notice/source obligations ของ dependencies เมื่อแจกจ่าย
ไม่สรุปว่า model card ที่ระบุ license ของ dataset เป็นใบอนุญาตน้ำหนักโมเดลแยกต่างหาก

แหล่งหลัก:
- https://pypi.org/project/piper-tts/1.8.0/
- https://github.com/OHF-Voice/piper1-gpl
- https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md
- https://huggingface.co/rhasspy/piper-voices/blob/main/th/th_TH/tsync2/medium/MODEL_CARD
- https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/ljspeech/medium/MODEL_CARD
- https://pypi.org/project/tltk/1.10/
- https://pypi.org/project/onnxruntime/1.31.0/

เสียงทั้งสองชุดที่ทดลอง pin repository revision `c10ece1aade47bb51c153c893d14e5bf8e5b7117`
ไฟล์จริงและ provenance เก็บใน `scratch/piper-voices` (gitignored); ไม่ใส่น้ำหนักโมเดลใน Git

SHA-256:
- ไทย ONNX: `136669378ee72dd27012604994099ca3c833e625c7cfdff05ccdf468a1a2115f`
- ไทย JSON: `212aff35b8beed828a5f2af33b3cfc7744f4fac3c3dc7e6a2f464692e8257bb7`
- อังกฤษ ONNX: `6f52a751e2349abe7a76735eb09dc1875298c77ea2342ffd2fef79ff81b87f22`
- อังกฤษ JSON: `141d612cc0a95ed7efc1ca936b845c2364967f2e9217c5dbfcf69fc4d6c65860`

## ผลทดลองจริงบน Windows

แต่ละแถวเริ่ม process/โหลดโมเดลใหม่; CPU-only, เสียง mono PCM16 22,050 Hz
บล็อก Python socket connect/getaddrinfo/bind ระหว่าง import และ synthesis แล้วสังเคราะห์สำเร็จ
นี่เป็นการทดสอบใน process ที่ห้าม network ไม่ใช่การถอดสาย/ปิดเครือข่ายทั้งเครื่อง

| ข้อความ | เวลาจนไฟล์พร้อม รวมเริ่ม process | ความยาวเสียง | RAM สูงสุดรวม process ลูก | CPU time รวม |
|---|---:|---:|---:|---:|
| นิทานพยัญชนะ 3 | 5.15 s | 2.38 s | 500.8 MiB | 5.11 CPU-s |
| สวัสดีครับ วันนี้อ่านอักษรเบรลล์ภาษาไทย | 5.31 s | 4.25 s | 559.6 MiB | 5.56 CPU-s |
| Hello. This is a Braille reader. | 2.53 s | 1.76 s | 159.9 MiB | 3.03 CPU-s |

เฉลี่ยประมาณ 1.0–1.2 CPU cores ระหว่างงาน ไม่ใช่ 100% ของ CPU ทุก core
วัด RAM/CPU จาก process tree เพราะ Windows venv มี launcher process; ไม่ใช้เฉพาะ launcher ซึ่งวัดได้ต่ำผิดจริง
เวลาในตารางคือ readiness ของ WAV ก่อนเล่น ไม่ใช่ค่าที่วัดจากไมโครโฟน/ลำโพงจริง

- ทดลองเล่นอังกฤษจนจบผ่าน Windows Sound API สำเร็จ รวมประมาณ 4.40 s
- ทดลองหยุดหลังสังเคราะห์ ระหว่างสถานะ Speaking สำเร็จ ประมาณ 0.002 s และลบไฟล์ชั่วคราวได้
- ทดลองหยุดระหว่างโหลด/synthesis จริง ไม่มีลูก process ค้างหลังหยุดในเครื่องนี้
- WAV มีสัญญาณไม่เงียบ; ยังไม่ได้ให้ผู้ฟังตัดสินการออกเสียง/ความเป็นธรรมชาติ โดยเฉพาะไทย วรรณยุกต์ และตัวเลข
- ชุดทดสอบมี regression สำหรับไฟล์หาย, ภาษาไม่มีโมเดล, ข้อความว่าง, อ่านซ้ำ, หยุด worker จริง, cleanup หลัง launch ล้มเหลว,
  error โดยไม่ fallback, ข้อความภาษาอังกฤษไม่ถูก shell ตีความ, ปุ่ม mouse/keyboard, อ่านเฉพาะข้อความที่ยืนยัน และไม่ reset Grid
- รัน tests เดิมที่เกี่ยวข้องและ test suite ทั้งหมดผ่าน **397 tests** ด้วย `python -m unittest discover -s tests -q`
- ไม่มีผลทดสอบกล้อง+YOLO+TTS พร้อมกันบนฮาร์ดแวร์จริง

ผลและเสียงตัวอย่างในเครื่องนี้: `scratch/piper-benchmark.json`, `scratch/piper-playback-results.json`,
`scratch/piper-thai-0.wav`, `scratch/piper-thai-1.wav`, `scratch/piper-english-2.wav`

## ทดลองบนเครื่องนี้

ทางเปิดสำหรับผู้ใช้ปัจจุบัน: ดับเบิลคลิก `main.bat` แล้วเลือกข้อ 2 หรือใช้ `main.bat piper`
พร้อมส่ง scanner options ได้ เช่น `main.bat piper --res 1080p` โดยใช้ไฟล์ทดลองที่มีอยู่แล้ว
รายละเอียด Python ด้านล่างเก็บไว้สำหรับการติดตั้ง/งานพัฒนา

จาก project root (PowerShell):

```powershell
# ขั้นติดตั้งนี้ดำเนินการไว้แล้วใน environment ทดลอง
uv venv scratch/piper-env --python .venv/Scripts/python.exe
uv pip install --python scratch/piper-env/Scripts/python.exe -r requirements-piper.txt

.\.venv\Scripts\python.exe camera_reader.py --res 1080p --lang thai --tts-python scratch/piper-env/Scripts/python.exe --tts-thai-model scratch/piper-voices/th_TH-tsync2-medium.onnx --tts-english-model scratch/piper-voices/en_US-ljspeech-medium.onnx
```

รอข้อความยืนยัน → `[S]` หรือคลิก Read / repeat → `[B]` เพื่อหยุด → `[L]` สลับภาษา
ไม่ติดตั้ง Piper เพิ่มใน `.venv` เพราะใช้ `--tts-python` แยก worker ได้

หากต้องเตรียมเครื่องใหม่ หลังตรวจ license/compatibility แล้วเท่านั้น:

```powershell
scratch/piper-env/Scripts/python.exe -m piper.download_voices th_TH-tsync2-medium en_US-ljspeech-medium --data-dir scratch/piper-voices
```

คำสั่ง downloader นี้อาจได้ revision ใหม่ ต่างจากไฟล์ที่ใช้วัดผลข้างต้น ต้องตรวจ hash และ MODEL_CARD ใหม่
หลังเตรียมไฟล์ครบ แอปไม่มีขั้น download อัตโนมัติ

ทดสอบโดยไม่ต้องติดตั้ง Piper ใน environment หลัก:

```powershell
.venv/Scripts/python.exe -m unittest tests.test_piper_tts tests.test_scanner_ui tests.test_camera_recovery tests.test_confirmed_result_display tests.test_thai_multicell_symbols tests.test_thai_composition -q
```

## สิ่งที่ต้องตรวจบนบอร์ดจริงก่อนติดตั้ง

ไม่มีบอร์ดหรือ ADB/SSH endpoint ให้ตรวจในการทดลองนี้ **จึงยังไม่ยืนยันว่าใช้บนบอร์ดนี้ได้จริง**
ชื่อ QCS6490 ไม่ยืนยัน OS, Python ABI, libc, driver เสียง หรือพื้นที่/RAM ที่เหลือ

ให้เก็บผลบนบอร์ดก่อน:

```sh
uname -m
cat /etc/os-release
getconf GNU_LIBC_VERSION
python3 --version
python3 -m pip debug --verbose
free -m
df -h
command -v aplay
aplay -l
```

เกณฑ์ที่ต้องตรวจ:
- Linux aarch64 + glibc ที่ wheel ของ Piper และ ONNX Runtime ใช้ได้จริง; Android/Bionic หรือ musl ใช้ wheel นี้ตรง ๆ ไม่ได้
- Python ของโปรเจกต์ต้อง >=3.12; wheel Piper เป็น cp39-abi3 แต่ไม่ได้หมายความว่า dependencies ทุกตัวรองรับ Python/OS เดียวกัน
- `pip` ต้อง resolve dependencies ไทยได้โดยไม่ดึง GPU runtime; รุ่น ORT ที่ทดสอบใช้ glibc >=2.28
- มี ALSA `aplay` และ default audio device ที่เล่น WAV mono 22,050 Hz ได้จริง
- RAM สูงสุดข้างต้นเป็น TTS อย่างเดียว ต้องวัดพร้อมกล้อง/YOLO และตรวจ OOM, swap, อุณหภูมิ
- วัดเริ่มมีเสียงจากกดปุ่ม, การหยุด, เล่นซ้ำ, ภาษาไทย/อังกฤษ และการเปลี่ยนภาษาจริง
- ฟังคำควบกล้ำ วรรณยุกต์ สระประสม ตัวเลขและเครื่องหมาย และให้ผู้ใช้ภาษาไทยประเมิน
- ทดสอบปิด Wi-Fi/ถอด network จริงหลังเตรียม dependencies/voices ครบ

ถ้าไม่ผ่าน ABI/libc/audio/RAM ให้หยุดรายงานก่อนเปลี่ยน runtime หรือวิธี playback; ไม่มี Cloud fallback
