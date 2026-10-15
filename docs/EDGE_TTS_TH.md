# Edge TTS สำหรับ Braille Scanner

2026-10-15 · branch `feature/thai-offline-tts` · เปลี่ยนจาก commit `71f9c8a`

## วิธีติดตั้งและเปิดหลัง pull

Edge เป็น online ทั้งไทยและอังกฤษตามที่ผู้ใช้เลือก ไม่ต้องมี Piper/MMS models หรือ API key สำหรับโหมดนี้ ต้องมีอินเทอร์เน็ตขณะอ่าน
เครื่องเพื่อนต้องติดตั้ง environment เอง เพราะ scratch/ และ .venv/ ไม่อยู่ใน Git

```powershell
.\main.bat setup
.\.venv\Scripts\python.exe -m venv scratch\edge-tts-env
.\scratch\edge-tts-env\Scripts\python.exe -m pip install -r requirements-edge-tts.txt
.\main.bat edge
```

หรือดับเบิลคลิก main.bat แล้วเลือกข้อ 6 หลังติดตั้งครบ กด Read/repeat หรือ S เพื่ออ่าน/อ่านซ้ำ และ Stop voice หรือ B เพื่อหยุด
โหมด offline ยังเปิดด้วย main.bat piper ตามเดิม; main.bat scan ยังเปิด scanner ไม่เปิดเสียง
เมื่อใช้งานบน Linux ต้องเตรียม environment ที่เหมาะกับ OS และระบุ tts-python เป็น venv/bin/python ผ่าน camera_reader.py; ยังไม่ได้ทดสอบ Linux/ARM64 และต้องมี ALSA aplay สำหรับตัวเล่น WAV เดิม

## เสียงและความเร็ว

- ไทย: `th-TH-NiwatNeural` (ปัจจุบัน; ผลทดลองช่วงแรกด้านล่างใช้ Premwadee)
- อังกฤษ: `en-US-JennyNeural`
- อัตราอ่าน Edge: `-15%` ตาม feedback ให้ช้าลง อยู่ใน EDGE_RATE ของ piper_tts.py ไม่แก้ข้อความเพื่อสร้างจังหวะหยุดเทียม
- ความเร็วนี้ใช้กับ utterance ทั้งชุด จังหวะระหว่างประโยคขึ้นกับเครื่องหมายเดิมและบริการ ไม่เพิ่มปุ่ม next-sentence หรือแยก/รวมข้อความใหม่

- ก่อนส่งเสียงไทย ระบบขยายพยัญชนะที่อยู่เป็นหน่วยอิสระครบ 44 ตัวเป็นชื่ออ่านผ่าน `thai_speech.py` ใช้ร่วมกับ offline โดยคงคำปกติและ Detection Result เดิม
- ขอบเขตชื่ออ่านรองรับช่องว่าง บรรทัดใหม่ วรรคตอน และตัวเลข; เลขในบริบท “พยัญชนะตัวที่ 1–44” อ่านเป็นคำไทย ตัวเลขทั่วไปคงเดิม
- ตัวอย่างเสียงครบ 44 ชื่อของ Niwat สร้างสำเร็จและ metadata ตรงตามลำดับ ยังไม่ถือว่าผ่านการฟังประเมินความชัดทุกชื่อ

## Flow ที่ตรวจ

cells/dots → decoder.decode_cells('thai') → thai_decoder/Thai parser/renderer → confirmation gate → scanner._confirmed_text → ปุ่ม S → prepare_thai_speech_text → job ผ่าน stdin JSON → worker Edge → MP3 streaming → miniaudio PCM playback พร้อมบันทึก WAV สำหรับอ่านซ้ำ

1. กดอ่านแล้วส่งเฉพาะข้อความที่ผ่าน confirmation เดิม ไม่ส่ง frame/cells ไป Microsoft ไม่เริ่มสังเคราะห์อัตโนมัติทุกเฟรม
2. ตรวจว่า decoded/ข้อความ UI/ข้อความ job ตรงกับ expected และ cells ไม่ถูกแก้ระหว่าง decode หรือส่งเสียง
3. prepare_thai_speech_text ใช้ normalization และชื่ออ่านเครื่องหมายจาก config เดิม เช่น วงเล็บเปิด; แยกข้อความสำหรับเสียงออกจาก UI ไม่มีการเติมคำเฉพาะให้ประโยคผ่าน
4. Edge ใช้ worker process เดิม จึงหยุด network request และ playback ด้วยปุ่ม B/เปลี่ยนภาษา/ปิด scanner ได้ตาม lifecycle เดิม ไม่สร้าง thread/controller ซ้ำ
5. ใช้ soundfile แปลง MP3 เป็น WAV เพื่อใช้ winsound/ALSA เดิม ไม่ใช้ shell เปิดโปรแกรมเล่น MP3 ไฟล์ชั่วคราวอยู่ใน workspace ของ job และล้างเมื่อหยุด
6. network/service error แสดง Speech error ผ่านสถานะเดิม ไม่มีการสลับเสียงหรือ Cloud/offline fallback เงียบ ๆ กดอ่านซ้ำเองได้

Legacy TextToSpeech ใน tts.py ยังมี gTTS/pyttsx3 สำหรับ call sites เก่า เช่นการอ่านภาพนิ่ง งานนี้เชื่อม Edge เฉพาะเส้นทาง Scanner ที่เปิดในโหมด Edge ไม่เปลี่ยนกฎถอดรหัสหรือเส้นทางเก่าเหล่านั้น

## ข้อมูลทดสอบและผล

- diagnostic จริง `diagnostic_1789489556525254800, frame 6699`: 22 cells → `คำที่มีใช้: ฐาน ฐิติ` ตรวจจาก fixture punctuation_cells.json
- fixture หนังสืออ้างอิงภาพ `codex-clipboard-f8bd4c38-dbed-47f8-ab9a-d4ac5f9d5807.png`: 12 patterns → `เขียดเขียวเกาะขอน` (fixture บันทึก actual_before_fix ว่า `เขียดเขียวกอะขอน` เป็นปัญหาประกอบข้อความในอดีต; baseline ปัจจุบันแก้ไว้แล้ว ไม่แก้ซ้ำในงานนี้)
- ภาพสังเคราะห์ในโปรเจกต์ sample_images/Test_thai_01.png: saved cell stream 12 cells → `สวัสดีครับผม` เก็บ fixture แบบ portable พร้อม image SHA-256; ไม่อ้างว่าเป็นภาพจากกล้องจริง
- acceptance cases เดิมของพยัญชนะ prefix, สระ, วรรณยุกต์ และคำเฉพาะ เช่น `ศักดิ์`, `ภู่`, `เสื้อ`, `เรียน`, `เดือน`, `เกิดมา`; เป็น reference-pattern tests แยกจาก diagnostic จริง
- ตัวเลข: number indicator + a/b/c patterns → `123`; punctuation colon จาก diagnostic และวงเล็บจาก config เดิม
- รวม **54 cases** ตรวจ exact text ตั้งแต่ patterns/recorded cells → decoded → prepare → S → job เข้า Edge ไม่พบคำหาย/ซ้ำ/สลับที่จาก expected ในชุดนี้ และเพิ่มกรณี label วงเล็บอีกหนึ่ง test

สร้างเสียงออนไลน์จริงจาก 7 กรณีไทย (diagnostic, หนังสือ, สวัสดี, ศักดิ์, เสื้อ, 123, ชุดวรรณยุกต์) และอังกฤษ Hello world
**พบ NoAudioReceived จริง**: รอบแรกของประโยคหนังสือไม่คืนเสียง การเรียก edge-tts โดยตรงนอกโปรเจกต์ก็ล้มเหลวได้ แต่ภายหลังข้อความเดิม/เสียงเดิมสร้างได้โดยไม่แก้อักษร สองตัวอย่างที่ลดความเร็วก็มีรอบแรกไม่คืนเสียงและ manual retry สำเร็จ
จึงแยกเป็นความล้มเหลวของบริการ/เส้นทางออนไลน์ ยังไม่มีหลักฐานว่าเป็นการตกคำของ decoder หรือการออกเสียงผิด ห้ามใช้กรณีไม่มีไฟล์เสียงเป็นคะแนน pronunciation และไม่ได้เพิ่ม auto-retry เพื่อซ่อนความล้มเหลว

ผู้ใช้ฟังตัวอย่าง diagnostic และหนังสือที่ความเร็วเดิมแล้วตอบว่า **อ่านตรงข้อความทั้งสองตัวอย่าง** จากนั้นขอให้ช้าลง
ตัวอย่างใหม่ที่ -15% สร้างได้แล้ว: punctuation ยาว 3.336 s, หนังสือยาว 2.712 s; ข้อมูลนี้ยืนยันความยาวไฟล์ ไม่ใช่คะแนนความถูกต้องจากผู้ฟังรอบใหม่
ยังไม่มีผลประเมินการออกเสียงครบทุก acceptance case จึงไม่อ้างว่าทุกคำหรือวรรณยุกต์ถูกต้องสมบูรณ์ และไม่ใช้ ASR แทนผู้ฟังโดยไม่ระบุข้อจำกัด

ตัวอย่าง local (ไม่ commit ไฟล์เสียง): output/edge_tts/punctuation_slow.wav, book_slow.wav, proper_name.wav, compound_vowel.wav, digits.wav, tones.wav, english.wav
ผลดิบและข้อความแต่ละขั้นอยู่ใน [edge-audit-results.json](edge-audit-results.json) เก็บ error รอบแรกควบคู่ผล retry ไม่ลบทิ้ง

## การแยกต้นเหตุ

| ชั้น | สิ่งที่ยืนยันได้ | ข้อจำกัด |
| --- | --- | --- |
| Braille decoding / reconstruction | baseline และ exact-text cases ปัจจุบันตรง expected; ไม่เพิ่ม dictionary/rule แก้เฉพาะคำ | replay จาก saved cells ไม่ใช่การวัด detection ใหม่; ภาพต้นฉบับบาง diagnostic ไม่ได้แนบครบ |
| pre-TTS / transport | ตัวอักษร/ลำดับตรงกันใน 54 cases; symbol speech labels เป็นกฎเดิมและ UI ไม่เปลี่ยน | ไม่รับประกันข้อความนอกชุดทดสอบ |
| Edge synthesis | ไทย/อังกฤษสร้าง PCM WAV ได้; ผู้ใช้ยืนยันความตรงของสองตัวอย่างเดิม | NoAudioReceived ไม่แน่นอน; ยังไม่ได้ฟังทุกคำ/ตัวเลขทุกบริบท |
| controller / playback | real worker อ่านซ้ำ/หยุดสองรอบและจบรอบเล่นหนึ่งรอบสำเร็จบน Windows; context/history ไม่เปลี่ยน | รอ WAV พร้อมเล่นเกือบ 12 s ในรอบ worker จริง ไม่รับรอง latency; ยังไม่มีบอร์ดให้ทดสอบ |

## Regression และไฟล์ที่แก้

baseline 49 tests ผ่านก่อนแก้; tests ของ Edge เริ่มจาก 4 errors เพราะยังไม่มี backend/API และ launcher test ล้มเหลวก่อนเพิ่มคำสั่ง Edge
test ของ rate ตรวจพบการไม่ส่ง -15% ก่อนเพิ่มค่าใหม่จริง ผลชุด regression สุดท้ายผ่าน **150 tests** ใน 22.557 s และ git diff --check ผ่าน บันทึกในผลดิบ
ครอบคลุมเสียงตามภาษา, ไม่ต้องมี local model, network error, cleanup partial MP3, PCM WAV, exact reference text, original symbol labels, launcher/menu และ regression เดิมของ Thai/English/UI

- piper_tts.py: เพิ่ม Edge ใน worker/controller เดิมและ config เสียง/อัตราอ่าน
- camera_reader.py: เลือก backend Edge ได้โดยไม่มีไฟล์ model; ใช้ปุ่มเดิม
- tools/launch.py: คำสั่ง edge และเมนูข้อ 6 ตรวจ environment ก่อนเปิด ไม่มี auto-install
- requirements-edge-tts.txt, .gitignore, README.md และเอกสารนี้
- tests/test_edge_tts.py, tests/test_main_launcher.py, tests/fixtures/thai_hello_cells.json

## Dependencies และขอบเขตบริการ

edge-tts 7.3.1: LGPLv3 (ส่วน srt_composer เป็น MIT), soundfile 0.13.1: BSD-3-Clause; runtime ที่ทดลองคือ Windows x64/Python 3.12.13
ไลบรารี edge-tts เป็นโครงการภายนอกที่เรียกบริการ Microsoft Edge ไม่ใช่ Azure Speech API ทางการ การไม่ต้องใช้ key ไม่ใช่คำรับประกันโควตา/SLA หรือสิทธิ์ใช้งานบริการทุกรูปแบบ

- [edge-tts README/API](https://github.com/rany2/edge-tts) · [license](https://github.com/rany2/edge-tts/blob/master/LICENSE)
- [เสียงภาษาไทย Microsoft](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts)
- [soundfile](https://pypi.org/project/soundfile/0.13.1/)


## ลดเวลาเริ่มอ่าน (2026-10-15)

ผลก่อนแก้พบว่า worker รอ MP3 ทั้งชุดและแปลง WAV จบก่อนเล่น; S ซ้ำยังยกเลิกงานที่รอและเริ่มสังเคราะห์ใหม่ ปัจจุบัน Edge เล่น PCM ระหว่างรับ MP3 ผ่าน miniaudio 1.71 (MIT) ใน worker เดิม ไม่บล็อก scanner และไม่เปลี่ยนข้อความ เสียง หรือความเร็ว

ติดตั้ง dependency เพิ่มใน Edge environment ด้วย `python -m pip install -r requirements-edge-tts.txt` โดยใช้ Python ของ environment นั้น การอ่านซ้ำใช้ WAV ล่าสุดที่สร้างสำเร็จในหน่วยความจำของ session; key รวมข้อความ ภาษา backend model voice และ rate ไม่ใช้ไฟล์ที่หยุดกลางทาง กด S ขณะคำขอเดิมกำลังรอจะคง worker เดิม; B หยุด worker และล้างไฟล์งาน โดยเก็บ cache ที่สมบูรณ์ไว้ให้อ่านซ้ำ

ก่อนแก้ข้อความยาวรับ MP3 ครบที่ 22.334 s จึงเริ่มเตรียมเล่นได้ หลังแก้รอบสุดท้ายส่ง PCM ให้ตัวเล่นที่ 0.859 s และรับครบที่ 11.149 s (คนละรอบเครือข่าย จึงไม่ใช้คำนวณเปอร์เซ็นต์ความเร็ว) S คืนใน 7.4 ms, repeat cache พร้อมใน 129 ms, B คืนใน 2.7 ms และตรวจว่า worker หยุดจริง ตัวเลข PCM/ready marker ไม่ใช่เวลาที่วัดเสียงออกลำโพง

เทียบ PCM จาก MP3 เดียวกันพบความต่างสูงสุด 1 หน่วยหลังจัดแนว; ส่วนต้นที่หาย 48 ms เป็น silence ล้วนและไม่มีส่วนท้ายหาย Regression ที่เกี่ยวข้อง 155 tests ผ่าน รวม pending S, cache invalidation, B, partial WAV, ลำดับ PCM และ decoder failure ไม่ค้าง ผลดิบแบบย่ออยู่ใน [tts-latency-results.json](tts-latency-results.json)

ผลจริงนี้ทดสอบ Windows x64 เท่านั้น ยังไม่ได้ทดสอบบอร์ด Linux ARM64; network/service latency และ NoAudioReceived ยังเกิดได้ การเล่น Edge ครั้งแรกใช้ miniaudio ส่วน cache/offline ใช้ตัวเล่น WAV เดิม คำอธิบาย flow/latency ก่อนหน้านี้เป็นผล baseline ก่อนปรับ streaming
