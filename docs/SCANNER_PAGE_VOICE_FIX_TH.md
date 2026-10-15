# ตรวจข้อความและเสียงจากภาพพยัญชนะ

Branch `feature/thai-offline-tts` เริ่มจาก tree เดียวกับ `main` (`dcf146b`)
ไม่มีการแก้ main, push หรือ merge กลับ และเก็บไฟล์ untracked เดิมไว้

## บรรทัด 2

- ภาพ `codex-clipboard-4f9ba141-e516-4ee0-8f70-fcd992d197f9.png` แสดง 63 cells / 194 dots / 4 lines
- อ่าน patterns C18–C40 จากจุดและ overlay ในภาพ เก็บ provenance/SHA-256 ใน fixture
- ท้ายบรรทัด `[5][345]` ขาด mapping วงเล็บปิด จึงกลายเป็น `ํ` กับ `ั` และถูกประกอบต่อท้าย “แล้ว”
- เพิ่ม `close_parenthesis` ใน `config_thai.py` ตามโครงสร้างวงเล็บเปิดเดิม พร้อมชื่อเสียงจาก config เดียวกัน
- ยืนยัน `(...)` จาก [Liblouis character definitions](https://github.com/liblouis/liblouis/blob/master/tables/en-ueb-chardefs.uti#L114) ซึ่ง [ตารางไทย](https://github.com/liblouis/liblouis/blob/master/tables/th-g1.utb#L45) include อยู่ ไม่เปลี่ยนมาตรฐานหรือ parser
- หลังแก้ patterns เดิมถอดเป็น `(ปัจจุบันเลิกใช้แล้ว)` และข้อความ dashboard ตรงกัน ทดสอบ detection สังเคราะห์ทั้งแดง/น้ำเงิน รวมคู่ที่ห้ามข้ามบรรทัด ช่องว่าง และเซลล์ไม่สมบูรณ์

ยังไม่มีภาพกล้องต้นฉบับ/diagnostic ของภาพนี้ จึงไม่อ้างว่าได้วัดความแม่นยำ detection จากภาพต้นฉบับใหม่

## ฃ ไม่ถูกพูด

- NFC/normalization และ JSON ส่งต่อ U+0E03 ครบ ไม่มี queue แยกคำในเส้นทาง S
- เรียก Edge โดยตรง: `ฃ` ไม่ได้ audio; `ฃ พยัญชนะตัวที่ 3` ได้ audio แต่ word metadata เริ่มที่ “พยัญชนะ” ต่างจาก `ข` ที่คืนเสียงและ metadata ของตัวอักษร
- `edge_stream.prepare_edge_text` ใช้ชื่อมาตรฐาน “ขอ ขวด” เฉพาะ ฃ ที่อยู่เดี่ยวและเสียงไทย Edge ทั้ง streaming และ export WAV ใน `piper_tts.py`
- ไม่เปลี่ยน UI, ข้อความต้นฉบับ, ฃ ภายในคำ/ติดสระหรือวรรณยุกต์, พยัญชนะอื่น หรือ local Piper/MMS
- ทดลองออนไลน์ได้ audio ของชื่ออ่านและ metadata “ขอ”, “ขวด” ในส่วนแรก แต่คำขอหน้าทั้งชุดยังมี service error ไม่ถือว่าผ่านการฟังครบหน้า และยังไม่มีผู้ฟังประเมินชื่ออ่านใหม่

## เวลาเริ่มพูด

- โค้ดก่อนงานนี้มี streaming/cache และกัน S ซ้ำระหว่าง pending อยู่แล้ว ไม่โหลดโมเดล local ใน Edge
- หน้าอ้างอิงที่ถอดข้อความจากภาพ: ก่อนแก้ first MP3 3.616 s, first PCM 3.885 s, network จบ 11.674 s คอขวดอยู่ก่อนบริการส่ง audio แรก ไม่ใช่ UI/queue
- ทดลองแบ่งคำขอตามบรรทัด แต่ไม่สำเร็จครบหน้า พบ `NoAudioReceived` และเรียกข้อความเดียวกันตรงไป Edge ก็ล้มเหลวได้ จึงถอดการทดลองนี้ออก ไม่เพิ่ม retry/fallback และไม่อ้างว่า latency ดีขึ้น
- หลังแก้รอบ whole-page อีกครั้งยังเกิด service error จึงไม่มี before/after ของหน้าที่เทียบกันได้
- ทดสอบจริงข้อความสั้น “เขียดเขียวเกาะขอน”: S คืน 7.7 ms, first PCM 0.893 s, cached repeat พร้อม 0.129 s, B คืน 2.2 ms; PID คงเดิมเมื่อ S ซ้ำ, worker หยุดและไฟล์งานถูกล้าง, อ่านซ้ำหลัง B ใช้ cache
- เวลา PCM คือส่งข้อมูลให้ playback callback; cache คือ ready marker ไม่ใช่เวลาวัดเสียงจากลำโพง เครือข่ายต่างรอบไม่ใช้คำนวณเปอร์เซ็นต์ความเร็ว

ผลเต็มแบบย่อ: [scanner-three-issues-results.json](scanner-three-issues-results.json)

## ตรวจซ้ำ

`python -X utf8 -m unittest discover -s tests -q` ผ่าน 442 tests หลังแก้ รวม Thai/English parser, สระ/วรรณยุกต์, UI/camera, offline TTS, S/B/cache และ network error; build wheel/sdist ผ่าน

เปิด `main.bat edge` แล้วใช้ภาพกล้องเดิม กด D เก็บ raw diagnostic, ตรวจท้ายบรรทัดเป็น `)` จากนั้นกด S, S ซ้ำ, B และ S หลังหยุด ตรวจว่าชื่อ “ขอ ขวด” ไม่หายและข้อความบนจอคงเดิม ไม่ต้องเพิ่ม dependency

ยังต้องทดสอบเสียงครบหน้ากับผู้ฟังและบนบอร์ด ARM64; ปัญหา latency/NoAudioReceived ของบริการยังเหลือ
