# Web API ของหน้าเว็บครู

สัญญาระหว่างหน้าเว็บ (`web/index.html`) กับ web server ที่รันคู่กับ `finger_reader.py`
ถ้าจะเปลี่ยนชื่อเส้นทางหรือ field ให้แก้เอกสารนี้และแจ้งทีมก่อน

- server ฟังที่ `127.0.0.1` ใน port ที่สั่ง (เช่น `--web 8080`) ถ้าเปิดให้เครื่องอื่นด้วย `--web-host 0.0.0.0` ต้องใส่ `--web-pin` เลข 6 หลัก
- ข้อความเป็น UTF-8 ทั้งหมด JSON ใช้ `Content-Type: application/json; charset=utf-8`

## PIN

เมื่อใช้ `--web-pin` ทุกเส้นทางยกเว้น `GET /` และ `POST /api/login` ตอบ `401` ถ้าไม่มี cookie ที่ถูกต้อง

## `POST /api/login`

body: `{"pin": "123456"}`

- ถูกต้อง: `200` พร้อม cookie `pin` อายุ 30 วัน (`HttpOnly`, `SameSite=Strict`)
- ผิด: `401` หลังรอ 1 วินาที ลองได้ครั้งละหนึ่งคำขอทั้งเครื่อง
- body ผิด: `400`

## `GET /`

ส่งไฟล์ `web/index.html`

## `GET /api/state`

หน้าเว็บเรียกทุก 0.3 วินาที server ต้องตอบเร็ว ห้ามรอเฟรมกล้อง ให้ส่ง snapshot ล่าสุดที่ลูปหลักเก็บไว้

```json
{
  "status": "reading",
  "markers": true,
  "hand": true,
  "fps": 24.6,
  "symbol": "ชอช้าง",
  "word": "ช้าง",
  "active_word": 0,
  "last_spoken": "ช้าง",
  "lines": [
    [{"index": 0, "text": "ช้าง"}, {"index": 1, "text": "กิน"}, {"index": 2, "text": "กล้วย"}],
    [{"index": 3, "text": "แมว"}, {"index": 4, "text": "นอน"}]
  ],
  "settings": {"mode": "word", "spell_max_cells": 2, "camera": true}
}
```

| field | ชนิด | ความหมาย | ที่มาในโค้ด |
|---|---|---|---|
| `status` | `"waiting_page"` \| `"learning"` \| `"reading"` | สถานะระบบ | `FingerReader.status` |
| `markers` | bool | เห็น marker ครบ 4 มุม | `reader.calibration is not None` |
| `hand` | bool | เห็นปลายนิ้วที่ใช้อ่าน | `reader.tip is not None` |
| `fps` | number | เฟรมต่อวินาทีของลูปหลัก | วัดในลูป `main()` |
| `symbol` | string \| null | ชื่อตัวอักษรใต้นิ้ว | `reader.active_cell` → `cell.name` |
| `dots` | array ของ int (1–6) | จุดเบรลล์ที่นูนในเซลล์ปัจจุบัน | `sorted(cell.dots)` |
| `word` | string \| null | คำที่นิ้วอยู่ | `page.words[cell.word_index]` |
| `active_word` | int \| null | index ของ `word` ใน `lines` ใช้ไฮไลต์ | `cell.word_index` |
| `last_spoken` | string \| null | ข้อความล่าสุดที่เล่นเสียง | `SpeechQueue.say()` |
| `muted` | bool | สถานะปิดเสียง | `speech.muted` |
| `audio_config` | object | ค่าคอนฟิกเสียง: `engine` ("edge" \| "piper"), `volume` (0–100) | `SpeechQueue` |
| `lines` | array ของบรรทัด แต่ละบรรทัดเป็น array ของ `{index, text}` | ข้อความทั้งหน้า เรียงตามบรรทัด | `reader.page.lines` + `page.words` |
| `settings` | object | ค่าตั้งปัจจุบัน ดู `POST /api/settings` | เก็บไว้ในหน่วยความจำ: ค่าจาก `panel_settings.load()` ตอนเริ่ม แล้วแทนด้วยค่าที่ `apply_settings()` คืนมา ห้ามอ่านไฟล์ทุกครั้งที่ถูกเรียก (บน Windows การอ่านไฟล์ถี่ ๆ ทำให้บันทึกค่าใหม่ไม่ได้) |

ยังไม่มีหน้า (`status` เป็น `waiting_page`): `lines` เป็น `[]` และ `symbol`, `word`, `active_word` เป็น `null`

## `POST /api/settings`

body เป็น JSON ส่งเฉพาะ field ที่จะเปลี่ยนได้

```json
{"mode": "spell"}
```

| field | ค่าที่รับ | ค่าเริ่มต้น | ผล |
|---|---|---|---|
| `mode` | `"spell"` \| `"word"` | `"spell"` | `spell`: สะกดทุกเซลล์แล้วพูดคำ · `word`: พูดเฉพาะคำ ยกเว้นคำสั้น |
| `spell_max_cells` | int 1–5 | 2 | ในโหมด `word` คำที่มีเซลล์ไม่เกินค่านี้ยังสะกด |
| `camera` | bool | true | false: server หยุดเข้ารหัส JPEG และ `/stream.mjpg` ตอบ 503 |

server เรียก `panel_settings.apply_settings(speech, path, body)` ฟังก์ชันนี้ตรวจค่า ตั้งค่าให้ `SpeechQueue` และบันทึกไฟล์

- สำเร็จ: `200` พร้อม settings ทั้งหมดหลังเปลี่ยน เช่น `{"mode": "spell", "spell_max_cells": 2, "camera": true}`
- บันทึกไฟล์ไม่ได้ (`OSError`): `500` พร้อม `{"error": "บันทึกค่าตั้งไม่ได้"}` ค่าในระบบเสียงไม่เปลี่ยน
- ค่าผิดหรือ JSON ผิด: `400` พร้อม `{"error": "<ข้อความภาษาไทย>"}` และไม่เปลี่ยนค่าใดเลย (ถ้า field หนึ่งผิด field อื่นใน body เดียวกันก็ไม่ถูกใช้)

## `POST /api/action`

สั่งการทำงานของสถานีแบบทันที (Interactive Controls)

```json
{"action": "replay"}
```

| `action` | ค่าเสริมใน body | หน้าที่ |
|---|---|---|
| `"replay"` | — | เล่นเสียงข้อความล่าสุดซ้ำอีกครั้ง |
| `"speak"` | `"text": "คำที่ต้องการพูด"` | สั่งให้ออกเสียงคำหรือข้อความที่ระบุทันที |
| `"toggle_mute"` | — | สลับสถานะปิด/เปิดเสียง |
| `"set_audio"` | `"engine": "edge" \| "piper"`, `"volume": 100` | ปรับเอนจินเสียง (เปลี่ยนแล้วเตรียมเสียงของหน้าปัจจุบันใหม่) และระดับความดัง (0–100%) |

- สำเร็จ: `200` พร้อมผลลัพธ์ เช่น `{"ok": true, "action": "set_audio"}`
- ข้อผิดพลาด: `400` เมื่อ JSON ผิด หรือ action ไม่ถูกต้อง

- สำเร็จ: `200` พร้อมผลลัพธ์ เช่น `{"ok": true, "action": "replay"}`
- ข้อผิดพลาด: `400` เมื่อ JSON ผิด หรือ action ไม่ถูกต้อง

## `GET /stream.mjpg`

ภาพจากกล้องพร้อมโครงมือและกรอบเซลล์ (ภาพเดียวกับหน้าจอ dev) แบบ MJPEG

- `Content-Type: multipart/x-mixed-replace; boundary=frame`
- ไม่เกิน `--web-fps` (ค่าเริ่มต้น 5, รับ 1–30) ความกว้างไม่เกิน 960 px
- เข้ารหัส JPEG เฉพาะเมื่อ `camera` เป็น true และมีผู้ชมอย่างน้อยหนึ่งคน
- `camera` เป็น false: ตอบ `503`

## `GET /api/connect` และ `GET /qr.png`

ใช้กับการ์ด "เชื่อมต่อจากมือถือ" บนจอของสถานีเท่านั้น ตอบเฉพาะคำขอจาก `127.0.0.1` หรือ `::1`
และไม่ต้องใส่ PIN เครื่องอื่นได้ `404` เสมอ จึงอ่าน PIN จากเครื่องอื่นไม่ได้

```json
{"network": true, "url": "http://172.20.10.3:8080", "pin": "482913"}
```

| field | ความหมาย |
|---|---|
| `network` | เปิดให้เครื่องอื่นเข้าไหม (`--web-host` ไม่ใช่ `127.0.0.1`) |
| `url` | ที่อยู่สำหรับมือถือ ใช้ IP ในวง WiFi หรือ hotspot ตอนนี้ (`null` ถ้าไม่ได้เปิดหรือไม่มีเครือข่าย) |
| `pin` | PIN ของสถานี (`null` ถ้าไม่ได้ตั้ง) |

`/qr.png` คือ QR code ของ `url` เป็น PNG สร้างด้วย OpenCV ไม่ใช้อินเทอร์เน็ต ไม่มี `url` จะตอบ `404`
หน้าเว็บถาม `/api/connect` ทุก 5 วินาที ถ้า hotspot แจก IP ใหม่ QR จะเปลี่ยนตาม ไม่ใส่ PIN ไว้ใน QR

## ทดสอบหน้าเว็บโดยไม่มี server

เปิด `web/index.html` ตรงจากไฟล์ หรือเพิ่ม `?demo` ท้าย URL หน้าเว็บจะใช้ข้อมูลจำลองแทนการเรียก API
