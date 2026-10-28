# Web API ของหน้าเว็บครู

สัญญาระหว่างหน้าเว็บ (`web/index.html`) กับ web server ที่รันคู่กับ `finger_reader.py`
ถ้าจะเปลี่ยนชื่อเส้นทางหรือ field ให้แก้เอกสารนี้และแจ้งทีมก่อน

- server ฟังทุก interface ใน port ที่สั่ง (เช่น `--web 8080`) ใช้ในวง WiFi เดียวกัน ไม่มีรหัสผ่าน
- ข้อความเป็น UTF-8 ทั้งหมด JSON ใช้ `Content-Type: application/json; charset=utf-8`

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
| `word` | string \| null | คำที่นิ้วอยู่ | `page.words[cell.word_index]` |
| `active_word` | int \| null | index ของ `word` ใน `lines` ใช้ไฮไลต์ | `cell.word_index` |
| `last_spoken` | string \| null | ข้อความล่าสุดที่เล่นเสียง | `SpeechQueue.say()` |
| `lines` | array ของบรรทัด แต่ละบรรทัดเป็น array ของ `{index, text}` | ข้อความทั้งหน้า เรียงตามบรรทัด | `reader.page.lines` + `page.words` |
| `settings` | object | ค่าตั้งปัจจุบัน ดู `POST /api/settings` | `panel_settings.load()` |

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
- ค่าผิดหรือ JSON ผิด: `400` พร้อม `{"error": "<ข้อความภาษาไทย>"}` และไม่เปลี่ยนค่าใดเลย (ถ้า field หนึ่งผิด field อื่นใน body เดียวกันก็ไม่ถูกใช้)

## `GET /stream.mjpg`

ภาพจากกล้องพร้อมโครงมือและกรอบเซลล์ (ภาพเดียวกับหน้าจอ dev) แบบ MJPEG

- `Content-Type: multipart/x-mixed-replace; boundary=frame`
- ไม่เกิน 5 fps ความกว้างไม่เกิน 960 px
- เข้ารหัส JPEG เฉพาะเมื่อ `camera` เป็น true และมีผู้ชมอย่างน้อยหนึ่งคน
- `camera` เป็น false: ตอบ `503`

## ทดสอบหน้าเว็บโดยไม่มี server

เปิด `web/index.html` ตรงจากไฟล์ หรือเพิ่ม `?demo` ท้าย URL หน้าเว็บจะใช้ข้อมูลจำลองแทนการเรียก API
