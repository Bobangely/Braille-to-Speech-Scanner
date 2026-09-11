# รายงานแก้กล้องและ Scan Recovery

วันที่ตรวจ: 10 กันยายน 2026  
Branch: `codex/camera-scan-recovery`  
ฐานงาน: `main` ที่ commit `a90ef97` (รักษา alias `--source` ของผู้ใช้ไว้)

## ขอบเขตและ flow ที่ตรวจ

อ่านและไล่ implementation/call sites ของ entry points, camera, worker, YOLO, geometry/crop, decoder,
result rendering และ tests ที่เกี่ยวข้องร่วมกัน ก่อนแก้ไข ไม่ได้วิเคราะห์จากไฟล์กล้องเพียงไฟล์เดียว

```text
Start_Scanner*.bat / main.py / camera_reader.main()
  → RealTimeBrailleScanner.run()
  → ThreadedCameraCapture: VideoCapture → read → latest frame + frame ID
  → _apply_zoom(): crop ตาม ROI โดยรักษาพิกเซลต้นฉบับ
  → AsyncBrailleWorker: สำเนาเฟรมล่าสุด + ภาษา + sharpness + context
  → ตรวจ BGR / optional sharpening
  → YOLOBrailleDetector.detect()
      → YOLO overview / tiles
      → plan_cells() / _line_rows()
      → CellStream: crop แต่ละเซลล์ → YOLO crop batch → อ่านจุด → pair markers
  → decode_cells() + decode_cells_verbose()
      → thai_decoder หรือ English decoder
  → result: cells / text / status / reason / stage / context
  → LivePreview.render() → grid / ข้อความ → mini viewfinder → HUD
  → imshow() / waitKey() / hotkeys
  → stop worker / release camera / destroy windows
```

ตรวจ config ภาษาไทย/อังกฤษและการใช้งานผ่าน main/launcher แล้ว ไม่ต้องแก้ mapping เพื่อแก้ปัญหาการปิด session
ชุด training/dataset และ archive ไม่ได้อยู่ใน live camera flow นี้ โมเดลและชุดข้อมูลไม่ได้ถูกเปลี่ยน

## Root cause และหลักฐานก่อนแก้

| สาเหตุ | จุดเกี่ยวข้อง | หลักฐาน / วิธีแก้ |
|---|---|---|
| แยกขอบเขตแถวไม่ได้ถูกโยนเป็น ValueError ทั่วไป และกลายเป็น SCAN ERROR | `yolo_cell_stream._line_rows()`, `CellStream.iter_cells()`, `AsyncBrailleWorker._worker_loop()` | จำลองแถวกำกวมได้ และพบกับ ROI สองบรรทัดจากภาพแนบ แยก exception ชนิด UnreadableBrailleFrame ซึ่งยังเป็น ValueError subclass |
| native read exception หลุดออกจาก capture thread | `ThreadedCameraCapture._capture_loop()` | ฉีด read exception แล้ว thread เดิมจบ แก้ให้เก็บเหตุผล ล้างเฟรมเสีย และ retry แบบมี backoff |
| รับภาพไม่ได้ 300 รอบแล้วสั่งออกจาก loop | `RealTimeBrailleScanner.run()` | ก่อนแก้: read 300 ครั้ง, waitKey 0 ครั้ง, release 1 ครั้ง แก้ให้ UI ยังรับปุ่มและรอเฟรมต่อ |
| การวาด mini viewfinder ใหญ่กว่าภาพปลายทาง | `_draw_mini_viewfinder()` | ภาพ 120×80 ทำให้เกิด broadcast ValueError จริง แก้ให้ขนาดพอดีพื้นที่และคำนวณ ROI จากขนาดต้นฉบับ |
| exception ฝั่ง preprocessing/result/preview หลุดถึง finally | `run()`, `LivePreview.render()` และ HUD | เพิ่มขอบเขตต่อเฟรม พร้อม traceback และภาพ fallback ข้าม result ที่วาดเสียจนมี result ใหม่ |
| read/set/release ใช้ handle ข้าม thread และ join เพียง 1 วินาที | `set_resolution()`, `release()`, `start()` | ตรวจพบช่อง race จากโค้ด แก้ให้ capture thread เป็นเจ้าของ native handle หลังเริ่มทำงาน และรับคำขอเปลี่ยน resolution ผ่าน queue |
| start ซ้ำ / stop ขณะ inference ยังทำงาน | `AsyncBrailleWorker.start()/stop()` และ session lifecycle | ทำ start/stop ให้เรียกซ้ำได้อย่างปลอดภัย ปฏิเสธการเริ่ม worker/session ใหม่ขณะตัวเก่ายังไม่จบ และห้ามเผยแพร่ผลหลัง stop |
| ข้อความไทยใน LOCKED HUD ผ่าน OpenCV Hershey font | `_draw_top_hud()` | เปลี่ยนเฉพาะข้อความ Unicode ให้ใช้ Pillow และฟอนต์ไทยเดิมของ detector |

ภาพ screenshot เต็มเมื่อนำมารันใหม่ **ไม่เกิด exception เดิม** แต่ให้ข้อความที่อ่านได้เพียงบางส่วน
จึงยังไม่ทราบ traceback ของเหตุการณ์เดิมจากภาพเพียงอย่างเดียว การทดสอบ ROI เป็นหลักฐานของเส้นทาง error
ที่เกิดได้จริง ไม่ใช่การยืนยันว่าใช้เฟรมและพารามิเตอร์ตรงกับเหตุการณ์เดิมทั้งหมด

## การจัดการผลอ่านและเครื่องหมาย

- `ok`: มีข้อความที่ decoder ไม่แจ้งความไม่แน่นอน จึงนำเข้าประวัติความนิ่งได้ สถานะนี้ไม่ได้รับประกันว่าคำถูกต้องกับหนังสือ
- `empty`: ตรวจไม่พบข้อความเบรลล์ อ่านเฟรมใหม่ต่อ
- `unreadable`: กำหนดกริดไม่ได้ / จำนวนเซลล์เกิน budget / input ไม่ถูกต้อง มี reason และ stage
- `uncertain`: มีเซลล์ว่างจาก crop หรือสัญลักษณ์ไม่ครบ เก็บ `�` และคำเตือนตาม decoder เดิม
- `error`: exception จริงของ preprocessing/YOLO/decoder มี traceback ระบุขั้นตอนและ frame ID แล้วลองเฟรมใหม่

ไม่ได้แทน `?` หรือเดาตัวอักษรด้วย hardcode: `?` ยังเป็นเครื่องหมายคำถามภาษาอังกฤษที่ถูกต้องได้
ส่วนข้อความไทยใน HUD ใช้เส้นทางวาด Unicode ผลที่มีความไม่แน่นอนไม่ถูกยืนยันเป็น LOCKED

## เหตุใด scan error จึงไม่สั่งปิดกล้อง

worker ประมวลผลทีละเฟรมและเก็บ pending เพียงเฟรมล่าสุด ผลอ่านไม่ออกและ exception จบที่ผลของเฟรมนั้น
ไม่มีคำสั่ง reopen/release camera จากผล OCR การวาดหน้าจอที่เสียก็ถูกแยกจาก session และรายงานสาเหตุไว้
UI ยังคงเรียก waitKey ทั้งเมื่อมีภาพและไม่มีภาพ

การ reconnect เกิดเฉพาะ **capture อ่านภาพล้มเหลวต่อเนื่อง**: รออย่างน้อย 3 วินาทีก่อนลองเปิดใหม่
จำกัด 3 ครั้งต่อช่วงขัดข้อง ถ้าหมดจำนวนครั้ง หน้าต่างแสดง CAMERA UNAVAILABLE และยังรับ Q/ESC ได้
เมื่ออ่านเฟรมดีได้อีกครั้งจะเริ่มนับรอบขัดข้องใหม่ ไม่มีการเปิดกล้องใหม่ทุก scan error

เฟรมเดิมที่ไม่มีการอัปเดตเกิน 2 วินาทีจะไม่ถูกแสดงเป็นภาพสด ผลอ่านคนละภาษา/ROI/ช่วง capture
และผลเก่าขณะ AI ค้างเกิน 5 วินาทีจะไม่ถูกนำไป lock ข้อความปัจจุบัน
งาน inference ที่ใช้เวลานานยังมีเพียงงานเดียว ไม่มีการยิงงานซ้ำขนานเพื่อแก้ timeout

## ไฟล์ที่เปลี่ยน

- `camera_reader.py`: capture ownership/retry, lifecycle, worker status, frame isolation, UI recovery และ Unicode HUD
- `yolo_cell_stream.py`: exception สำหรับเฟรมที่ระบุกริดไม่ได้ โดยรักษาการจับ ValueError ของผู้เรียกเดิม
- `tests/test_camera_recovery.py`: fault injection และทดสอบ flow ด้วย worker/preview จริงโดยจำลองกล้องและหน้าต่าง
- `tests/test_review_regressions.py`: ปรับ fake capture ให้รองรับการคืน handle และจำกัด retry ในการทดสอบ
- `docs/CAMERA_SCAN_RECOVERY_TH.md`: รายงานนี้

## ผลตรวจหลังแก้

- `python -m unittest discover -s tests -v`: **64 tests ผ่าน**
- ครอบคลุมอ่านภาพเสียมากกว่า 300 ครั้งแล้วฟื้น, read exception, empty frame, preprocessing/model/decoder error,
  ภาพกำกวมแล้วกลับมาอ่านได้, duplicate start, pending frame ownership, stop ขณะ native read/inference ยังไม่จบ,
  queued resolution, stale frame/result, bounded reconnect, setup cleanup, window close และ snapshot/preview failure
- ทดสอบ flow โดยใช้ worker และ LivePreview จริงกับ detector จำลอง: หลายเฟรม error แล้วอ่านสำเร็จ
  ใช้ camera session และ worker เดิมเพียงตัวเดียว
- YOLO โมเดลจริงกับ synthetic reference fixtures: **99/99 ผ่าน** เป็น regression ของภาพสังเคราะห์ ไม่ใช่ความแม่นยำกล้องจริง
- โมเดลจริงบนภาพแนบ → ROI สองบรรทัด → ROI เบลอ → ภาพแนบอีกครั้ง:
  ได้สถานะ uncertain → unreadable → empty → uncertain และ worker ยังทำงาน
- compileall, CLI --help และ git diff --check ผ่าน; alias --camera/--source ยังอยู่
- รายงานดิบเฉพาะเครื่องอยู่ใน `scratch/camera_recovery_benchmark.json`
  และ `scratch/camera_recovery_image_diagnostics.json` ไม่ได้เขียนทับผล benchmark เดิมที่ติดตามใน Git

## สิ่งที่ยังต้องทดสอบ

ยังไม่ได้เปิด webcam จริงหรือทดสอบบนบอร์ด Redxa ในงานนี้ ต้องทดสอบใช้งานต่อเนื่อง, ปิดหน้าต่าง/กด Q,
เปลี่ยน resolution และ zoom ขณะอ่าน, ถอดเสียบกล้องระหว่างใช้งาน รวมถึงภาพเบลอ/แสงสะท้อนและหนังสือจริง

ข้อจำกัดของ thread: ถ้า native camera driver หรือ YOLO ค้างภายในฟังก์ชัน Python ไม่สามารถยกเลิกได้อย่างปลอดภัย
การ stop จะรอ 1 วินาทีแล้วแจ้งว่ายังรอเจ้าของทรัพยากร ไม่แย่ง release handle และไม่เริ่ม worker ซ้อน
เจ้าของเดิมจะคืนทรัพยากรเมื่อ native call กลับมา จึงไม่อ้างว่าครอบคลุม driver ที่ค้างถาวร
การเปิด camera handle ครั้งแรกยังใช้ backend เดิมและอาจได้รับผลจากเวลาตอบสนองของ driver

ทดลองจากโฟลเดอร์โปรเจกต์ใน PowerShell:

```powershell
.\.venv\Scripts\python.exe camera_reader.py --camera 0 --detector yolo --yolo-pipeline stream --res fhd --sharp 0 --lang thai
```

หากยังเกิด SCAN ERROR ให้ดู traceback ที่มี frame ID และขั้นตอนใน console ซึ่งช่วยแยกได้ว่าพังที่
preprocessing, detection, decoding หรือ preview โดยไม่ต้องเดาจากข้อความบนภาพเพียงอย่างเดียว

