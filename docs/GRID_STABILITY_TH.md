# วิเคราะห์และแก้ Grid กระตุกระหว่างสแกน

ตรวจจาก `main` ที่ `5c6b23c` และแก้บน `codex/stable-braille-grid` วันที่ 14 กันยายน 2026

## เส้นทางที่ตรวจ

`ThreadedCameraCapture → เฟรมล่าสุด → zoom/crop → AsyncBrailleWorker → sharpening ตามค่า → YOLOBrailleDetector.detect → ColoredCellStream` หรือ `ColoredRoiReader → ColoredCellStream → plan_cells/read_cell/pair_markers → decode_cells/verbose → ผลดิบ → RealTimeBrailleScanner._render_frame → LivePreview → หน้าจอ`

ตรวจ call sites ใน main, กล้อง, detector, color/ROI, geometry, Thai decoder, motion, preview, diagnostics/export, tests และ launcher ทั้งหมดที่เกี่ยวข้อง โหมดสีไม่ได้ใช้ YOLO จำแนกอักษร; YOLO เสนอ ROI เฉพาะ `--roi yolo` เท่านั้น

## สาเหตุหลักที่ยืนยันได้

1. `_line_rows()` เดิมใช้ความคลาดเคลื่อนของระยะแถว 15% ทุกกรณี แถวที่แต้มสีแล้ว centroid ไม่สม่ำเสมออาจถูกแบ่งเป็นหลายบรรทัด จึงจัด cell ใหม่และเกิด `ambiguous_row_grid` ตัวอย่างสังเคราะห์เลื่อน centroid แถวล่างทำให้จาก 4 cell กลายเป็น 6/8 cell ได้
2. `plan_cells()` คำนวณ spacing, angle, line_id และ line_cell_index ใหม่จากข้อมูลเฟรมเดียว ส่วน `LivePreview` ใช้ optical flow เพื่อขยับ overlay ระหว่างเฟรม แต่แทน Grid ทั้งชุดเมื่อ result_id เปลี่ยน ไม่มีการจับคู่ geometry ข้ามผลอ่าน
3. `_render_frame()` เดิมใช้ 6 ผลตรงกันเพื่อแสดง LOCKED เท่านั้น ข้อความและอักษรเหนือ cell ยังแสดงผลเฟรมล่าสุดทันที จึงเห็นข้อความเปลี่ยนก่อนยืนยัน
4. `LivePreview` สร้าง overlay สถานะรอขึ้นใหม่ทุกผลอ่าน แม้ขนาด/ภาษา/สีไม่เปลี่ยน เป็นงานวาดที่ไม่จำเป็นและกระทบเวลา preview

## สาเหตุรองและข้อจำกัดของหลักฐาน

- HSV มี threshold คงที่ และ component filters ใช้ขนาด/รูปร่าง/พื้นที่ ภาพเบลอ หมึกจาง หรือแสงเปลี่ยนอาจทำให้จุดติด–หลุดได้ ยังไม่ปรับ threshold โดยเดาจาก screenshot
- screenshot แสดง SHARP LV.2 แม้ Python CLI ตั้ง default 0 เพราะ launchers หลักยังส่ง `--sharp 2` อยู่ แก้ launchers ให้เริ่ม 0 ตรงกับค่า CLI; ปุ่ม E ปรับความคมได้ตามเดิม
- YOLO bbox/confidence อาจเปลี่ยน ROI ในโหมดที่เปิด YOLO แต่ไม่ใช่ตัวอ่าน pattern ในเส้นทางสี จึงยังยืนยันไม่ได้ว่า screenshot นี้มี YOLO jitter
- กล้องใช้ latest-frame queue อยู่แล้ว จึงอาจข้ามเฟรมระหว่าง worker ทำงานโดยตั้งใจ ไม่มีการเพิ่ม queue สะสมหรือ inference ซ้ำในการแก้ครั้งนี้
- ภาพ screenshot มี overlay และไม่ใช่หลักฐานต่อเนื่องตามเวลา จึงไม่ใช้เป็น raw input สำหรับเทียบ accuracy หรือสรุปว่าการแต้มสีของบรรทัดนั้นผิด

## การแก้

- `yolo_cell_stream.py`: กรณีมองเห็น 3 แถวเท่านั้น ใช้ระยะรวมบน–ล่างประเมิน pitch ของบรรทัด และยอมรับ centroid ที่คลาดเล็กน้อยอย่างมีขอบเขต ไม่ผ่อนเกณฑ์ทุกช่องว่าง เพราะจะรวมบรรทัดข้างเคียงที่มีจุดไม่ครบ การทดสอบบรรทัดกำกวมเดิมยังต้องผ่าน
- `grid_stability.py`: จับคู่ cell แบบ mutual nearest-neighbour หลังชดเชย camera motion ทั้ง 6 slots ต้องอยู่ใกล้กันพอ แล้วใช้ 75% ของ geometry ที่ติดตามและ 25% ของค่าที่วัดใหม่ ไม่หน่วงการเลื่อนกล้องทั้งภาพและไม่จับคู่จาก index อย่างเดียว
- กำหนด track_id ให้เซลล์ที่จับคู่กัน หมายเลขของ cell ที่ยังอยู่ไม่เลื่อนเมื่ออีก cell หายไป ไม่แสดงหรือสร้าง dot ของเซลล์ที่หาย การกลับเข้าฉากหลังติดตามหลุดเริ่ม track ใหม่ได้
- ล้างสถานะ geometry เมื่อไม่มีเซลล์, เปลี่ยนสี/ภาษา/ROI/context, ติดตามภาพไม่ได้ หรือผลขาดช่วงเกิน 0.75 วินาที เก็บเฉพาะ geometry/ภาพ tracking ของผลล่าสุด ไม่สะสมเฟรม
- `live_preview.py`: แยก cache ของ geometry ออกจากข้อความ เพื่อไม่ smoothing ผลเดียวซ้ำตาม display FPS และ reuse waiting footer เมื่อขนาด/ภาษา/สีเดิม
- `camera_reader.py`: ข้อความหลักและป้ายอักษรเหนือ cell แสดงเมื่อผลใหม่ตรงกันครบ stability_threshold เท่านั้น ระหว่างรอมี Grid กับสถานะ CONFIRMING ไม่เก็บคำก่อนหน้าไว้แสดงเป็นผลของเฟรมที่อ่านไม่ได้ ความผิดพลาด/ผลเก่า/เปลี่ยน context ยังล้างประวัติตามเดิม
- `yolo_detector.py`: ใช้ track_id สำหรับป้าย Grid บนหน้ากล้อง ถ้าไม่มี track_id เช่น image CLI หรือ diagnostics ยังคงลำดับ C1…Cn เดิม

การ smoothing นี้ทำเฉพาะ geometry ที่แสดงผล **ไม่เฉลี่ย dot bits ไม่เดาคำ และไม่เปลี่ยน raw cells/text ที่ worker ส่งออก** การแก้จำนวนบรรทัดผิดอยู่ที่ geometry ก่อน decode ส่วน D/export ยังบันทึกภาพและผลอ่านดิบเพื่อตรวจสาเหตุได้ ป้าย track บนกล้องอาจต่างจากลำดับ cell ใน diagnostic ซึ่งเป็นลำดับของแต่ละเฟรม

## การตรวจและเวลา

- Unit/regression suite ผ่าน **110 tests** รวม 7 tests ใหม่: แถวไม่สม่ำเสมอ, Grid jitter, camera translation, cell หาย, context/scene/expiry, ไม่ smoothing result ซ้ำ, confirmation และ error clearing
- YOLO benchmark ด้วย weights จริงผ่าน **99/99** กรณี รวมอักษรเล็กและหลายบรรทัด
- `git diff --check` ผ่าน และ `uv build` สร้าง source distribution/wheel ได้ แต่การตั้งค่า packaging เดิมรวมเพียง package placeholder ไม่ได้รวมโปรแกรม scanner จึงไม่ใช้ build นี้ยืนยันว่า scanner พร้อม deploy; การทำงาน scanner ตรวจด้วย tests ที่รันจาก source
- Regression recovery เดิมเพิ่มเฟรมสำเร็จให้เพียงพอสำหรับการแสดงข้อความที่ต้องยืนยัน 6 ผล; ยังคงกรณี scan error และการกู้คืนทั้งหมด
- Benchmark ใช้ภาพสังเคราะห์ชุดเดิม 36 เฟรมต่อกรณี มี centroid noise ±1.4 พิกเซลและ camera translation พร้อม warm-up 5 เฟรม ไม่รวม capture, queue wait หรือเสียง จึงเป็น processing throughput ไม่ใช่ FPS กล้องจริง

| กรณี | ก่อน | หลัง |
|---|---:|---:|
| FHD แถวปกติ: Grid step RMS บน preview | 0.834 px | 0.738 px |
| 4K แถวปกติ: Grid step RMS บน preview | 0.417 px | 0.296 px |
| FHD แถวคลาด: จำนวนครั้งที่ cell count เปลี่ยน | 21 | 0 |
| FHD แถวคลาด: จำนวน cell ที่พบ | 4/6/8 | 4 |
| FHD แถวคลาด: ข้อความตรง fixture | 19/36 | 36/36 |
| FHD ปกติ: read + preview median | 46.37 ms | 37.42 ms |
| FHD แถวคลาด: read + preview median | 45.73 ms | 37.53 ms |
| 4K ปกติ: read + preview median | 68.42 ms | 66.12 ms |
| 4K ปกติ: read + preview p95 | 69.83 ms | 71.58 ms |

Processing capacity จาก median เทียบเท่าประมาณ FHD 21.6 → 26.7 รอบ/วินาที และ 4K 14.6 → 15.1 รอบ/วินาที ไม่รับรองว่าจะได้ FPS นี้จากกล้องจริง ค่า p95 ของ 4K เพิ่มเล็กน้อย จึงไม่อ้างว่าทุกเฟรมเร็วขึ้น

ยังใช้เกณฑ์ 6 ผลใหม่เท่าเดิม แต่เปลี่ยนให้การแสดงข้อความรอยืนยันด้วย ที่ READ 20 ครั้ง/วินาที ระยะจากผลแรกถึงผลที่ 6 ประมาณ 0.25 วินาทีบวกเวลาอ่านจริง; เมื่ออ่านไม่สม่ำเสมอต้องรอเพิ่ม Grid แสดงและติดตามได้ระหว่างรอ

หลักฐานดิบอยู่ใน `scratch/grid_review_20260914/`: `benchmark_grid.py`, `before.json`, `after.json`, `yolo_regression.json` (ignored ไม่ commit)

## กล้องจริงและสิ่งที่ยังต้องทดสอบ

เปิดกล้อง index 0 ได้และเก็บ 114 เฟรมในช่วง 4 วินาทีที่ร้องขอ ที่ 1280×720 จากนั้นคืนกล้องแล้ว ภาพที่ได้เป็นโต๊ะทำงาน ไม่ใช่หนังสือ/บรรทัดที่มีอาการ จึงใช้ตรวจการรับภาพได้ แต่ **ยังใช้ยืนยันว่าอาการเฉพาะบรรทัดใน screenshot หายแล้วไม่ได้** คลิปและภาพตัวอย่างอยู่ในโฟลเดอร์ scratch เดียวกัน ไม่มีการปิดโปรแกรมกล้องอื่น

ต้องทดสอบกับหนังสือจริงหรือคลิปดิบของบรรทัดนั้นต่อ โดยลองเริ่ม `--sharp 0`, ล็อกแสง/โฟกัสถ้ากล้องรองรับ, เปรียบเทียบ ROI off/yolo และเก็บ D/diagnostics ตอนที่จุดติด–หลุด ไม่เพิ่มการฝึกโมเดลหรือเปลี่ยน threshold สีจนกว่าจะเห็นหลักฐาน raw frame
