# Braille Reader — ระบบอ่านอักษรเบรลล์ด้วยจุดสี + OpenCV

ต้นแบบระบบ **Optical Braille Recognition (OBR)** ที่ตรวจจับจุดสีบนอักษรเบรลล์
แล้วแปลงเป็นข้อความ โดยใช้ **Python** และ **OpenCV**

##  แนวคิด

อักษรเบรลล์มีจุดนูนที่กล้องจับได้ยาก → **แต้มสีบนจุดนูน** แล้วใช้ Color Segmentation ตรวจจับ

```
ภาพเบรลล์ที่แต้มสี → HSV Color Mask → Dot Detection → Grid Alignment → Decode → ข้อความ
```

##  โครงสร้างไฟล์

```
braille-reader/
├── main.py              # Entry point — CLI สำหรับอ่านภาพ
├── detector.py          # Core — ตรวจจับจุดสี + จัด grid
├── decoder.py           # แปลง dot pattern → ตัวอักษร
├── config.py            # Braille mapping + detection parameters
├── generate_test.py     # สร้างภาพทดสอบ
├── requirements.txt     # Dependencies
├── sample_images/       # ภาพทดสอบที่สร้างจาก generate_test.py
└── output/              # ผลลัพธ์ (mask, annotated images)
```

##  เริ่มต้นใช้งาน

### 1. ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

### 2. สร้างภาพทดสอบ

```bash
python generate_test.py
```

### 3. รันตัวอ่าน

```bash
# อ่านภาพที่มีจุดสีน้ำเงิน
python main.py sample_images/test_hello_blue.png

# อ่านภาพที่มีจุดสีแดง
python main.py sample_images/test_world_red.png --color red

# บันทึกภาพ debug
python main.py sample_images/test_hello_blue.png --save
```

##  Pipeline

| ขั้นตอน | ฟังก์ชัน | คำอธิบาย |
|---------|---------|---------|
| 1. Preprocess | `_preprocess()` | Gaussian Blur ลด noise |
| 2. Color Segment | `_color_segment()` | แยกสีใน HSV space |
| 3. Morph Clean | `_morph_clean()` | Close/Open morphology |
| 4. Find Dots | `_find_dots()` | Contour + filter area/circularity |
| 5. Grid Align | `_cluster_into_cells()` | จัดจุดเข้า Braille cell (2×3) |
| 6. Decode | `decode_cells()` | Lookup table → ตัวอักษร |

##  สีที่รองรับ

- 🔵 **Blue** (default) — contrast ดีที่สุดกับกระดาษขาว
- 🔴 **Red**
- 🟢 **Green**

ปรับค่า HSV range ได้ใน `config.py`

##  Braille Reference

```
Braille Cell:
  (1) (4)
  (2) (5)
  (3) (6)

ตัวอย่าง:
  a = dot 1          ⠁
  b = dots 1,2       ⠃
  h = dots 1,2,5     ⠓
  l = dots 1,2,3     ⠇
  o = dots 1,3,5     ⠕
```
