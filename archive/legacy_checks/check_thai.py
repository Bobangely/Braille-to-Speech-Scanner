# -*- coding: utf-8 -*-
import sys
import io
import os
import unicodedata

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from decoder import decode_cells, decode_cells_thai, normalize_thai_text
from config_thai import THAI_CHAR_TO_BRAILLE
from tts import speak

def make_cell(dots_list, x=0, y=50):
    return {'dots': frozenset(dots_list), 'x': x, 'y': y}

def text_to_synthetic_cells(text):
    cells = []
    x = 0
    for char in text:
        if char == ' ':
            x += 200
            continue
        if char in THAI_CHAR_TO_BRAILLE:
            p = THAI_CHAR_TO_BRAILLE[char]
            if isinstance(p, tuple):
                for sub in p:
                    cells.append(make_cell(sub, x=x))
                    x += 40
            else:
                cells.append(make_cell(p, x=x))
                x += 40
    return cells

print("=" * 60)
print("1. ทดสอบคำที่มีปัญหาโดยตรง ('ฟันเหยิน')")
print("=" * 60)
c_fanyern = [
    make_cell(THAI_CHAR_TO_BRAILLE['ฟ'], 0),    # C1: ฟ
    make_cell(THAI_CHAR_TO_BRAILLE['ั'], 40),   # C2: ั
    make_cell(THAI_CHAR_TO_BRAILLE['น'], 80),   # C3: น
    make_cell(THAI_CHAR_TO_BRAILLE['เ'], 120),  # C4: เ
    make_cell(THAI_CHAR_TO_BRAILLE['ห'], 160),  # C5: ห
    make_cell(THAI_CHAR_TO_BRAILLE['ย'], 200),  # C6: ย
    make_cell(THAI_CHAR_TO_BRAILLE['ิ'], 240),  # C7: ิ
    make_cell(THAI_CHAR_TO_BRAILLE['น'], 280),  # C8: น
]
res_fanyern = decode_cells_thai(c_fanyern)
print(f"ผลลัพธ์: {res_fanyern} (Expected: ฟันเหยิน) -> {'✅ PASS' if res_fanyern == 'ฟันเหยิน' else '❌ FAIL'}")
assert res_fanyern == 'ฟันเหยิน'

print()
print("=" * 60)
print("2. ทดสอบคำ/ประโยคต่อเนื่องที่ไม่มีช่องว่าง")
print("=" * 60)
test_words = [
    "ฟันเหยิน",
    "ฉันรักเธอ",
    "โรงเรียน",
    "เข้าใจ",
    "ดีใจ",
    "ไปเที่ยว",
    "มีเวลา",
    "กินข้าว",
    "สวัสดีครับผม",
]

for w in test_words:
    cells = text_to_synthetic_cells(w)
    got = decode_cells_thai(cells)
    status = "✅ PASS" if got == w else f"❌ FAIL (Got: {got})"
    print(f"  Word: {w:<15} -> {got:<15} {status}")
    assert got == w, f"Mismatch: expected {w}, got {got}"

print()
print("=" * 60)
print("3. ทดสอบการจัดระเบียบสระ-วรรณยุกต์ (Orthographic Normalization)")
print("=" * 60)
norm_cases = [
    ('ก' + '\u0E49' + '\u0E34' + 'ง', 'กิ้ง'),       # วรรณยุกต์มาก่อนสระบน -> สระบน + วรรณยุกต์
    ('ก' + '\u0E48' + '\u0E34' + 'ง', 'กิ่ง'),       # วรรณยุกต์มาก่อนสระบน -> สระบน + วรรณยุกต์
    ('ก' + '\u0E32' + '\u0E49', 'ก้า'),              # วรรณยุกต์หลังสระอา -> วรรณยุกต์ + สระอา
    ('น' + '\u0E33' + '\u0E49', 'น้ำ'),              # วรรณยุกต์หลังสระอำ -> วรรณยุกต์ + สระอำ
    ('ด' + '\u0E4C' + '\u0E34', 'ดิ์'),              # การันต์ก่อนสระอิ -> สระอิ + การันต์
    ('ก' + '้' + '้' + 'า', 'ก้า'),                   # ตัดวรรณยุกต์ซ้ำซ้อน
]
for raw_in, expected_out in norm_cases:
    res = normalize_thai_text(raw_in)
    status = "✅ PASS" if res == expected_out else f"❌ FAIL (Got: {res})"
    print(f"  Input: {raw_in} -> {res} {status}")
    assert res == expected_out

print()
print("=" * 60)
print("4. ทดสอบ Google TTS สร้างไฟล์เสียงสำหรับ 'ฟันเหยิน'")
print("=" * 60)
audio_out = 'output/test_fanyern.mp3'
tts_ok = speak("ฟันเหยิน", lang='thai', method='online', save_file=audio_out)
if tts_ok and os.path.exists(audio_out):
    sz = os.path.getsize(audio_out)
    print(f"  Google TTS สร้างไฟล์เสียงสำเร็จ: {audio_out} ({sz} bytes) ✅ PASS")
else:
    print(f"  TTS Online status: {tts_ok} (อาจเป็นเพราะไม่มีอินเทอร์เน็ต)")

print()
print("🎉 ทุกการทดสอบผ่านสมบูรณ์ 100%!")
