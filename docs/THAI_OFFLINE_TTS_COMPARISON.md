# ทดลองเสียงไทยแบบ offline

วันที่ 2026-10-15 · branch `feature/thai-offline-tts` · เริ่มจาก `eb715e9`

## ผลและขอบเขต

ทดลอง Piper เดิม, Vachana หญิง/ชาย และ MMS Thai ด้วยข้อความไทยสามชุดเดียวกันสำเร็จบน Windows โดยห้าม socket connection ระหว่าง import/load/synthesis และใช้ไฟล์โมเดล local เท่านั้น
เชื่อม **MMS เป็นตัวเลือกทดลอง** เข้าปุ่มอ่าน/อ่านซ้ำ S และหยุด B เดิม เสียงอังกฤษยังเป็น Piper และค่าเริ่มต้นยังเป็น Piper ทั้งสองภาษา
ยังไม่เลือกผู้ชนะด้านความชัด/วรรณยุกต์/ความเป็นธรรมชาติ เพราะไม่มีคะแนนจากผู้ฟัง และยังไม่มีบอร์ดให้เข้าถึง จึงไม่ประกาศว่าโมเดลใดเหมาะกับบอร์ดที่สุด
ชื่อหัวข้องานมี online แต่ข้อกำหนดห้าม Cloud: ใช้เว็บเฉพาะค้นหลักฐาน/ดาวน์โหลดไฟล์ ไม่ส่งข้อความเบรลล์ไปบริการ TTS ออนไลน์ และไม่เพิ่ม Cloud fallback

## เครื่องที่ทดสอบจริง

- Windows 11 Home Single Language x64, build 26200; AMD Ryzen 7 7800X3D, 16 logical CPUs, RAM 30.88 GiB; Python 3.12.13
- แยก environment `scratch/thai-tts-env` ออกจาก `.venv` ของ scanner
- Torch 2.8.0+cpu, Transformers 4.57.1, ONNX Runtime 1.31.0 CPU, Piper 1.8.0, VachanaTTS 0.0.7, PyThaiNLP 5.4.0, SSG 0.0.8, psutil 7.1.0
- ผู้ใช้ยืนยันว่าไม่มีบอร์ดให้เข้าถึง: ยังไม่ยืนยัน OS/CPU/RAM/runtime จริงของ QCS6490 และไม่มีผลทดสอบ Linux ARM64 ในงานนี้

## เปรียบเทียบจากแหล่งต้นทาง

| ตัวเลือก | เสียงไทย/วิธีรันที่พบ | License ที่ตรวจ | ขนาดน้ำหนัก | ข้อจำกัด/สถานะ |
| --- | --- | --- | --- | --- |
| Piper tsync2 medium | ไทยเสียงเดียว, ONNX CPU | โค้ด Piper GPL-3.0; โมเดล CC BY-NC-SA 3.0 | 63.22 MB | baseline เดิม; offline ผ่านบน Windows |
| Vachana ผ่าน PyThaiTTS หรือ vachanatts | ไทย 4 เสียง th_f_1/th_m_1/th_f_2/th_m_2, VITS/ONNX; ทดสอบสองเสียงแรก | model card ระบุ MIT; PyThaiTTS Apache-2.0; **vachanatts 0.0.7 ไม่มี license ใน metadata/ไม่มี LICENSE ใน wheel** | 63.52 MB/เสียง | ทดลองผ่านด้วย local Voice.load; ไม่ตั้งเป็น backend ปกติจนกว่าจะเคลียร์ license ของ implementation/ส่วน G2P ที่ให้เครดิต TLTK และ thaig2p |
| MMS-TTS Thai | checkpoint ไทยโดยตรง, VITS/PyTorch CPU | โมเดล CC-BY-NC-4.0; Transformers Apache-2.0, PyTorch BSD-3-Clause | 145.25 MB (safetensors) | offline ผ่าน; เชื่อมเป็นตัวเลือกทดลอง; ไม่ใช่ตัวเลือกใช้งานเชิงพาณิชย์ |
| F5-TTS Thai v1/v2 | v1 รับข้อความไทย; v2 ใช้ phoneme, ต้องมี reference audio/text | โค้ด MIT; Thai v1 card ระบุ CC-BY-4.0 แต่ upstream weights ระบุ CC-BY-NC และ v2 card ที่ตรวจไม่มี license ชัดเจน | v2 checkpoint ที่หน้าไฟล์ประมาณ 1.35 GB ยังไม่รวม vocoder/ส่วนประกอบ | ไม่ดาวน์โหลด/ติดตั้ง เพราะสิทธิ์ของ checkpoint ที่จะใช้ยังไม่ชัดเจน; README ผู้พัฒนาระบุข้อจำกัดข้าม/ซ้ำคำและข้อความยาว |

PyThaiTTS เป็นไลบรารีเลือก backend ไม่ใช่โมเดลเสียงเดียว; ตรวจ source แล้วโหมด vachana เรียก vachanatts ต่อ การทดลองนี้เรียก Voice.load โดยตรงเพื่อระบุไฟล์ local และไม่ใช้ auto-download ของ wrapper
เสียงที่สร้างได้ไม่ใช่หลักฐานว่าทุกคำ/วรรณยุกต์ถูกต้อง ต้องฟังทั้งประโยคทั่วไปและชุดวรรณยุกต์ก่อนเลือกใช้จริง

## Benchmark จริงบน Windows

หนึ่ง process ใหม่ต่อ engine, อ่านสามข้อความต่อเนื่องโดย reuse โมเดลเฉพาะใน benchmark; ค่า import/model load รวมอยู่ใน load ส่วน interpreter startup ไม่รวมอยู่ใน ready_seconds
ใช้ค่า threads เริ่มต้นของแต่ละไลบรารี ไม่มีการอ้างว่าใช้ accelerator ของ Qualcomm; เป็นหนึ่งรอบวัดที่ filesystem/package caches อุ่นแล้ว ไม่ใช่การวัด cold boot ของบอร์ด

| Engine | import+load (s) | WAV แรกพร้อมเล่น (s) | synth ชุด 1/2/3 (s) | CPU รวม (core-s) | peak process-tree RSS (MiB) |
| --- | ---: | ---: | --- | ---: | ---: |
| Piper | 1.915 | 5.383 | 3.467 / 0.767 / 0.420 | 9.16 | 632.2 |
| Vachana หญิง | 3.835 | 4.457 | 0.621 / 0.071 / 0.120 | 6.86 | 497.3 |
| Vachana ชาย | 3.886 | 4.437 | 0.551 / 0.064 / 0.108 | 6.61 | 506.7 |
| MMS | 6.528 | 7.106 | 0.577 / 0.421 / 0.617 | 17.67 | 614.3 |

CPU เป็นเวลาประมวลผลสะสมของ process ไม่ใช่ wall time; ค่าเฉลี่ยเทียบ process wall ของรอบนี้ประมาณ Piper 129%, Vachana หญิง 121%, ชาย 119%, MMS 198% โดย 100% เท่ากับหนึ่ง logical CPU ไม่ใช่ทั้งเครื่อง
RSS รวม launcher และ descendant Python บน Windows (sampling ทุก 20 ms); ไม่ใช่ RAM ทั้งระบบและอาจพลาด peak ระหว่าง sample
เวลาถึง WAV พร้อมเล่น **ไม่ใช่เวลาถึงเสียงออกลำโพงจริง**; ไม่ได้วัด latency ของ audio device

ทดสอบเส้นทาง synthesize_wav ที่เชื่อมจริงของ MMS แยกอีกครั้งโดยปิด network: load 6.457 s, synth 0.529 s, ready 6.986 s, เสียงยาว 4.752 s
UI ปัจจุบันโหลดโมเดลใหม่ทุกครั้งที่กดอ่าน ดังนั้นห้ามใช้ synth-only ของรอบ warm เป็นเวลาเริ่มพูดของ UI
ผลดิบพร้อม corpus/provenance/hash อยู่ใน [windows-results.json](thai_tts/windows-results.json)

ตัวอย่างฟังเทียบ (ไฟล์ local อยู่ใน output ที่ไม่ commit):

- `output/thai_tts/measured/piper_1.wav` ถึง `_3.wav`
- `output/thai_tts/measured/vachana_f_1.wav` ถึง `_3.wav` และ `vachana_m_1.wav` ถึง `_3.wav`
- `output/thai_tts/measured/mms_1.wav` ถึง `_3.wav`
- `output/thai_tts/mms_integrated.wav` จากเส้นทางที่เชื่อมจริง

## ติดตั้ง/ทดลองซ้ำบน Windows

ตรวจ license ข้างต้นก่อนดาวน์โหลด โดยเฉพาะโมเดลที่จำกัด non-commercial และสถานะ vachanatts; ไม่ commit โมเดลหรือ environment

```powershell
uv venv scratch/thai-tts-env --python .venv/Scripts/python.exe
uv pip install --python scratch/thai-tts-env/Scripts/python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python scratch/thai-tts-env/Scripts/python.exe -r requirements-thai-tts.txt
# เฉพาะทดลองเปรียบเทียบ Vachana: license แพ็กเกจยังไม่ยืนยัน ห้ามถือว่า model MIT ครอบคลุมโค้ดนี้
uv pip install --python scratch/thai-tts-env/Scripts/python.exe vachanatts==0.0.7
scratch/thai-tts-env/Scripts/python.exe tools/diagnostics/benchmark_thai_tts.py --download
scratch/thai-tts-env/Scripts/python.exe tools/diagnostics/benchmark_thai_tts.py
```

download เป็นคำสั่ง explicit เท่านั้น ใช้ immutable revisions และตรวจ SHA-256 ของน้ำหนักก่อนแทนไฟล์เดิม บันทึก provenance ทุกไฟล์ ส่วนคำสั่ง benchmark ปิด network ระหว่างสังเคราะห์
สคริปต์ยังใช้โมเดล Piper เดิมใน scratch/piper-voices สำหรับ baseline และ English; ติดตั้งตามเอกสาร [Piper เดิม](PIPER_OFFLINE_TTS_TH.md) หากยังไม่มีไฟล์เหล่านี้
dependencies ระดับบน pin รุ่นตามที่ทดลอง แต่ไม่ใช่ lock transitive packages ทั้งหมด จึงไม่รับประกันตัวเลข benchmark เท่าเดิมหลังติดตั้งในอนาคต

เปิด scanner ด้วย MMS เฉพาะภาษาไทย:

```powershell
.\main.bat piper --tts-thai-backend mms
```

กด S/Read-repeat เพื่ออ่านข้อความที่ยืนยันแล้ว และ B/Stop voice เพื่อหยุด; English [L] ยังคงใช้โมเดล Piper อังกฤษเดิม
ไฟล์ขาดหรือสังเคราะห์ล้มเหลวต้องแสดง error ไม่มี download หรือ Cloud fallback
กลับ baseline ได้ทันทีด้วย `.\main.bat piper` โดยไม่ต้อง rollback โค้ด

## Linux ARM64 และบอร์ด

ตรวจ metadata ของ PyPI แล้วมี wheel CPython 3.12 Linux aarch64 ของ Torch 2.8.0 และ ONNX Runtime 1.31.0 ที่ระบุ manylinux_2_28; Transformers เป็น pure-Python wheel
นี่เป็นหลักฐานเฉพาะ runtime distribution ไม่ใช่การยืนยันว่าโมเดล/ชุด dependencies นี้ทำงานบน QCS6490 จริง การมี ONNX ไม่ได้แปลว่าใช้ QNN/NPU ได้อัตโนมัติ
ก่อนนำขึ้นบอร์ดต้องตรวจ `uname -m`, OS, `ldd --version`, Python ABI, RAM/free storage, ALSA/aplay และ audio device จริง แล้วทดสอบ dependencies ทั้งชุดใหม่
บน Linux ใช้ `venv/bin/python` และระบุ `--tts-python`, `--tts-thai-model`, `--tts-english-model` แบบ absolute ผ่าน `camera_reader.py`; default paths ของ main.bat เป็น Windows
ยังต้องวัดเวลาเริ่มเสียงจากปุ่ม, CPU/RAM เมื่อรันกล้องร่วมด้วย, ความร้อน/เสียงต่อเนื่อง, stop ขณะ ALSA เล่นจริง และฟังวรรณยุกต์/คำหายบนบอร์ด ไม่มีผลทดสอบเหล่านี้ในงานนี้

## Regression/ขอบเขตโค้ด

เพิ่ม local-file validation, language-specific backend dispatch, PCM16/clipping tests, launcher MMS tests และ model-download hash/atomic replacement tests
baseline ก่อนแก้ผ่าน 42 tests ของ TTS/UI/launcher/Thai symbols; regression tests ของ MMS ใหม่เริ่มจาก 4 errors เพราะยังไม่มี backend parameter และ launcher MMS test ไม่ผ่านก่อนแก้ หลังเชื่อมครบชุดที่เกี่ยวข้องผ่าน **124 tests** ใน 21.378 s และ `git diff --check` ผ่าน
English Piper smoke ใน environment ทดลองผ่านโดยปิด network (ready 3.834 s, เสียง 3.379 s); ไม่ได้เปลี่ยนไฟล์โมเดล/config อังกฤษ
ทดสอบ controller จริง: ไม่มี confirmed text ไม่เริ่มงาน; S/B สองรอบหยุดทั้ง worker/descendants และล้าง temp files โดยไม่เปลี่ยน grid context/history
ไม่แก้ Detection, Grid, Mapping, Thai Parser หรือเสียงอังกฤษ ไม่เปลี่ยนค่าเริ่มต้นไปโมเดลที่ยังไม่มีผลฟังประเมิน

## แหล่งต้นทาง

- [PyThaiTTS README/การเลือกโมเดล](https://github.com/PyThaiNLP/PyThaiTTS) และ [source dispatch](https://github.com/PyThaiNLP/PyThaiTTS/blob/main/pythaitts/__init__.py)
- [Vachana model card/เสียง](https://huggingface.co/VIZINTZOR/VachanaTTS), [implementation](https://github.com/VYNCX/VachanaTTS2), [PyPI vachanatts 0.0.7](https://pypi.org/pypi/vachanatts/0.0.7/json)
- [MMS Thai model card/วิธี inference/license](https://huggingface.co/facebook/mms-tts-tha)
- [F5 Thai README/ข้อจำกัด](https://github.com/VYNCX/F5-TTS-THAI), [Thai v1 model card](https://huggingface.co/VIZINTZOR/F5-TTS-THAI), [Thai v2](https://huggingface.co/VIZINTZOR/F5-TTS-TH-V2), [upstream license ของ pretrained models](https://github.com/SWivid/F5-TTS)
- [Piper Thai model card](https://huggingface.co/rhasspy/piper-voices/blob/main/th/th_TH/tsync2/medium/MODEL_CARD)
- [Torch 2.8.0 wheel metadata/license](https://pypi.org/pypi/torch/2.8.0/json), [ONNX Runtime 1.31.0 wheel metadata/license](https://pypi.org/pypi/onnxruntime/1.31.0/json)
