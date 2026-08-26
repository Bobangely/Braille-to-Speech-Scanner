@echo off
title Braille Scanner - Full HD 1080p Mode
cd /d "%~dp0"
echo.
echo  ======================================================
echo    Braille-to-Speech Real-Time Scanner (Full HD 1080p)
echo  ======================================================
echo.
echo  Controls:
echo    [V] / [F]       : สลับความละเอียด (Full HD 1080p <-> 4K UHD)
echo    [E]             : ปรับระดับความคมชัด (OFF -> LOW -> MED -> HIGH -> ULTRA)
echo    [Z] / [+]       : Zoom In  (ขยายภาพ)
echo    [X] / [-]       : Zoom Out (ย่อภาพ)
echo    [R] / [0]       : Reset Zoom (1.0x)
echo    [Mouse Wheel]   : หมุนล้อเมาส์ ซูมเข้า/ออก
echo    [SPACE]         : Speak Now (สั่งอ่านทันที)
echo    [C]             : สลับสีจุด (Blue / Red / Green / Black)
echo    [L]             : สลับภาษา (Thai / English)
echo    [A]             : เปิด/ปิด Auto-TTS
echo    [P]             : ถ่ายภาพ Snapshot ลง output/
echo    [Q]             : ปิดโปรแกรม
echo.
echo  Starting scanner in Full HD 1080p mode...
echo.
.venv\Scripts\python.exe camera_reader.py --camera 0 --res fhd --sharp 2 --color blue --lang thai
pause
