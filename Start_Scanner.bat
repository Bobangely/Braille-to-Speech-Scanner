@echo off
chcp 65001 >nul 2>&1
title Braille Scanner - Real-Time
cd /d "%~dp0"

echo.
echo  ======================================================
echo    Braille-to-Speech Real-Time Scanner (4K / FHD)
echo  ======================================================
echo.
echo  Controls ^& Shortcuts:
echo    [V] / [F]       : สลับความละเอียด (4K UHD ^<-^> Full HD 1080p)
echo    [E]             : ปรับระดับความคมชัด (OFF -^> LOW -^> MED -^> HIGH -^> ULTRA)
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
echo  Starting scanner...
echo.

REM หากยังไม่มี .venv ให้เรียกตัวติดตั้งอัตโนมัติ
if not exist ".venv\Scripts\python.exe" (
    echo  ⚠️ ไม่พบ Virtual Environment (.venv)
    echo  กำลังติดตั้ง Dependencies ที่จำเป็นให้อัตโนมัติ...
    echo.
    call install_dependencies.bat
)

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe camera_reader.py --camera 0 --res 4k --sharp 2 --color blue --lang thai
) else (
    python camera_reader.py --camera 0 --res 4k --sharp 2 --color blue --lang thai
)
pause
