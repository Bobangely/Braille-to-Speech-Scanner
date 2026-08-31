@echo off
chcp 65001 >nul 2>&1
title Braille Scanner - Full HD 1080p Mode
cd /d "%~dp0"

echo.
echo  ======================================================
echo    Braille-to-Speech Real-Time Scanner (Full HD 1080p)
echo  ======================================================
echo.
echo  Controls:
echo    [V] / [F]       : Switch Resolution (FHD ^<-^> 4K ^<-^> HD)
echo    [E]             : Sharpness Level (OFF -^> LOW -^> MED -^> HIGH -^> ULTRA)
echo    [Z] / [+]       : Zoom In
echo    [X] / [-]       : Zoom Out
echo    [R] / [0]       : Reset Zoom (1.0x)
echo    [Mouse Wheel]   : Zoom In/Out
echo    [SPACE]         : Speak Now
echo    [C]             : Switch Color (Blue / Red / Green / Black)
echo    [L]             : Switch Language (Thai / English)
echo    [A]             : Toggle Auto-TTS
echo    [P]             : Save Snapshot to output/
echo    [Q]             : Quit
echo.
echo  Starting scanner in Full HD 1080p mode...
echo.

REM หากยังไม่มี .venv ให้เรียกตัวติดตั้งอัตโนมัติ
if not exist ".venv\Scripts\python.exe" (
    echo  ⚠️ ไม่พบ Virtual Environment (.venv)
    echo  กำลังติดตั้ง Dependencies ที่จำเป็นให้อัตโนมัติ...
    echo.
    call install_dependencies.bat
)

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe camera_reader.py --camera 0 --res fhd --sharp 2 --color blue --lang thai
) else (
    python camera_reader.py --camera 0 --res fhd --sharp 2 --color blue --lang thai
)
pause
