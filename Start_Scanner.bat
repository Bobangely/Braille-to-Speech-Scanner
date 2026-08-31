@echo off
chcp 65001 >nul 2>&1
title Braille Scanner - Real-Time
cd /d "%~dp0"

echo.
echo  ======================================================
echo    Braille-to-Speech Real-Time Scanner (4K / FHD)
echo  ======================================================
echo.
echo  Controls:
echo    [V] / [F]       : Switch Resolution (4K ^<-^> FHD ^<-^> HD)
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
echo  Starting scanner...
echo.

REM Try .venv first, then uv run, then system python
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe camera_reader.py --camera 0 --res 4k --sharp 2 --color blue --lang thai
) else (
    where uv >nul 2>&1
    if %errorlevel%==0 (
        uv run camera_reader.py --camera 0 --res 4k --sharp 2 --color blue --lang thai
    ) else (
        python camera_reader.py --camera 0 --res 4k --sharp 2 --color blue --lang thai
    )
)
pause
