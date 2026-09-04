@echo off
chcp 65001 >nul 2>&1
title Braille Scanner - Hybrid CV + YOLO Mode
cd /d "%~dp0"

echo.
echo  ======================================================
echo    Braille Scanner - Real-Time Hybrid (CV + YOLO)
echo  ======================================================
echo.
echo  Key Controls:
echo    [Y] / [M]       : Switch Mode (HYBRID / YOLO / OPENCV)
echo    [V] / [F]       : Switch Resolution (Full HD / 4K / HD)
echo    [E]             : Sharpness Level (OFF to ULTRA)
echo    [Z] / [X]       : Zoom In / Zoom Out
echo    [R] / [0]       : Reset Zoom (1.0x)
echo    [C]             : Switch Color (Blue / Red / Green / Black)
echo    [L]             : Switch Language (Thai / English)
echo    [P]             : Save Snapshot to output/
echo    [Q] / [ESC]     : Quit
echo.
echo  Starting scanner with Hybrid CV+YOLO in Full HD 1080p...
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" camera_reader.py --camera 0 --detector hybrid --res fhd --sharp 2 --color blue --lang thai
) else (
    python camera_reader.py --camera 0 --detector hybrid --res fhd --sharp 2 --color blue --lang thai
)
pause
