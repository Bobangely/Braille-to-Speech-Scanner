@echo off
chcp 65001 >nul 2>&1
title Braille Scanner - YOLO Cell Stream
cd /d "%~dp0"

echo.
echo  ======================================================
echo    Braille Scanner - YOLO Cell Stream
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
echo  Starting YOLO cell crops in Full HD 1080p...
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" camera_reader.py --camera 0 --detector yolo --yolo-pipeline stream --res fhd --sharp 0 --lang thai
) else (
    python camera_reader.py --camera 0 --detector yolo --yolo-pipeline stream --res fhd --sharp 0 --lang thai
)
pause
