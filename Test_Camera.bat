@echo off
chcp 65001 >nul 2>&1
title Braille Scanner - Camera Test
cd /d "%~dp0"

echo.
echo  ============================================
echo    Camera Test - Braille Reader
echo  ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo  ⚠️ ไม่พบ Virtual Environment (.venv)
    echo  กำลังติดตั้ง Dependencies ที่จำเป็นให้อัตโนมัติ...
    echo.
    call install_dependencies.bat
)

echo  Testing camera...
echo  Press Q in the camera window to quit.
echo.

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe test_camera.py
) else (
    python test_camera.py
)
pause
