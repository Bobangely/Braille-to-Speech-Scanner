@echo off
title Braille Scanner - Camera Test
cd /d "%~dp0"
echo.
echo  ============================================
echo    Camera Test - Braille Reader
echo  ============================================
echo.
echo  Testing camera...
echo  Press Q in the camera window to quit.
echo.
.venv\Scripts\python.exe test_camera.py
pause
