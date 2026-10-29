@echo off
setlocal
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title Braille Station Launcher - Live Camera and Web Interface

echo ===================================================================
echo   BRAILLE STATION - Live Camera and Web Interface
echo ===================================================================
echo   Opening: http://localhost:8080
echo   Close this window to stop the system.
echo ===================================================================
echo.

start /b cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8080"
".venv\Scripts\python.exe" finger_reader.py ^
    --tts-thai-model scratch\piper-voices\th_TH-tsync2-medium.onnx ^
    --tts-python .venv\Scripts\python.exe ^
    --symbol-voice edge --word-voice edge ^
    --edge-python .venv\Scripts\python.exe ^
    --web 8080 --headless %*

endlocal
pause
