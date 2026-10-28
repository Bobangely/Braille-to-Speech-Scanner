@echo off
rem Double-click to start the finger reader with the teacher panel on http://localhost:8080
chcp 65001 >nul
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe
for %%F in ("%PY%" "scratch\piper-voices\th_TH-tsync2-medium.onnx" "scratch\piper-env\Scripts\python.exe" "scratch\edge-tts-env\Scripts\python.exe" "models\hand_landmarker.task") do (
    if not exist %%F (
        echo ไม่พบไฟล์ %%~F
        echo ดูวิธีเตรียมใน README หัวข้อ "เริ่มใช้งาน"
        pause
        exit /b 1
    )
)

rem The page shows "ขาดการเชื่อมต่อ" until the voices are ready, then connects by itself.
start "" http://localhost:8080

"%PY%" finger_reader.py ^
    --tts-thai-model scratch\piper-voices\th_TH-tsync2-medium.onnx ^
    --tts-python scratch\piper-env\Scripts\python.exe ^
    --symbol-voice edge --word-voice edge ^
    --edge-python scratch\edge-tts-env\Scripts\python.exe ^
    --web 8080 %*

if errorlevel 1 (
    echo.
    echo โปรแกรมหยุดเพราะเกิดข้อผิดพลาด ดูข้อความด้านบน
    pause
)
