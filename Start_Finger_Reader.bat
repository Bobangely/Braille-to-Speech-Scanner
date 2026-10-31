@echo off
rem Double-click to start the finger reader with the teacher panel on http://localhost:8080
chcp 65001 >nul
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe
for %%F in ("%PY%" "scratch\piper-voices\th_TH-tsync2-medium.onnx" "models\hand_landmarker.task") do (
    if not exist %%F (
        echo ไม่พบไฟล์ %%~F
        echo ดูวิธีเตรียมใน README หัวข้อ "เริ่มใช้งาน"
        pause
        exit /b 1
    )
)

rem Voices run in their own environments when present (README), else in .venv.
set PIPER_PY=%PY%
set EDGE_PY=%PY%
if exist scratch\piper-env\Scripts\python.exe set PIPER_PY=scratch\piper-env\Scripts\python.exe
if exist scratch\edge-tts-env\Scripts\python.exe set EDGE_PY=scratch\edge-tts-env\Scripts\python.exe
rem Stop now if a voice library is missing, not in the middle of reading.
"%PIPER_PY%" -c "import piper" 2>nul || (
    echo ไม่พบ piper-tts ใน %PIPER_PY%
    echo ติดตั้งด้วย: %PIPER_PY% -m pip install -r requirements-piper.txt
    pause
    exit /b 1
)
"%EDGE_PY%" -c "import edge_tts" 2>nul || (
    echo ไม่พบ edge-tts ใน %EDGE_PY%
    echo ติดตั้งด้วย: %EDGE_PY% -m pip install -r requirements-edge-tts.txt
    pause
    exit /b 1
)

rem Start_Finger_Reader_Window.bat sets SHOW_WINDOW to keep the OpenCV preview.
set HEADLESS=--headless
if defined SHOW_WINDOW set HEADLESS=

start "" http://localhost:8080

"%PY%" finger_reader.py ^
    --tts-thai-model scratch\piper-voices\th_TH-tsync2-medium.onnx ^
    --tts-python "%PIPER_PY%" ^
    --symbol-voice edge --word-voice edge ^
    --edge-python "%EDGE_PY%" ^
    --web 8080 %HEADLESS% %*

if errorlevel 1 (
    echo.
    echo โปรแกรมหยุดเพราะเกิดข้อผิดพลาด ดูข้อความด้านบน
    pause
)
