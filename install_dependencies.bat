@echo off
chcp 65001 >nul 2>&1
title Braille Scanner - One-Click Installer
cd /d "%~dp0"

echo.
echo  ======================================================
echo    Braille-to-Speech Scanner: Environment Setup
echo  ======================================================
echo.
echo  กำลังตรวจสอบและติดตั้งโปรแกรมที่จำเป็น...
echo.

REM 1. เช็คว่ามี Python ในเครื่องหรือไม่
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  ❌ ไม่พบ Python ในเครื่อง!
    echo.
    echo  กรุณาติดตั้ง Python ก่อน (แนะนำ Python 3.10 - 3.12):
    echo  1. ดาวน์โหลดที่: https://www.python.org/downloads/
    echo  2. ตอนติดตั้ง อย่าลืมติ๊กถูก [x] "Add python.exe to PATH"
    echo.
    pause
    exit /b 1
)

echo  ✅ ตรวจพบ Python:
python --version
echo.

REM 2. สร้าง Virtual Environment (.venv) หากยังไม่มี
if not exist ".venv" (
    echo  📦 กำลังสร้าง Virtual Environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo  ❌ สร้าง .venv ไม่สำเร็จ!
        pause
        exit /b 1
    )
    echo  ✅ สร้าง .venv สำเร็จ!
    echo.
) else (
    echo  ✅ พบโฟลเดอร์ .venv อยู่แล้ว
    echo.
)

REM 3. อัปเกรด pip และติดตั้ง Requirements
echo  📥 กำลังติดตั้ง Dependencies จาก requirements.txt...
echo  (opencv-python, numpy, Pillow, pyttsx3, gTTS)
echo.

.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo  ⚠️ การติดตั้งบางแพ็กเกจอาจมีปัญหา ลองติดตั้งผ่าน pip โดยตรง...
    pip install -r requirements.txt
)

echo.
echo  ======================================================
echo    🎉 ติดตั้ง Dependencies ทั้งหมดเรียบร้อยแล้ว!
echo  ======================================================
echo.
echo  คุณสามารถเริ่มใช้งานได้ทันทีโดยดับเบิ้ลคลิก:
echo    - Start_Scanner_FHD.bat (โหมด Full HD 1080p)
echo    - Start_Scanner_4K.bat  (โหมด 4K Ultra HD)
echo    - Test_Camera.bat       (ทดสอบกล้อง)
echo.
pause
