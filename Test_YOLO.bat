@echo off
chcp 65001 >nul 2>&1
title Braille Scanner - Hybrid CV + YOLO Tester
cd /d "%~dp0"

echo.
echo  ==============================================================
echo     🧠 Braille Dot Detector (Hybrid CV + YOLO Deep Learning)
echo  ==============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo  ⚠️ ไม่พบ .venv กำลังติดตั้ง dependencies...
    call install_dependencies.bat
)

if "%~1"=="" goto menu

REM ถ้าลากไฟล์รูปภาพมาวางบน .bat ให้รันรูปนั้นทันที
echo  🔍 ตรวจจับภาพ: %~1
.venv\Scripts\python.exe yolo_detector.py "%~1" --lang thai --save
goto end

:menu
echo  เลือกภาพตัวอย่างที่ต้องการทดสอบด้วย YOLO:
echo.
echo    [1] Test_thai_01.png (ภาพจุดสีดำ: "สวัสดีครับผม")
echo    [2] test_thai_home.png (ภาพจุดสีฟ้า: "บ้าน")
echo    [3] test_thai_cat.png (ภาพจุดสีฟ้า: "แมว")
echo    [4] test_hello_blue.png (ภาษาอังกฤษ: "hello")
echo    [5] ระบุ path รูปภาพเอง หรือลากไฟล์ภาพมาวาง
echo    [Q] ออกจากโปรแกรม
echo.

set /p choice="  กรุณาเลือกเมนู (1-5 หรือ Q): "

if /i "%choice%"=="1" (
    .venv\Scripts\python.exe yolo_detector.py sample_images/Test_thai_01.png --lang thai --save
    goto menu
)
if /i "%choice%"=="2" (
    .venv\Scripts\python.exe yolo_detector.py sample_images/test_thai_home.png --lang thai --save
    goto menu
)
if /i "%choice%"=="3" (
    .venv\Scripts\python.exe yolo_detector.py sample_images/test_thai_cat.png --lang thai --save
    goto menu
)
if /i "%choice%"=="4" (
    .venv\Scripts\python.exe yolo_detector.py sample_images/test_hello_blue.png --lang english --save
    goto menu
)
if /i "%choice%"=="5" (
    echo.
    set /p custom_img="  ใส่ path ของรูปภาพ (หรือลากไฟล์มาวางที่นี่): "
    if not "%custom_img%"=="" (
        .venv\Scripts\python.exe yolo_detector.py %custom_img% --lang thai --save
    )
    goto menu
)
if /i "%choice%"=="Q" goto end

:end
echo.
echo  เสร็จสิ้นการทำงาน
pause
