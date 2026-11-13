@echo off
rem Double-click once to put a "สถานีอ่านเบรลล์" shortcut with the logo on the desktop.
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Make_Desktop_Shortcut.ps1"
pause
