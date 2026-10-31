@echo off
setlocal
rem Same as Start_Finger_Reader.bat, plus the OpenCV preview window for development.
set SHOW_WINDOW=1
call "%~dp0Start_Finger_Reader.bat" %*
