@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul 2>&1
title Braille Reader
set "BRAILLE_CALLER_CWD=%CD%"
pushd "%~dp0" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Cannot enter the project folder.
    endlocal & exit /b 1
)
if not exist "tools\launch.py" (
    echo [ERROR] Missing tools\launch.py. Restore the project files.
    set "BRAILLE_EXIT=1"
    goto done
)
if exist ".venv\Scripts\python.exe" goto local_python
py -3 -c "import sys; sys.exit(sys.version_info < (3,12))" >nul 2>&1
if not errorlevel 1 goto py_launcher
python -c "import sys; sys.exit(sys.version_info < (3,12))" >nul 2>&1
if not errorlevel 1 goto path_python
echo [ERROR] Python 3.12 or newer is required. Install Python, then run main.bat setup.
set "BRAILLE_EXIT=1"
goto done

:local_python
".venv\Scripts\python.exe" -X utf8 "tools\launch.py" %*
set "BRAILLE_EXIT=%errorlevel%"
goto done

:py_launcher
py -3 -X utf8 "tools\launch.py" %*
set "BRAILLE_EXIT=%errorlevel%"
goto done

:path_python
python -X utf8 "tools\launch.py" %*
set "BRAILLE_EXIT=%errorlevel%"

:done
if not "%BRAILLE_EXIT%"=="0" echo [ERROR] Launcher stopped. Read the error above.
if "%~1"=="" if not "%BRAILLE_EXIT%"=="0" pause
popd
endlocal & exit /b %BRAILLE_EXIT%
