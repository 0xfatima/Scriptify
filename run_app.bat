@echo off
setlocal
cd /d "%~dp0"

if exist ".\.venv\Scripts\python.exe" (
  ".\.venv\Scripts\python.exe" ".\app.py"
  exit /b %ERRORLEVEL%
)

echo Could not find .venv\Scripts\python.exe
echo Please create a venv and install requirements:
echo   python -m venv .venv
echo   .\.venv\Scripts\pip.exe install -r requirements.txt
pause
exit /b 1

