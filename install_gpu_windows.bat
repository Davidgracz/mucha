@echo off
setlocal
cd /d %~dp0

if not exist .venv (
  py -3.12 -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-gpu.txt

echo.
echo ================================================
echo  GPU setup finished.
echo  Run: python tools\check_gpu.py
echo  Then: python bot.py
echo ================================================
echo.
pause
