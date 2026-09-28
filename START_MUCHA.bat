@echo off
setlocal
cd /d "%~dp0"
title Mucha

set PYTHONUTF8=1

if not exist ".venv\Scripts\python.exe" (
  echo Brak .venv.
  echo Najpierw uruchom install_windows.bat
  pause
  exit /b 1
)

if not exist ".env" (
  echo Brak pliku .env.
  echo Skopiuj .env z backupu albo uzupelnij token Discorda.
  pause
  exit /b 1
)

if not exist "data\connectome\matrix.npz" (
  echo Brak prawdziwego connectome:
  echo   data\connectome\matrix.npz
  echo Skopiuj folder data\connectome z backupu.
  pause
  exit /b 1
)

echo ================================================
echo   START MUCHA
echo   Folder: %CD%
echo ================================================
echo.

".venv\Scripts\python.exe" bot.py
set EXITCODE=%ERRORLEVEL%

echo.
echo Mucha zakonczyla dzialanie. Kod: %EXITCODE%
pause
exit /b %EXITCODE%
