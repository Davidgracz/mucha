@echo off
setlocal
cd /d "%~dp0"
title Mucha + Chaser launcher

if not exist "%~dp0START_MUCHA.bat" (
  echo Brak START_MUCHA.bat
  pause
  exit /b 1
)

if not exist "%~dp0..\mucha-chaser\START_CHASER.bat" (
  echo Brak:
  echo   %~dp0..\mucha-chaser\START_CHASER.bat
  echo.
  echo Sklonuj chasera do folderu obok Muchy:
  echo   F:\mucha-chaser
  pause
  exit /b 1
)

start "Mucha" cmd /c call "%~dp0START_MUCHA.bat"
timeout /t 2 /nobreak >nul
start "Mucha Chaser" cmd /c call "%~dp0..\mucha-chaser\START_CHASER.bat"

exit /b 0
