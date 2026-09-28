@echo off
setlocal
cd /d "%~dp0"
title Mucha - Windows setup

echo ================================================
echo   MUCHA - instalacja Windows
echo ================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Tworze .venv w Python 3.12...
  py -3.12 -m venv .venv
  if errorlevel 1 goto :fail
) else (
  echo [1/4] .venv juz istnieje.
)

call ".venv\Scripts\activate.bat"

echo [2/4] Aktualizuje pip...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

echo [3/4] Instaluje zaleznosci...
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [4/4] Sprawdzam pliki lokalne...
if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo Utworzono .env - wpisz token Discorda.
) else (
  echo .env zachowany.
)

if exist "data\connectome\matrix.npz" (
  echo Prawdziwy connectome jest na miejscu - NIE NADPISUJE GO.
) else (
  echo.
  echo UWAGA: brak data\connectome\matrix.npz
  echo Skopiuj data\connectome z backupu przed uruchomieniem Muchy.
  echo Instalator NIE tworzy automatycznie demo connectome.
)

echo.
echo GOTOWE.
echo Dla CUDA uruchom: install_gpu_windows.bat
echo Start Muchy: START_MUCHA.bat
echo.
pause
exit /b 0

:fail
echo.
echo BLAD instalacji. Kod: %ERRORLEVEL%
pause
exit /b %ERRORLEVEL%
