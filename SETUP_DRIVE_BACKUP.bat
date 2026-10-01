@echo off
setlocal
cd /d "%~dp0"
echo.
echo === MUCHA - SETUP GOOGLE DRIVE BACKUP ===
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\tools\install_drive_backup_task.ps1"
echo.
if errorlevel 1 (
  echo [ERROR] Nie udalo sie wlaczyc automatycznego backupu.
) else (
  echo [OK] Automatyczny backup zostal wlaczony.
)
echo.
pause
