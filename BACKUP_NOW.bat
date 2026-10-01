@echo off
setlocal
cd /d "%~dp0"
echo.
echo === MUCHA - BACKUP NOW ===
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\tools\backup_state_to_drive.ps1"
echo.
if errorlevel 1 (
  echo [ERROR] Backup nie powiodl sie. Sprawdz state\drive_backup.log
) else (
  echo [OK] Backup zakonczony.
)
echo.
pause
