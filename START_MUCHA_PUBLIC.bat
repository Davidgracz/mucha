@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Mucha + Public Dashboard

set "LOCAL_URL=http://127.0.0.1:8765/public"
set "HEALTH_URL=http://127.0.0.1:8765/health"
set "TUNNEL_LOG=%TEMP%\mucha-cloudflared.log"

echo ================================================
echo   MUCHA + PUBLIC DASHBOARD
echo ================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [BLAD] Brak .venv.
  echo Najpierw uruchom install_windows.bat
  pause
  exit /b 1
)

if not exist ".env" (
  echo [BLAD] Brak pliku .env.
  echo Uzupelnij DISCORD_TOKEN i MUCHA_DASHBOARD_PASSWORD.
  pause
  exit /b 1
)

if not exist "data\connectome\matrix.npz" (
  echo [BLAD] Brak data\connectome\matrix.npz
  pause
  exit /b 1
)

echo [1/4] Uruchamiam Muche...
start "Mucha" cmd /k call "%~dp0START_MUCHA.bat"

echo [2/4] Czekam na dashboard...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$deadline=(Get-Date).AddSeconds(90); while((Get-Date)-lt $deadline){ try { $r=Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%HEALTH_URL%'; if($r.StatusCode -eq 200){ exit 0 } } catch {}; Start-Sleep -Milliseconds 500 }; exit 1"

if errorlevel 1 (
  echo.
  echo [BLAD] Dashboard nie wystartowal w ciagu 90 sekund.
  echo Sprawdz okno Muchy - tam powinien byc powod bledu.
  pause
  exit /b 1
)

echo [OK] Dashboard dziala lokalnie.

where cloudflared >nul 2>nul
if errorlevel 1 (
  echo.
  echo [3/4] Cloudflared nie jest zainstalowany.
  echo Otwieram publiczny widok lokalnie:
  echo   %LOCAL_URL%
  echo.
  start "" "%LOCAL_URL%"
  echo Aby znajomi mogli wejsc z Internetu, zainstaluj:
  echo   winget install --id Cloudflare.cloudflared
  echo.
  pause
  exit /b 0
)

echo [3/4] Uruchamiam publiczny tunel Cloudflare...
del /q "%TUNNEL_LOG%" >nul 2>nul
start "Mucha Cloudflare Tunnel" /min cloudflared tunnel --logfile "%TUNNEL_LOG%" --url http://127.0.0.1:8765

echo [4/4] Czekam na publiczny adres...
set "PUBLIC_URL="

for /f "usebackq delims=" %%U in (`powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$deadline=(Get-Date).AddSeconds(30); while((Get-Date)-lt $deadline){ if(Test-Path '%TUNNEL_LOG%'){ $txt=Get-Content '%TUNNEL_LOG%' -Raw -ErrorAction SilentlyContinue; $m=[regex]::Match($txt,'https://[a-zA-Z0-9.-]+\.trycloudflare\.com'); if($m.Success){ Write-Output $m.Value; exit 0 } }; Start-Sleep -Milliseconds 500 }; exit 1"`) do set "PUBLIC_URL=%%U"

if not defined PUBLIC_URL (
  echo.
  echo [UWAGA] Tunel wystartowal, ale nie udalo sie automatycznie odczytac adresu.
  echo Otwieram lokalny publiczny dashboard:
  echo   %LOCAL_URL%
  start "" "%LOCAL_URL%"
  echo.
  echo Log tunelu:
  echo   %TUNNEL_LOG%
  pause
  exit /b 0
)

set "PUBLIC_DASHBOARD=!PUBLIC_URL!/public"

echo.
echo ================================================
echo   GOTOWE
echo ================================================
echo Publiczny dashboard:
echo   !PUBLIC_DASHBOARD!
echo.
echo Ten link mozesz wyslac znajomym.
echo Config i prywatne API nadal wymagaja logowania admina.
echo.
start "" "!PUBLIC_DASHBOARD!"

echo To okno mozesz zostawic otwarte.
echo Zamkniecie okna tunelu Cloudflare zatrzyma publiczny link.
echo ================================================
echo.
pause
