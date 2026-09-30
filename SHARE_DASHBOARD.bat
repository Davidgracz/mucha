@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Mucha - Public Dashboard

echo ================================================
echo   MUCHA - PUBLIC READ-ONLY DASHBOARD
echo ================================================
echo.
echo Sprawdzam lokalny dashboard...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 http://127.0.0.1:8765/health; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1"

if errorlevel 1 (
  echo.
  echo [BLAD] Dashboard Muchy nie odpowiada na 127.0.0.1:8765.
  echo Najpierw uruchom START_MUCHA.bat i zostaw Muchę wlaczona.
  echo.
  pause
  exit /b 1
)

where cloudflared >nul 2>nul
if errorlevel 1 (
  echo.
  echo [BRAK] Nie znaleziono cloudflared.
  echo.
  echo Zainstaluj Cloudflare Tunnel, a potem uruchom ten plik ponownie.
  echo Oficjalny instalator: https://developers.cloudflare.com/tunnel/downloads/
  echo.
  echo Jezeli masz winget, mozesz sprobowac:
  echo   winget install --id Cloudflare.cloudflared
  echo.
  pause
  exit /b 1
)

echo.
echo Dashboard dziala.
echo Za chwile Cloudflare wypisze adres w stylu:
echo   https://random-words.trycloudflare.com
echo.
echo ZNAJOMYM wysylasz TEN ADRES z dopiskiem:
echo   /public
echo.
echo Przyklad:
echo   https://random-words.trycloudflare.com/public
echo.
echo /config i prywatne API nadal wymagaja logowania administratora.
echo Nie udostepniaj hasla MUCHA_DASHBOARD_PASSWORD.
echo.
echo Zatrzymanie udostepniania: Ctrl+C albo zamknij to okno.
echo ================================================
echo.

cloudflared tunnel --url http://127.0.0.1:8765

echo.
echo Tunel zakonczony.
pause
