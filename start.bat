@echo off
title BIST Pro - Paper Trading
echo ==========================================
echo   BIST Pro v7.1.0 - Paper Trading Motoru
echo ==========================================
echo.

cd /d "%~dp0"

:: Sanal ortami aktif et
call venv\Scripts\activate.bat

:: Backend'i baslat
echo [BASLATILIYOR] Backend ayaga kalkiyor...
echo [BILGI] Tarayicidan eris: http://localhost:5000
echo [BILGI] Kapatmak icin bu pencereyi kapat veya Ctrl+C bas
echo.
python backend.py

pause
