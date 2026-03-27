@echo off
title BIST Pro - Paper Trading + Telefon Erisimi
echo ==========================================
echo   BIST Pro v7.1.0 - Paper Trading Motoru
echo   + Telefon Erisimi (ngrok tunnel)
echo ==========================================
echo.

cd /d "%~dp0"

:: Sanal ortami aktif et
call venv\Scripts\activate.bat

:: ngrok kurulu mu kontrol et
where ngrok >nul 2>nul
if %errorlevel% neq 0 (
    echo [HATA] ngrok bulunamadi!
    echo.
    echo Kurulum:
    echo   1. https://ngrok.com/download adresinden indir
    echo   2. ngrok.exe dosyasini bu klasore kopyala
    echo   3. https://dashboard.ngrok.com/signup adresinden ucretsiz hesap ac
    echo   4. Token'i kopyala ve calistir: ngrok config add-authtoken SENIN_TOKEN
    echo.
    echo ngrok olmadan sadece yerel erisimle baslatiliyor...
    echo Tarayicidan eris: http://localhost:5000
    echo.
    python backend.py
    pause
    exit /b
)

:: Backend'i arka planda baslat
echo [1/2] Backend baslatiliyor...
start /b python backend.py

:: 3 saniye bekle (backend'in ayaga kalkmasi icin)
timeout /t 3 /nobreak >nul

:: ngrok tunnel ac
echo [2/2] Telefon erisimi aciliyor (ngrok)...
echo.
echo ==========================================
echo   Asagidaki URL'yi telefonunda ac:
echo   (Forwarding satirindaki https://... linki)
echo ==========================================
echo.
ngrok http 5000
