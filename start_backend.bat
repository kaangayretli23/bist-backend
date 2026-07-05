@echo off
REM BIST Pro backend baslatici — Task Scheduler / cift tik ile calisir.
REM pythonw.exe: konsol penceresi ACMADAN arka planda calisir (log dosyaya yazilir).
cd /d C:\Users\Kaan\bist-backend
"C:\Users\Kaan\bist-backend\venv\Scripts\pythonw.exe" backend.py >> "C:\Users\Kaan\bist-backend\backend.log" 2>&1
