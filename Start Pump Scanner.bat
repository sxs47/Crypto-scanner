@echo off
title Pump Scanner - keep this window open (close it to stop the app)
cd /d "D:\Ai Tools\Crypto-scanner"

rem Already running? Just open the page, so two copies never run at once.
curl -s -o nul --max-time 3 http://127.0.0.1:5000/ && (
    start "" http://127.0.0.1:5000/
    exit /b
)

rem Open the page once the app has had time to load, then run the app in this window.
start "" /min cmd /c "timeout /t 25 /nobreak >nul & start "" http://127.0.0.1:5000/"
.venv\Scripts\python.exe app.py
echo.
echo The app stopped. Press any key to close this window.
pause >nul
