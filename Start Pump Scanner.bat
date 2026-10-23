@echo off
title Pump Scanner - keep this window open (close it to stop the app)
cd /d "D:\Ai Tools\Crypto-scanner"

rem Already running? Just open the page, so two copies never run at once.
curl -s -o nul --max-time 3 http://127.0.0.1:5000/ && (
    start "" http://127.0.0.1:5000/
    exit /b
)

rem Open the page once the app has had time to load (skipped when started with Windows).
if /i not "%~1"=="--startup" start "" /min cmd /c "timeout /t 25 /nobreak >nul & start "" http://127.0.0.1:5000/"

rem Run the app; if it ever stops by itself, start it again after 15 seconds.
:run
.venv\Scripts\python.exe app.py
echo.
echo The app stopped at %date% %time%. Restarting in 15 seconds - close this window to stop it for good.
timeout /t 15 /nobreak >nul
goto run
