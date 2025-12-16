@echo off
title rPPG Lie Detection System v2.0
color 0B

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    rPPG LIE DETECTION SYSTEM v2.0                            ║
echo ║                    Remote Physiological Analysis                             ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.9+
    pause
    exit /b 1
)

:: Check if dependencies are installed
echo [*] Checking dependencies...
python -c "import flask, flask_socketio, cv2, mediapipe, numpy, scipy" 2>nul
if errorlevel 1 (
    echo [*] Installing dependencies...
    pip install flask flask-socketio opencv-python mediapipe numpy scipy eventlet --quiet
)

echo [*] Dependencies OK
echo.
echo [*] Starting server...
echo [*] Open your browser at: http://localhost:5000
echo.
echo ═══════════════════════════════════════════════════════════════════════════════
echo.

python rppg_lie_detector.py

pause
