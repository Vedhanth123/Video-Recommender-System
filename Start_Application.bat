@echo off
color 0A
echo.
echo ========================================================
echo    Emotion-Based Video Recommendation System Launcher
echo ========================================================
echo.

echo Activating virtual environment...
call "MP\Scripts\activate.bat"

echo.
echo Starting application...
echo.
python launcher.py

pause
