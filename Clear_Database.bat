@echo off
color 0C
echo *********************************************************************
echo *                                                                   *
echo *               DATABASE CLEARING UTILITY                           *
echo *                                                                   *
echo *********************************************************************
echo.
echo WARNING: This utility will delete ALL data from ALL tables in the database.
echo This includes:
echo  - User emotion data
echo  - Video interactions
echo  - User preferences
echo  - All other stored data
echo.
echo This action CANNOT be undone.
echo.

REM Activate virtual environment if available
if exist MP\Scripts\activate.bat (
    call MP\Scripts\activate.bat
    echo Using virtual environment...
    echo.
)

echo Starting database clearing process...
python clear_database.py

echo.
echo Press any key to exit...
pause > nul
