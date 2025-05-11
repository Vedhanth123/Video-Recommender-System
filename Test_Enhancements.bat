@echo off
echo Testing the Enhanced Recommendation System...
echo.

:: Activate the virtual environment
call MP\Scripts\activate

:: Run the test script
python test_enhancements.py

:: Deactivate the virtual environment
call MP\Scripts\deactivate

echo.
pause
