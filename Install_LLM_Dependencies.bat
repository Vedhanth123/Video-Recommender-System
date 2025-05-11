@echo off
echo *********************************************************************
echo *                                                                   *
echo *               Install LLM Dependencies                            *
echo *                                                                   *
echo *********************************************************************
echo.
echo This script will install the required packages for the local LLM model.
echo This is needed to use local models for recommendation explanation instead
echo of the paid Gemini API.
echo.

REM Activate virtual environment if available
if exist MP\Scripts\activate.bat (
    call MP\Scripts\activate.bat
    echo Using virtual environment...
    echo.
)

echo Installing required packages...
pip install transformers==4.36.2 torch==2.1.2 accelerate==0.25.0

echo.
echo Installation complete!
echo.
echo Note about model download:
echo - The LLM model (~1.1GB) will be downloaded on first application run
echo - This download happens only ONCE, then it's stored locally
echo - The model files are saved in the "models\tinyllama_cache" folder
echo.
echo You can now run the main application with:
echo   Start_Application.bat
echo.
pause
