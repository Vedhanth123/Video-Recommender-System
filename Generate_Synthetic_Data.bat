@echo off
echo ************************************************************
echo *                                                          *
echo *         Synthetic Data Generator for Neural Network      *
echo *                                                          *
echo ************************************************************
echo.
echo This script will generate synthetic data for training the neural
echo network model. The data will be saved in the user_data.db SQLite
echo database file.
echo.

REM Check if virtual environment exists and activate it
if exist MP\Scripts\activate.bat (
    echo Activating virtual environment...
    call MP\Scripts\activate.bat
) else (
    echo No virtual environment found, using system Python.
)

echo.
echo Generating synthetic data...
python generate_synthetic_data.py
echo.

echo Data generation complete!
echo You can now train the neural network model using:
echo   Train_Model.bat
echo.

pause
