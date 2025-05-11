@echo off
echo *********************************************************************
echo *                                                                   *
echo *               Neural Network Model Training Script                *
echo *                                                                   *
echo *********************************************************************
echo.
echo This script will train the neural network model for emotion-based
echo video recommendations. The trained model will be saved in the 
echo 'models' directory and automatically used by the main application.
echo.
echo Requirements:
echo - At least 10 video interactions in the database
echo.

REM Activate virtual environment if available
if exist MP\Scripts\activate.bat (
    call MP\Scripts\activate.bat
    echo Using virtual environment for training...
    echo.
)

echo Starting training process...
python train_neural_model.py --epochs 100 --batch-size 8 --verbose 1

echo.
echo Training process completed.
echo.
echo You can now run the main application with:
echo   streamlit run app.py
echo.
pause
