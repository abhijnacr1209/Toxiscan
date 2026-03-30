@echo off
echo =====================================================
echo   ToxAI - Starting Drug Toxicity Server
echo =====================================================
echo.

:: Activate venv if it exists
if exist "toxai-env\Scripts\activate.bat" (
    call toxai-env\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [INFO] No venv found, using global Python
)

:: Check if models are trained
if not exist "models\ensemble_meta.pkl" (
    echo [WARNING] Models not trained yet!
    echo.
    echo Training models now with quick mode...
    echo This uses synthetic data if tox21.csv is not in data/
    echo.
    python train.py --quick --no-shap
    echo.
)

echo [OK] Starting server...
echo.
echo   Frontend: http://localhost:8000
echo   API Docs: http://localhost:8000/api/docs
echo.
python app.py
pause
