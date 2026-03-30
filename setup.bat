@echo off
echo =====================================================
echo   ToxAI - Drug Toxicity Prediction Platform
echo   Setup Script
echo =====================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9+
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv toxai-env
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call toxai-env\Scripts\activate.bat

echo [3/4] Installing core dependencies...
pip install --upgrade pip
pip install numpy pandas scikit-learn xgboost joblib tqdm requests openpyxl pyarrow

echo [4/4] Installing chemistry and API dependencies...
pip install rdkit fastapi uvicorn[standard] python-multipart aiofiles shap reportlab

echo.
echo [OPTIONAL] Installing additional packages...
pip install mordred umap-learn optuna || echo [WARNING] Some optional packages failed - continuing

echo.
echo =====================================================
echo   Installation Complete!
echo =====================================================
echo.
echo Next steps:
echo   1. OPTIONAL: Download Tox21 dataset from Kaggle
echo      https://www.kaggle.com/datasets/epicskills/tox21-dataset
echo      Place tox21.csv inside the 'data' folder
echo.
echo   2. Train the models (uses synthetic data if no dataset):
echo      python train.py --quick
echo.
echo   3. Start the server:
echo      python app.py
echo.
echo   4. Open browser: http://localhost:8000
echo =====================================================
pause
