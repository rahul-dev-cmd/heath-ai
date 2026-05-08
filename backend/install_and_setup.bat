@echo off
echo ========================================
echo  SmartKitchen AI X - Setup Script
echo ========================================
echo.

echo [1/3] Installing Python dependencies...
pip install pandas numpy scikit-learn xgboost joblib pydantic pydantic-settings fastapi uvicorn[standard] sqlalchemy python-jose[cryptography] passlib[bcrypt] bcrypt python-multipart python-dotenv httpx
echo.

echo [2/3] Generating synthetic data and training ML models...
cd /d "%~dp0"
python setup_ml.py
echo.

echo [3/3] Setup complete!
echo.
echo To start the server, run:
echo   cd backend
echo   uvicorn app.main:app --reload --port 8000
echo.
echo Then open: http://localhost:8000/docs
echo.
pause
