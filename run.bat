@echo off
echo ================================================================
echo      IntruSense - AI Powered Intrusion Detection System Pipeline
echo ================================================================
echo.

REM Activating virtual environment for entruSense......
REM If using venv:
REM call venv\Scripts\activate

REM If using conda:
REM call conda activate intrusense

echo [1/8] Reading and mergin entruSense raw data: NSL KDD & CICIDS2017 .....
python src\test_data_creation.py
if %errorlevel% neq 0 (
    echo ERROR in test_data_creation.py
    pause
    exit /b %errorlevel%
)
echo.

echo [2/8] entruSense Data Preprocessing ...
python src\data_preprocess.py
if %errorlevel% neq 0 (
    echo ERROR in data_preprocess.py
    pause
    exit /b %errorlevel%
)
echo.

echo [3/8] entruSense Supervised Model Training: RandomForest & XGBoost ...
python src\supervised_model.py
if %errorlevel% neq 0 (
    echo ERROR in supervised_model.py
    pause
    exit /b %errorlevel%
)
echo.

echo [4/8] entruSense Autoencoder Model Training ...
python src\autoencoder_model.py
if %errorlevel% neq 0 (
    echo ERROR in autoencoder_model.py
    pause
    exit /b %error
)
echo.

echo [5/8] entruSense Hybrid Model Training ...
python src\hybrid_model.py
if %errorlevel% neq 0 (
    echo ERROR in hybrid_model.py
    pause
    exit /b %error
)
echo.

echo [6/8] entruSense Model Evaluation ...
python src\evaluate_models.py
if %errorlevel% neq 0 (
    echo ERROR in evaluate_models.py
    pause
    exit /b %error
)
echo.

echo [7/8] entruSense Visual Generation ...
python src\generate_visuals.py
if %errorlevel% neq 0 (
    echo ERROR in generate_visuals.py
    pause
    exit /b %error
)
echo.

echo [8/8] entruSense Simulation  ...
python src\entruSense.py
if %errorlevel% neq 0 (
    echo ERROR in entruSense.py
    pause
    exit /b %error
)
echo.
