@echo off
echo Starting Cardiovascular Disease Predictor...
echo.
echo Choose an option:
echo 1. Original UI (all fields at once) - app.py
echo 2. Step-by-step Wizard UI - app_wizard.py
echo.
set /p choice="Enter choice (1 or 2): "
if "%choice%"=="1" (
    python -m streamlit run app.py
) else if "%choice%"=="2" (
    python -m streamlit run app_wizard.py
) else (
    echo Invalid choice. Running default app.py...
    python -m streamlit run app.py
)
pause

