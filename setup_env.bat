@echo off
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Done! Activate with: venv\Scripts\activate
