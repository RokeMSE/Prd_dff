@echo off
REM ============================================================
REM Build Stain Detective into a distributable app
REM ============================================================
REM Usage:  cd "Product Diff" && build.bat
REM Output: dist\StainDetective.exe
REM
REM Prerequisites:
REM   pip install pyinstaller PySide6 opencv-python numpy pandas
REM   pip install pillow python-dotenv google-genai openai
REM ============================================================

echo [1/3] Installing PyInstaller (if needed)...
pip install pyinstaller --quiet

echo [2/3] Building StainDetective...
pyinstaller --clean --noconfirm StainDetective.spec 

echo [3/3] Done!
echo.
echo Executable: dist\StainDetective.exe
echo.
echo IMPORTANT: Place your .env file next to StainDetective.exe
echo            before running, so VLM credentials are available.
pause
