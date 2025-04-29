@echo off
setlocal

echo Installing CPU-only requirements for TakuNet...
python install_cpu_requirements.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Installation failed with error code %ERRORLEVEL%
    echo Please check the error messages above for more information.
    echo.
    echo You can try installing PyTorch manually with:
    echo pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Installation completed successfully!
echo.
pause 