@echo off
echo TakuNet Inference Runner
echo ----------------------

REM Check if checkpoint exists
if not exist "ckpts\TakuNet_AIDERV2.ckpt" (
    echo Error: Checkpoint file not found.
    echo Please make sure ckpts\TakuNet_AIDERV2.ckpt exists or update the path in this script.
    pause
    exit /b 1
)

echo Choose a test folder:
echo 1. Earthquake
echo 2. Fire
echo 3. Flood
echo 4. Normal
echo 5. Custom image

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    python fixed_inference.py --dir Test\Earthquake --checkpoint ckpts\TakuNet_AIDERV2.ckpt --visualize
) else if "%choice%"=="2" (
    python fixed_inference.py --dir Test\Fire --checkpoint ckpts\TakuNet_AIDERV2.ckpt --visualize
) else if "%choice%"=="3" (
    python fixed_inference.py --dir Test\Flood --checkpoint ckpts\TakuNet_AIDERV2.ckpt --visualize
) else if "%choice%"=="4" (
    python fixed_inference.py --dir Test\Normal --checkpoint ckpts\TakuNet_AIDERV2.ckpt --visualize
) else if "%choice%"=="5" (
    set /p image_path="Enter path to image file: "
    python fixed_inference.py --image "%image_path%" --checkpoint ckpts\TakuNet_AIDERV2.ckpt --visualize
) else (
    echo Invalid choice
)

echo.
echo Done!
pause 