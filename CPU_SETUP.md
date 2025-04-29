# TakuNet CPU-Only Setup

This document explains how to install the CPU-only version of TakuNet's dependencies.

## Contents

1. [`requirements_cpu.txt`](src/requirements/requirements_cpu.txt) - CPU-compatible requirements file
2. [`install_cpu_requirements.py`](install_cpu_requirements.py) - Python script for installing dependencies
3. [`install_cpu_requirements.bat`](install_cpu_requirements.bat) - Windows batch script for easy execution

## Installation Options

### Option 1: Using the Python Script (Recommended)

This method uses optimized installation options and handles timeouts for large packages:

```bash
# For Windows:
install_cpu_requirements.bat

# For Linux/Mac:
python install_cpu_requirements.py
```

The script performs the following steps:
1. Installs PyTorch CPU packages first with an extended timeout
2. Installs all remaining dependencies
3. Uses `--no-cache-dir` to ensure fresh packages are downloaded

### Option 2: Install PyTorch manually first

If you encounter download timeouts, you can install PyTorch separately:

```bash
# Step 1: Install PyTorch CPU packages manually
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu --timeout 180

# Step 2: Install all other requirements 
pip install --no-cache-dir --upgrade-strategy only-if-needed -r src/requirements/requirements_cpu.txt
```

### Option 3: Manual Conda Setup

If you prefer using Conda:

```bash
# Create a new conda environment
conda create -n TakuNet python=3.10 -y
conda activate TakuNet

# Install CPU-only PyTorch with extended timeout
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu --timeout 180

# Install remaining requirements
pip install --no-cache-dir --upgrade-strategy only-if-needed -r src/requirements/requirements_cpu.txt
```

## Troubleshooting

### Handling timeout errors

If you see timeout errors like this:
```
HTTPSConnectionPool(...): Read timed out.
```

Try the following:
1. Use a more stable internet connection
2. Install PyTorch packages manually first with the command in Option 2
3. If using a proxy, ensure it allows large downloads (PyTorch is ~200MB)

## Verification

To verify CPU-only installation, run:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # Should print False
``` 