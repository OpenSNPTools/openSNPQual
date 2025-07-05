"""
Build script for creating OpenSNPQual Windows executable
"""

import os
import sys
import subprocess
import shutil

# Create setup.py for proper packaging
SETUP_PY = """
from setuptools import setup, find_packages

setup(
    name="OpenSNPQual",
    version="1.0.0",
    description="S-Parameter Quality Evaluation Tool",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'scikit-rf>=0.20.0',
        'matplotlib>=3.4.0',
    ],
    entry_points={
        'console_scripts': [
            'opensnpqual=opensnpqual:main',
        ],
    },
)
"""

# PyInstaller spec file for advanced build options
SPEC_FILE = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['opensnpqual.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'scipy.special._ufuncs_cxx',
        'scipy._lib.messagestream',
        'sklearn.utils._typedefs',
        'sklearn.neighbors._partition_nodes',
        'pandas._libs.tslibs.timedeltas',
        'pandas._libs.tslibs.nattype',
        'pandas._libs.tslibs.np_datetime',
        'pandas._libs.skiplist',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='OpenSNPQual',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='opensnpqual.ico',
)
"""

# Windows batch file for building
BATCH_FILE = """@echo off
echo Building OpenSNPQual Windows Executable...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv build_env
call build_env\\Scripts\\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install required packages
echo Installing dependencies...
pip install numpy pandas scipy scikit-rf matplotlib pyinstaller

REM Build executable
echo Building executable...
pyinstaller --onefile --windowed --name OpenSNPQual opensnpqual.py

REM Copy executable to dist folder
if exist dist\\OpenSNPQual.exe (
    echo.
    echo Build successful! Executable located at: dist\\OpenSNPQual.exe
    
    REM Create installer package
    echo Creating installer package...
    mkdir OpenSNPQual_Package
    copy dist\\OpenSNPQual.exe OpenSNPQual_Package\\
    copy README.md OpenSNPQual_Package\\ 2>nul
    copy LICENSE OpenSNPQual_Package\\ 2>nul
    
    echo.
    echo Package created in OpenSNPQual_Package folder
) else (
    echo.
    echo Build failed!
)

REM Cleanup
call deactivate
pause
"""

# Linux/Mac shell script for building
SHELL_SCRIPT = """#!/bin/bash

echo "Building OpenSNPQual Executable..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv build_env
source build_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install required packages
echo "Installing dependencies..."
pip install numpy pandas scipy scikit-rf matplotlib pyinstaller

# Build executable
echo "Building executable..."
pyinstaller --onefile --windowed --name OpenSNPQual opensnpqual.py

# Check if build was successful
if [ -f "dist/OpenSNPQual" ]; then
    echo
    echo "Build successful! Executable located at: dist/OpenSNPQual"
    
    # Create package
    echo "Creating package..."
    mkdir -p OpenSNPQual_Package
    cp dist/OpenSNPQual OpenSNPQual_Package/
    [ -f README.md ] && cp README.md OpenSNPQual_Package/
    [ -f LICENSE ] && cp LICENSE OpenSNPQual_Package/
    
    echo
    echo "Package created in OpenSNPQual_Package folder"
else
    echo
    echo "Build failed!"
fi

# Cleanup
deactivate
"""

# README file
README = """# OpenSNPQual - S-Parameter Quality Evaluation Tool

## Overview
OpenSNPQual is a utility for evaluating the quality of S-parameter (touchstone) files according to IEEE P370 standards.

## Features
- Evaluate Passivity, Reciprocity, and Causality metrics
- Both frequency and time domain analysis
- GUI and CLI interfaces
- Export results to CSV and Markdown formats
- Color-coded quality indicators

## Installation

### Windows
1. Download OpenSNPQual.exe from the releases
2. Place in desired directory
3. Optional: Add to PATH for command-line usage

### From Source
```bash
pip install numpy pandas scipy scikit-rf matplotlib
python opensnpqual.py
```

## Usage

### GUI Mode
Double-click OpenSNPQual.exe or run:
```
OpenSNPQual.exe
```

Right-click any .s*p file and select "Open with" > OpenSNPQual for automatic loading.

### CLI Mode
```
OpenSNPQual.exe --cli -i input_files.csv -o output_prefix
```

## Quality Metrics

### Passivity
Verifies that the network doesn't generate energy (|S| ≤ 1)

### Reciprocity  
Checks if S_ij = S_ji for passive networks

### Causality
Evaluates if the network response is causal (no response before excitation)

## Quality Levels
- 🟢 **Great**: < 0.01
- 🔵 **Good**: 0.01 - 0.05  
- 🟡 **Acceptable**: 0.05 - 0.1
- 🔴 **Bad**: > 0.1

## License
[Your License Here]

## Credits
Based on IEEE P370 standards for S-parameter quality metrics.
"""

def build_windows_exe():
    """Build Windows executable"""
    print("Building OpenSNPQual for Windows...")
    
    # Write build files
    with open('setup.py', 'w') as f:
        f.write(SETUP_PY)
    
    with open('OpenSNPQual.spec', 'w') as f:
        f.write(SPEC_FILE)
    
    with open('build_windows.bat', 'w') as f:
        f.write(BATCH_FILE)
    
    with open('README.md', 'w') as f:
        f.write(README)
    
    print("\nBuild files created!")
    print("To build the executable:")
    print("1. Run: build_windows.bat")
    print("2. The executable will be in the 'dist' folder")
    
    # Optionally run the build
    response = input("\nRun build now? (y/n): ")
    if response.lower() == 'y':
        subprocess.run(['build_windows.bat'], shell=True)

def build_unix_exe():
    """Build Unix/Linux/Mac executable"""
    print("Building OpenSNPQual for Unix/Linux/Mac...")
    
    with open('build_unix.sh', 'w') as f:
        f.write(SHELL_SCRIPT)
    
    # Make script executable
    os.chmod('build_unix.sh', 0o755)
    
    print("\nBuild script created!")
    print("To build the executable:")
    print("1. Run: ./build_unix.sh")
    print("2. The executable will be in the 'dist' folder")

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        build_windows_exe()
    else:
        build_unix_exe()
