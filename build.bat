@echo off
REM This script creates a virtual environment, installs dependencies,
REM pre-downloads the embedding model, and builds the rag_app.py 
REM into a single, self-contained executable.

set PYTHON_EXE=python
set VENV_DIR=venv

echo Checking for Python...
%PYTHON_EXE% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not found in the system path. Please install Python and try again.
    pause
    exit /b
)

echo --- RAG App Builder ---

REM 1. Create a virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo Creating virtual environment in '%VENV_DIR%'...
    %PYTHON_EXE% -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b
    )
)

REM 2. Activate the virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

REM 3. Install dependencies
echo Installing required libraries...
pip install --upgrade pip

REM MODIFIED: Enabled and updated the command to install the specific CUDA-enabled PyTorch.
REM The URL provided by the user is specific to their CUDA 12.9 toolkit.
echo Installing CUDA-enabled PyTorch...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
if %errorlevel% neq 0 (
    echo Failed to install PyTorch for CUDA. Please check your internet connection and NVIDIA driver compatibility.
    pause
    exit /b
)

echo Installing application dependencies...
pip install langchain langchain-google-genai langchain_community sentence-transformers PyMuPDF Pillow pyinstaller huggingface_hub faiss-cpu
if %errorlevel% neq 0 (
    echo Failed to install dependencies. Please check your internet connection and try again.
    pause
    exit /b
)

REM 4. Pre-download the embedding model
echo Pre-downloading the embedding model...
python pre_download_model.py
if %errorlevel% neq 0 (
    echo Failed to pre-download the model.
    pause
    exit /b
)

REM 5. Build the executable with PyInstaller, bundling the model
echo Building executable (rag_app.exe)...
echo This may take a few minutes.

pyinstaller --onefile --windowed --name rag_app --icon=NONE ^
--add-data "local_model/all-MiniLM-L6-v2;all-MiniLM-L6-v2" ^
rag_app.py

if %errorlevel% neq 0 (
    echo Failed to build the executable.
    pause
    exit /b
)

echo.
echo --- Build Complete! ---
echo Your executable file 'rag_app.exe' can be found in the 'dist' folder.
echo.
pause

