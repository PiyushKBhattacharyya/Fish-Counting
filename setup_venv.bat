@echo off
echo Setting up virtual environment for Fish Counting project...

REM Create virtual environment
python -m venv fish_counting_env

REM Activate virtual environment
call fish_counting_env\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

REM Install PyTorch with CUDA support (adjust CUDA version as needed)
REM pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Virtual environment setup complete!
echo To activate the environment in future sessions, run:
echo call fish_counting_env\Scripts\activate
pause