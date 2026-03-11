# install.ps1 - Set up a Python virtual environment and install dependencies
# Usage: .\install.ps1

$ErrorActionPreference = "Stop"

$VenvDir = "venv"

Write-Host "Creating Python virtual environment in '$VenvDir'..."
python -m venv $VenvDir

Write-Host "Activating virtual environment..."
& "$VenvDir\Scripts\Activate.ps1"

Write-Host "Upgrading pip..."
pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete. Activate the environment with:"
Write-Host "  .\venv\Scripts\Activate.ps1"
