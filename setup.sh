#!/bin/bash
# Setup script for Trans_NeXt_Conv

set -e

echo "=== Trans_NeXt_Conv Setup ==="

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# 2. Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found."
fi

# Install optional dev dependencies
pip install ipykernel black flake8

# 3. Create directories
echo "Creating project directories..."
mkdir -p output
mkdir -p data
mkdir -p src/evaluation
mkdir -p Trans_next_Conv/images/evaluation_results

# 4. Prompt for Kaggle API
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "NOTE: Kaggle API key not found at ~/.kaggle/kaggle.json"
    echo "To download the dataset, please place your kaggle.json file there."
    echo "chmod 600 ~/.kaggle/kaggle.json"
fi

echo ""
echo "=== Setup Complete! ==="
echo "Activate environment with: source venv/bin/activate"
