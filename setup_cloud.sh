#!/bin/bash

set -e

echo "========================================="
echo "Ergonomic Chair AI Training - Cloud Setup"
echo "========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

echo "Installing dependencies..."
pip install --quiet --upgrade pip

# Check if dependencies are already installed
if python3 -c "import pybullet; import stable_baselines3; import torch" 2>/dev/null; then
    echo "Dependencies already installed."
else
    echo "Installing dependencies (this may take a few minutes)..."
    pip install --quiet --index-url https://pypi.org/simple/ pybullet numpy gymnasium matplotlib
    pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch
    pip install --quiet --index-url https://pypi.org/simple/ stable-baselines3
fi

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Usage:"
echo "  1. Quick test (CPU):"
echo "     python train.py --timesteps 1000 --n-envs 2 --no-gpu"
echo ""
echo "  2. Standard training (CPU):"
echo "     python train.py --timesteps 100000 --n-envs 4 --no-gpu"
echo ""
echo "  3. Long training with checkpoints:"
echo "     python train.py --timesteps 500000 --n-envs 4 --save-freq 50000"
echo ""
echo "  4. Continue from checkpoint:"
echo "     python train.py --load-path ./models/chair_ppo_final_xxx.zip --timesteps 100000"
echo ""
