#!/bin/bash
# Quick start script for new users

set -e  # Exit on error

echo "=========================================="
echo "Atmospheric Surrogate Model - Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"
echo ""

# Run tests
echo "Running tests..."
python3 -m pytest test_utils.py -v --tb=short
echo ""

# Check if data exists
echo "Checking for data files..."
if ls *.npz 1> /dev/null 2>&1; then
    echo "✓ Found data files:"
    ls -lh *.npz | awk '{print "  -", $9, "(" $5 ")"}'
else
    echo "⚠ No data files found (.npz)"
    echo ""
    echo "To use this model, you need atmospheric profile data."
    echo "See DATA.md for instructions on preparing your data."
fi
echo ""

# Show next steps
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Prepare your data (see DATA.md):"
echo "   - Format: NPZ file with logp_arr, T_arr, q_arr, Ts_K, Fnet_arr"
echo ""
echo "2. Run fine-tuning:"
echo "   python run_finetune.py \\"
echo "     --pretrained_ckpt pretrained_ckpt/base_model_10k.pt \\"
echo "     --target_data your_data.npz \\"
echo "     --job_name my_experiment \\"
echo "     --epochs 50"
echo ""
echo "3. Check results in output_my_experiment/"
echo ""
echo "For more information:"
echo "  - README.md: Project overview and usage"
echo "  - DATA.md: Data preparation guide"
echo "  - CONTRIBUTING.md: How to contribute"
echo ""
