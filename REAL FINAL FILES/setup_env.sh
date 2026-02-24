#!/usr/bin/env bash
# =============================================================
# MTFC Carbon Forecasting Pipeline — Environment Setup Script
# Run once to install all required Python libraries.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
# =============================================================

set -e  # Exit immediately on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo ""
echo "=============================================="
echo "  MTFC Pipeline — Environment Setup"
echo "=============================================="
echo ""

# ── 1. Check Python version ──────────────────────────────────
echo "→ Checking Python version..."
python3 --version 2>/dev/null || { echo "ERROR: python3 not found. Install Python 3.10+ from https://python.org"; exit 1; }

PY_VERSION=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PY_VERSION" -lt 10 ]; then
    echo "ERROR: Python 3.10 or higher is required. Found: $(python3 --version)"
    exit 1
fi
echo "   ✓ Python OK: $(python3 --version)"

# ── 2. Create virtual environment ────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo ""
    echo "→ Virtual environment already exists at .venv"
    echo "   To recreate it, delete the .venv folder and re-run this script."
else
    echo ""
    echo "→ Creating virtual environment at .venv ..."
    python3 -m venv "$VENV_DIR"
    echo "   ✓ Virtual environment created."
fi

# ── 3. Activate venv ─────────────────────────────────────────
echo ""
echo "→ Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "   ✓ Activated: $(which python)"

# ── 4. Upgrade pip ───────────────────────────────────────────
echo ""
echo "→ Upgrading pip..."
pip install --upgrade pip --quiet
echo "   ✓ pip upgraded."

# ── 5. Install all requirements ──────────────────────────────
echo ""
echo "→ Installing dependencies from requirements.txt..."
pip install -r "$SCRIPT_DIR/requirements.txt"
echo ""
echo "   ✓ All packages installed."

# ── 6. Verify key imports ────────────────────────────────────
echo ""
echo "→ Verifying imports..."
python3 -c "
import numpy; print(f'   ✓ numpy         {numpy.__version__}')
import pandas; print(f'   ✓ pandas        {pandas.__version__}')
import sklearn; print(f'   ✓ scikit-learn  {sklearn.__version__}')
import xgboost; print(f'   ✓ xgboost       {xgboost.__version__}')
import matplotlib; print(f'   ✓ matplotlib    {matplotlib.__version__}')
import pytest; print(f'   ✓ pytest        {pytest.__version__}')
"

# ── 7. Done ──────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  To activate the environment in your terminal:"
echo "    source .venv/bin/activate"
echo ""
echo "  To run the pipeline:"
echo "    python carbon_prediction_pipeline.py"
echo ""
echo "  To run the tests:"
echo "    pytest tests/"
echo "=============================================="
echo ""
