#!/bin/bash
# ============================================================================
# SVC Converter Setup Script for TensorDock/RunPod
# ============================================================================
#
# Run this on a fresh GPU instance (A100/A10/RTX 4090 recommended)
#
# Usage:
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh
#
# ============================================================================

set -e

echo "=============================================="
echo "SVC Converter Setup for Grilly Training Data"
echo "=============================================="

# ----------------------------------------------------------------------------
# 1. System Setup
# ----------------------------------------------------------------------------

echo "[1/6] Updating system..."
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git wget

# ----------------------------------------------------------------------------
# 2. Create Virtual Environment
# ----------------------------------------------------------------------------

echo "[2/6] Creating virtual environment..."
python3 -m venv /workspace/svc_env
source /workspace/svc_env/bin/activate

# ----------------------------------------------------------------------------
# 3. Install Dependencies
# ----------------------------------------------------------------------------

echo "[3/6] Installing dependencies..."
pip install --upgrade pip wheel setuptools -q

# Detect CUDA version and install appropriate cupy
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | cut -d',' -f1 | cut -d'.' -f1)
if [ "$CUDA_VERSION" == "12" ]; then
    pip install cupy-cuda12x -q
elif [ "$CUDA_VERSION" == "11" ]; then
    pip install cupy-cuda11x -q
else
    echo "Warning: Could not detect CUDA version, trying cuda12x"
    pip install cupy-cuda12x -q
fi

pip install spacy>=3.7.0 spacy-transformers>=1.3.0 tqdm -q

# ----------------------------------------------------------------------------
# 4. Download spaCy Model
# ----------------------------------------------------------------------------

echo "[4/6] Downloading spaCy transformer model..."
python -m spacy download en_core_web_trf

# ----------------------------------------------------------------------------
# 5. Verify GPU
# ----------------------------------------------------------------------------

echo "[5/6] Verifying GPU setup..."
python3 -c "
import spacy
spacy.require_gpu(0)
nlp = spacy.load('en_core_web_trf')
doc = nlp('Test sentence for GPU verification.')
print(f'GPU verification: SUCCESS')
print(f'Tokens: {[t.text for t in doc]}')
"

# ----------------------------------------------------------------------------
# 6. Create workspace structure
# ----------------------------------------------------------------------------

echo "[6/6] Setting up workspace..."
mkdir -p /workspace/data/input
mkdir -p /workspace/data/output
mkdir -p /workspace/scripts

# Copy converter script
cp convert_to_svc.py /workspace/scripts/

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Upload your data to /workspace/data/input/"
echo "   - temporal_dataset.jsonl (1.7GB)"
echo "   - instruct_anonymized_cleaned.json (251MB)"
echo "   - conversations_dataset_anonymized_cleaned.jsonl (62MB)"
echo ""
echo "2. Run the converter:"
echo "   source /workspace/svc_env/bin/activate"
echo "   cd /workspace/scripts"
echo "   python convert_to_svc.py \\"
echo "       --input /workspace/data/input \\"
echo "       --output /workspace/data/output \\"
echo "       --gpu 0 \\"
echo "       --batch-size 100"
echo ""
echo "3. For a quick test (100 entries):"
echo "   python convert_to_svc.py \\"
echo "       --input /workspace/data/input \\"
echo "       --output /workspace/data/output \\"
echo "       --gpu 0 \\"
echo "       --max-entries 100"
echo ""
echo "=============================================="
