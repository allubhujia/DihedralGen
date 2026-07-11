#!/bin/bash

# ==============================================================================
# Unseen Peptide Pipeline Orchestrator
# This script downloads an unseen peptide from HuggingFace, runs the diffusion 
# model to predict its trajectory, and generates Ramachandran and Histogram plots.
# ==============================================================================

if [ -z "$1" ]; then
    echo "❌ Usage: ./run_unseen_pipeline.sh <PEPTIDE_SEQUENCE>"
    echo "   Example: ./run_unseen_pipeline.sh AFA"
    exit 1
fi

PEPTIDE=$(echo "$1" | tr '[:lower:]' '[:upper:]')
BASE_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")

echo "============================================================"
echo "🚀 Starting pipeline for unseen peptide: ${PEPTIDE}"
echo "============================================================"

# Ensure huggingface_hub is installed
pip install huggingface_hub -q

# 1. Preprocessing / Downloading
echo ""
echo "▶ STEP 1: Downloading & Preprocessing Data..."
python "${BASE_DIR}/unseen_peptide_test/preprocess_huggingface.py" --peptide "${PEPTIDE}"
if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed. Aborting pipeline."
    exit 1
fi

# 2. Diffusion Prediction
echo ""
echo "▶ STEP 2: Running Diffusion Model Prediction..."
python "${BASE_DIR}/trajectory_pre_/predict_trajectory.py" --peptide "${PEPTIDE}"
if [ $? -ne 0 ]; then
    echo "❌ Prediction failed. Aborting pipeline."
    exit 1
fi

# 3. Analysis & Plotting
echo ""
echo "▶ STEP 3: Generating Analysis Plots..."
python "${BASE_DIR}/trajectory_analysis(main)/dihedral_comparison.py" --peptide "${PEPTIDE}"
if [ $? -ne 0 ]; then
    echo "❌ Plotting failed."
    exit 1
fi

echo ""
echo "============================================================"
echo "🎉 PIPELINE COMPLETE!"
echo "Check 'trajectory_analysis(main)/${PEPTIDE}_plot' for your visual results!"
echo "============================================================"
