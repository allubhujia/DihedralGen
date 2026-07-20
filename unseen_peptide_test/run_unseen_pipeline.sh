#!/bin/bash

# ==============================================================================
# Unseen Peptide Pipeline Orchestrator
# This script downloads an unseen peptide from HuggingFace, runs the Transformer
# trajectory model to predict its dihedral distribution, and generates
# Ramachandran and histogram plots.
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

# 2. Trajectory Prediction
echo ""
echo "▶ STEP 2: Running Trajectory Model Prediction..."
python "${BASE_DIR}/trajectory_pre_/predict_trajectory.py" --peptide "${PEPTIDE}"
if [ $? -ne 0 ]; then
    echo "❌ Prediction failed. Aborting pipeline."
    exit 1
fi

# 3. Stage predictions where dihedral_comparison.py expects them.
# predict_trajectory.py writes to trajectory_pre_/predictions/{PEPTIDE}/, but
# dihedral_comparison.py reads from trajectory_pdb_files/{PEPTIDE}/ — copy across.
mkdir -p "${BASE_DIR}/trajectory_pdb_files/${PEPTIDE}"
cp "${BASE_DIR}/trajectory_pre_/predictions/${PEPTIDE}/"* "${BASE_DIR}/trajectory_pdb_files/${PEPTIDE}/"

# 4. Analysis & Plotting
echo ""
echo "▶ STEP 4: Generating Analysis Plots..."
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
