#!/usr/bin/env bash
set -e

echo "==============================================================="
echo "   IntruSense - AI Powered Intrusion Detection System Pipeline"
echo "==============================================================="
echo

# ------------------------------------------------------------
# (Optional) Activate virtual environment
# ------------------------------------------------------------
# If using venv:
# source venv/bin/activate
#
# If using conda:
# conda activate intrusense

# Helper to run a step and check for errors
run_step() {
  local step_num="$1"
  local description="$2"
  local cmd="$3"

  echo "[${step_num}] ${description}"
  # Use eval so we can pass full command as a single string
  eval "${cmd}"
  local status=$?
  if [ $status -ne 0 ]; then
    echo "ERROR running: ${cmd}"
    echo "Exit code: $status"
    exit $status
  fi
  echo
}

run_step "1/8" "Reading and merging entruSense raw data: NSL KDD & CICIDS2017 ....." \
  "python src/test_data_creation.py"

run_step "2/8" "entruSense Data Preprocessing ..." \
  "python src/data_preprocess.py"

run_step "3/8" "entruSense Supervised Model Training: RandomForest & XGBoost ..." \
  "python src/supervised_model.py"

run_step "4/8" "entruSense Autoencoder Model Training ..." \
  "python src/autoencoder_model.py"

run_step "5/8" "entruSense Hybrid Model Training ..." \
  "python src/hybrid_model.py"

run_step "6/8" "entruSense Model Evaluation ..." \
  "python src/evaluate_models.py"

run_step "7/8" "entruSense Visual Generation ..." \
  "python src/generate_visuals.py"

run_step "8/8" "entruSense Simulation ..." \
  "python src/entruSense.py"

echo "==============================================================="
echo "         All entruSense pipeline steps completed ✅"
echo "==============================================================="
