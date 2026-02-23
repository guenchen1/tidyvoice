#!/bin/bash

# Auto Analysis Script called by train.py
# Uses CPU for extraction and scoring to avoid OOM during training

EPOCH=$1
EXP_DIR=$2
DATASET=${3:-"tidyvoice_dev"}

. ./path.sh

echo "[AutoAnalysis] Starting analysis for Epoch ${EPOCH} (CPU Mode)"
echo "[AutoAnalysis] Exp: ${EXP_DIR}"

MODEL_PATH="${EXP_DIR}/models/model_${EPOCH}.pt"
DATA_DIR="data"
TRIALS="data/${DATASET}/trials/trials.kaldi"

# 1. Extraction (GPU Mode)
# Warning: This may cause OOM if running on the same GPU as training process
echo "[AutoAnalysis] Extracting embeddings (GPU Mode)..."
mkdir -p "${EXP_DIR}/embeddings/${DATASET}"
local/extract_single_dataset.sh \
  --exp_dir $EXP_DIR \
  --model_path $MODEL_PATH \
  --nj 1 \
  --gpus "[0]" \
  --data_type shard \
  --data $DATA_DIR \
  --dataset $DATASET > ${EXP_DIR}/embeddings/${DATASET}/extract_epoch_${EPOCH}.log 2>&1

if [ $? -ne 0 ]; then
    echo "[AutoAnalysis] Extraction failed. See log."
    exit 1
fi

# 2. Scoring (CPU)
echo "[AutoAnalysis] Scoring..."
mkdir -p ${EXP_DIR}/scores
python wespeaker/bin/score.py \
  --exp_dir ${EXP_DIR} \
  --eval_scp_path ${EXP_DIR}/embeddings/${DATASET}/xvector.scp \
  --cal_mean True \
  --cal_mean_dir ${EXP_DIR}/embeddings/${DATASET} \
  ${TRIALS}

# Rename score file
SCORE_FILE="${EXP_DIR}/scores/trials.kaldi.score"
EPOCH_SCORE="${EXP_DIR}/scores/epoch_${EPOCH}_trials.kaldi.score"

if [ -f "$SCORE_FILE" ]; then
    mv $SCORE_FILE $EPOCH_SCORE
else
    echo "[AutoAnalysis] Score file not found."
    exit 1
fi

# 3. Plotting
echo "[AutoAnalysis] Plotting..."

# comprehensive_analysis.py uses the score filename prefix to name output files
# e.g. epoch_1_trials.kaldi.score -> epoch_1_trials.kaldi_evaluation_curves.png
# This prevents overwrites.

# Capture python output to log
python comprehensive_analysis.py "${EPOCH_SCORE}" "${TRIALS}"

echo "[AutoAnalysis] Epoch ${EPOCH} analysis completed."
