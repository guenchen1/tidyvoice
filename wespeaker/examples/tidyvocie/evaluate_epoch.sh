#!/bin/bash
# Evaluate specific epoch on dev set

# Load environment paths
. ./path.sh || exit 1

EPOCH=1  # Change this to evaluate different epochs

echo "Evaluating epoch ${EPOCH} on tidyvoice_dev..."

# Set paths
exp_dir=exp/samresnet34_voxblink_ft_tidy
data=data
gpus="[0]"
eval_dataset="tidyvoice_dev"
trials="trials.kaldi"

# Use specific epoch model (not averaged)
model_path=${exp_dir}/models/model_${EPOCH}.pt

# Extract embeddings for dev set
echo "Extracting embeddings..."
local/extract_single_dataset.sh \
  --exp_dir $exp_dir --model_path $model_path \
  --nj 1 --gpus $gpus --data_type shard \
  --data ${data} --dataset $eval_dataset

# Score on dev set
echo "Computing scores..."
local/score_custom.sh \
  --stage 1 --stop-stage 2 \
  --data ${data} \
  --exp_dir $exp_dir \
  --eval_dataset $eval_dataset \
  --trials "$trials" \
  --cal_mean True

echo "Results saved to: ${exp_dir}/scores/"
echo "Check EER and MinDCF metrics in the score files"
