#!/bin/bash
# Evaluate specific epoch on dev set with GPU-accelerated scoring

# Load environment paths
. ./path.sh || exit 1

EPOCH=30 # Change this to evaluate different epochs
SKIP_EXTRACT=false # Set to false to re-extract embeddings


echo "Evaluating epoch ${EPOCH} on tidyvoice_dev with GPU acceleration..."

# Set paths
exp_dir=exp/fusion_wav2vec2_samresnet34_voxblink_ft_tidy18
data=data
gpus="[0]"

#tidyvoice_eval_A
eval_dataset="tidyvoice_dev"
trials="trials/trials.kaldi"  # Fixed: trials is in a subdirectory

# Use specific epoch model (not averaged)
model_path=${exp_dir}/models/model_${EPOCH}.pt

# Extract embeddings for dev set (uses GPU for neural network inference)
if [ "$SKIP_EXTRACT" = false ]; then
  echo "Extracting embeddings with GPU..."
  local/extract_single_dataset.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 1 --gpus $gpus --data_type shard \
    --data ${data} --dataset $eval_dataset
else
  echo "Skipping extraction - using existing embeddings"
fi

# GPU-accelerated cosine similarity scoring
echo "Computing scores with GPU acceleration..."
store_score_dir=${exp_dir}/scores
mkdir -p ${store_score_dir}

eval_scp_path=${exp_dir}/embeddings/${eval_dataset}/xvector.scp
cal_mean_dir=${exp_dir}/embeddings/${eval_dataset}

echo "Using GPU score script: wespeaker/bin/score_gpu.py"
python wespeaker/bin/score_gpu.py \
  --exp_dir=${exp_dir} \
  --eval_scp_path=${eval_scp_path} \
  --cal_mean=True \
  --cal_mean_dir=${cal_mean_dir} \
  --use_gpu=True \
  --batch_size=100000 \
  ${data}/${eval_dataset}/${trials}

# Compute metrics (EER and MinDCF)
echo "Computing EER and MinDCF..."
score_file=${store_score_dir}/$(basename ${trials}).score
result_file=${store_score_dir}/$(basename ${trials}).result

python wespeaker/bin/compute_metrics.py \
  --p_target 0.01 \
  --c_fa 1 \
  --c_miss 1 \
  ${score_file} \
  2>&1 | tee ${result_file}

echo "================================"
echo "Results saved to: ${store_score_dir}/"
echo "Score file: $(basename ${trials}).score"
echo "Metrics file: $(basename ${trials}).result"
echo "================================"
cat ${result_file}
