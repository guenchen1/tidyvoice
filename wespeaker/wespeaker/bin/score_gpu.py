#!/usr/bin/env python3
# GPU-accelerated version using CuPy for faster scoring

import os
from pathlib import Path

import fire
import kaldiio
import numpy as np
from tqdm import tqdm

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: CuPy not available, falling back to CPU (sklearn)")
    from sklearn.metrics.pairwise import cosine_similarity


def calculate_mean_from_kaldi_vec(scp_path):
    vec_num = 0
    mean_vec = None

    for _, vec in kaldiio.load_scp_sequential(scp_path):
        if mean_vec is None:
            mean_vec = np.zeros_like(vec)
        mean_vec += vec
        vec_num += 1

    return mean_vec / vec_num


def cosine_similarity_gpu(emb1, emb2):
    """Compute cosine similarity using CuPy on GPU"""
    dot_product = cp.dot(emb1, emb2)
    norm1 = cp.linalg.norm(emb1)
    norm2 = cp.linalg.norm(emb2)
    return float(dot_product / (norm1 * norm2))


def cosine_similarity_batch_gpu(emb_array1, emb_array2, batch_size=100000):
    """Batch process cosine similarity on GPU with memory-efficient approach"""
    n_samples = len(emb_array1)
    scores = np.zeros(n_samples, dtype=np.float32)
    
    # Process in batches to avoid GPU OOM
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        
        # Transfer batch to GPU
        emb1_batch = cp.asarray(emb_array1[i:end_idx], dtype=cp.float32)
        emb2_batch = cp.asarray(emb_array2[i:end_idx], dtype=cp.float32)
        
        # Normalize embeddings
        emb1_norm = emb1_batch / cp.linalg.norm(emb1_batch, axis=1, keepdims=True)
        emb2_norm = emb2_batch / cp.linalg.norm(emb2_batch, axis=1, keepdims=True)
        
        # Compute dot products (cosine similarity)
        batch_scores = cp.sum(emb1_norm * emb2_norm, axis=1)
        
        # Transfer back to CPU and store
        scores[i:end_idx] = cp.asnumpy(batch_scores)
        
        # Clear GPU memory
        del emb1_batch, emb2_batch, emb1_norm, emb2_norm, batch_scores
        cp.get_default_memory_pool().free_all_blocks()
    
    return scores


def trials_cosine_score_gpu(eval_scp_path='',
                            store_dir='',
                            mean_vec=None,
                            trials=(),
                            use_gpu=True,
                            batch_size=50000):
    if mean_vec is None or not os.path.exists(mean_vec):
        mean_vec = 0.0
    else:
        mean_vec = np.load(mean_vec)

    # Pre-load embeddings into memory
    emb_dict = {}
    print("Loading embeddings...")
    for utt, emb in tqdm(kaldiio.load_scp_sequential(eval_scp_path)):
        emb = emb - mean_vec
        emb_dict[utt] = emb
    
    print(f"Loaded {len(emb_dict)} embeddings")

    for trial in trials:
        store_path = os.path.join(store_dir,
                                  os.path.basename(trial) + '.score')
        
        print(f"Processing trial: {os.path.basename(trial)}")
        
        with open(trial, 'r') as trial_r:
            lines = trial_r.readlines()
        
        # Prepare batch data
        print(f"Preparing {len(lines)} trial pairs...")
        emb1_list = []
        emb2_list = []
        line_data = []
        
        for line in lines:
            segs = line.strip().split()
            if segs[0] in emb_dict and segs[1] in emb_dict:
                emb1_list.append(emb_dict[segs[0]])
                emb2_list.append(emb_dict[segs[1]])
                line_data.append(segs)
            else:
                print(f"Warning: Missing embedding for {segs[0]} or {segs[1]}")
        
        emb1_array = np.array(emb1_list)
        emb2_array = np.array(emb2_list)
        
        print(f"Computing {len(emb1_array)} similarity scores on GPU...")
        
        if use_gpu and GPU_AVAILABLE:
            # Batch GPU processing
            scores = cosine_similarity_batch_gpu(emb1_array, emb2_array, batch_size)
        else:
            # Fallback to CPU
            print("Using CPU (sklearn)...")
            from sklearn.metrics.pairwise import cosine_similarity
            scores = []
            for emb1, emb2 in tqdm(zip(emb1_array, emb2_array), total=len(emb1_array)):
                score = cosine_similarity(emb1.reshape(1, -1), 
                                         emb2.reshape(1, -1))[0][0]
                scores.append(score)
            scores = np.array(scores)
        
        # Write results
        print(f"Writing results to {store_path}...")
        with open(store_path, 'w') as w_f:
            for segs, cos_score in zip(line_data, scores):
                if len(segs) == 3:  # enroll_name test_name target/nontarget
                    w_f.write('{} {} {:.5f} {}\n'.format(
                        segs[0], segs[1], cos_score, segs[2]))
                else:  # enroll_name test_name
                    w_f.write('{} {} {:.5f}\n'.format(segs[0], segs[1],
                                                      cos_score))
        
        print(f"Completed: {store_path}")


def main(exp_dir, eval_scp_path, cal_mean, cal_mean_dir, *trials, 
         use_gpu=True, batch_size=50000):

    print(f"GPU acceleration: {use_gpu and GPU_AVAILABLE}")
    print(f"Batch size: {batch_size}")
    
    if not cal_mean:
        print("Do not do mean normalization for evaluation embeddings.")
        mean_vec_path = None
    else:
        scp_path = os.path.join(cal_mean_dir, 'xvector.scp')
        print("Calculate mean statistics from {}.".format(scp_path))
        mean_vec = calculate_mean_from_kaldi_vec(scp_path)
        mean_vec_path = os.path.join(cal_mean_dir, 'mean_vec.npy')
        np.save(mean_vec_path, mean_vec)

    # scoring trials
    store_score_dir = os.path.join(exp_dir, 'scores')
    Path(store_score_dir).mkdir(parents=True, exist_ok=True)
    trials_cosine_score_gpu(eval_scp_path, store_score_dir, mean_vec_path, 
                           trials, use_gpu, batch_size)


if __name__ == "__main__":
    fire.Fire(main)
