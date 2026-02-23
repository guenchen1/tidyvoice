import os
import sys
import argparse
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore")

def process_file(line):
    try:
        parts = line.strip().split()
        if len(parts) < 2:
            return None
        utt_id = parts[0]
        wav_path = parts[1]
        
        # Load audio with original sampling rate to be faster, but mono=True is default and good
        y, sr = librosa.load(wav_path, sr=None)
        
        total_samples = len(y)
        duration = total_samples / sr
        
        # VAD - Silence detection
        # top_db=30 means anything 30dB below the peak is considered silence
        non_silent_intervals = librosa.effects.split(y, top_db=30)
        
        total_non_silence_samples = 0
        for start, end in non_silent_intervals:
            total_non_silence_samples += (end - start)
            
        total_silence_samples = total_samples - total_non_silence_samples
        silence_portion = total_silence_samples / total_samples if total_samples > 0 else 0
        
        # Silence from start
        start_silence_samples = 0
        if len(non_silent_intervals) > 0:
            start_silence_samples = non_silent_intervals[0][0]
        else:
            start_silence_samples = total_samples # All silence
            
        start_silence_sec = start_silence_samples / sr
        
        # Silence from end
        end_silence_samples = 0
        if len(non_silent_intervals) > 0:
            end_silence_samples = total_samples - non_silent_intervals[-1][1]
        else:
            end_silence_samples = 0 # Already counted in start_silence for the all-silence case, effectively
            # But technically strictly from end:
            if start_silence_samples == total_samples:
                end_silence_samples = total_samples
                
        end_silence_sec = end_silence_samples / sr
        
        # Number of silence segments
        # If we have N non-silent intervals, we potentially have N+1 silence intervals (including edges)
        # But we really want to count distinct silent regions.
        # Intervals are [start, end) of non-silence.
        # Gaps between intervals are silence.
        # Plus potential start and end silence.
        
        num_silence_segments = 0
        if len(non_silent_intervals) == 0:
            num_silence_segments = 1
        else:
            # Check gap before first
            if non_silent_intervals[0][0] > 0:
                num_silence_segments += 1
            
            # Check gaps between
            for i in range(len(non_silent_intervals) - 1):
                if non_silent_intervals[i+1][0] > non_silent_intervals[i][1]:
                    num_silence_segments += 1
            
            # Check gap after last
            if non_silent_intervals[-1][1] < total_samples:
                num_silence_segments += 1

        return {
            'utt_id': utt_id,
            'duration': duration,
            'silence_portion': silence_portion,
            'start_silence_sec': start_silence_sec,
            'end_silence_sec': end_silence_sec,
            'num_silence_segments': num_silence_segments,
            'combined_edge_silence_vs_total': (start_silence_sec + end_silence_sec) / (total_silence_samples / sr) if total_silence_samples > 0 else 0
        }
    except Exception as e:
        return None

def analyze_scp(scp_path, set_name, max_workers=16):
    print(f"Analyzing {set_name} from {scp_path}...")
    with open(scp_path, 'r') as f:
        lines = f.readlines()
    
    # For testing/speed, we will process all lines as requested but with a progress bar
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use a list to catch results as they complete to show progress
        futures = [executor.submit(process_file, line) for line in lines]
        for future in tqdm(futures, total=len(lines), desc=f"Processing {set_name}"):
            res = future.result()
            if res:
                results.append(res)
                
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No data processed.")
        return

    output_file = f"{set_name.lower().replace(' ', '_')}_stats.txt"
    with open(output_file, 'w') as f:
        f.write(f"--- {set_name.upper()} Analysis Results ---\n")
        f.write(f"Total files: {len(df)}\n")
        
        # Length Distribution
        f.write("\n[Duration Distribution (seconds)]\n")
        f.write(df['duration'].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
        
        # Silence Portion
        f.write("\n\n[Silence Portion Distribution (0-1)]\n")
        f.write(df['silence_portion'].describe(percentiles=[0.25, 0.5, 0.75]).to_string())
        
        # Number of Silence Segments
        f.write("\n\n[Number of Silence Segments Distribution]\n")
        f.write(df['num_silence_segments'].describe(percentiles=[0.25, 0.5, 0.75]).to_string())
        
        # Start Silence
        f.write("\n\n[Silence from Start (seconds)]\n")
        f.write(df['start_silence_sec'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
        
        # End Silence
        f.write("\n\n[Silence from End (seconds)]\n")
        f.write(df['end_silence_sec'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
        
        # Combined Edge vs Whole Silence
        f.write("\n\n[Portion of Combined Start+End Silence vs Total Silence]\n")
        valid_silence = df[df['silence_portion'] > 0]
        f.write(valid_silence['combined_edge_silence_vs_total'].describe(percentiles=[0.25, 0.5, 0.75]).to_string())
        
        # Overall summary stats
        total_audio_time = df['duration'].sum() / 3600
        f.write(f"\n\nTotal Audio Time: {total_audio_time:.2f} hours\n")
    
    print(f"Analysis saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_scp', type=str, default='data/tidyvoice_train/wav.scp')
    parser.add_argument('--dev_scp', type=str, default='data/tidyvoice_dev/wav.scp')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    
    if os.path.exists(args.dev_scp):
        analyze_scp(args.dev_scp, "Dev Set", args.workers)
        
    if os.path.exists(args.train_scp):
        analyze_scp(args.train_scp, "Train Set", args.workers)

if __name__ == '__main__':
    main()
