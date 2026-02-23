#!/usr/bin/env python3
"""
Comprehensive Speaker Verification Analysis and Visualization Tool
Generates all evaluation plots and metrics in one go
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import sys
import os

# ============================================================================
# Metric Computation Functions
# ============================================================================

def compute_pmiss_pfa_rbst(scores, labels, weights=None):
    """Compute FNR and FPR"""
    sorted_ndx = np.argsort(scores)
    labels = labels[sorted_ndx]
    if weights is not None:
        weights = weights[sorted_ndx]
    else:
        weights = np.ones(labels.shape, dtype='f8')
    
    tgt_wghts = weights * (labels == 1).astype('f8')
    imp_wghts = weights * (labels == 0).astype('f8')
    
    fnr = np.cumsum(tgt_wghts) / np.sum(tgt_wghts)
    fpr = 1 - np.cumsum(imp_wghts) / np.sum(imp_wghts)
    return fnr, fpr

def compute_eer(fnr, fpr, scores=None):
    """Compute Equal Error Rate"""
    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))
    
    if scores is not None:
        score_sort = np.sort(scores)
        return fnr[x1] + a * (fnr[x2] - fnr[x1]), score_sort[x1]
    
    return fnr[x1] + a * (fnr[x2] - fnr[x1])

def compute_min_dcf(fnr, fpr, p_target=0.01, c_miss=1, c_fa=1):
    """Compute minimum Detection Cost Function"""
    c_det = min(c_miss * fnr * p_target + c_fa * fpr * (1 - p_target))
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    return c_det / c_def

# ============================================================================
# DET and ROC Curves
# ============================================================================

def plot_det_and_roc_curves(scores, labels, output_dir, output_prefix):
    """Generate DET curve and ROC-style curve"""
    
    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer = compute_eer(fnr, fpr)
    min_dcf = compute_min_dcf(fnr, fpr)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ===== Plot 1: DET Curve =====
    p_miss = norm.ppf(fnr)
    p_fa = norm.ppf(fpr)
    
    xytick = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    xytick_labels = [f"{x*100:.2f}" for x in xytick]
    
    ax1.plot(p_fa, p_miss, 'b-', linewidth=2, label='DET Curve')
    ax1.set_xticks(norm.ppf(xytick))
    ax1.set_xticklabels(xytick_labels)
    ax1.set_yticks(norm.ppf(xytick))
    ax1.set_yticklabels(xytick_labels)
    ax1.set_xlim(norm.ppf([0.00051, 0.5]))
    ax1.set_ylim(norm.ppf([0.00051, 0.5]))
    ax1.set_xlabel("False Alarm Rate (%)", fontsize=12)
    ax1.set_ylabel("False Reject Rate (%)", fontsize=12)
    ax1.set_title("Detection Error Tradeoff (DET) Curve", fontsize=14, fontweight='bold')
    
    ax1.plot(norm.ppf(eer), norm.ppf(eer), 'ro', markersize=10, label=f'EER = {eer*100:.2f}%')
    ax1.annotate(
        f"EER = {eer*100:.2f}%\nMinDCF = {min_dcf:.3f}",
        xy=(norm.ppf(eer), norm.ppf(eer)),
        xytext=(norm.ppf(eer + 0.08), norm.ppf(eer + 0.08)),
        arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.2", fc="white", ec="black"),
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
        fontsize=11, fontweight='bold'
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # ===== Plot 2: ROC-style Curve =====
    ax2.plot(fpr * 100, fnr * 100, 'g-', linewidth=2, label='Operating Curve')
    ax2.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Random')
    ax2.plot(eer * 100, eer * 100, 'ro', markersize=10, label=f'EER = {eer*100:.2f}%')
    
    ax2.set_xlabel("False Accept Rate (FAR) %", fontsize=12)
    ax2.set_ylabel("False Reject Rate (FRR) %", fontsize=12)
    ax2.set_title("Error Rate Trade-off", fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 50])
    ax2.set_ylim([0, 50])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    stats_text = f"""Trials: {len(scores):,}
Target: {np.sum(labels):,}
Non-target: {len(labels) - np.sum(labels):,}
Score range: [{scores.min():.3f}, {scores.max():.3f}]"""
    
    ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'{output_prefix}_evaluation_curves.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ DET and ROC curves saved to: {output_file}")
    plt.close()
    
    return eer, min_dcf

# ============================================================================
# Category Analysis
# ============================================================================

def parse_utterance_info(utt_path):
    """Extract speaker ID and language from utterance path"""
    parts = utt_path.split('/')
    return parts[0], parts[1]  # speaker_id, language

def load_and_categorize_trials(trials_file, scores_file):
    """Load trials and categorize by speaker/language matching"""
    
    categories = {
        'tgt_diff_lang_vs_nontgt_diff_lang': {'scores': [], 'labels': [], 
            'desc': 'Target (Same Spk, Diff. Lang.) vs Non-target (Diff. Spk, Diff. Lang.)'},
        'tgt_diff_lang_vs_nontgt_same_lang': {'scores': [], 'labels': [], 
            'desc': 'Target (Same Spk, Diff. Lang.) vs Non-target (Diff. Spk, Same Lang.)'},
        'tgt_same_lang_vs_nontgt_diff_lang': {'scores': [], 'labels': [], 
            'desc': 'Target (Same Spk, Same Lang.) vs Non-target (Diff. Spk, Diff. Lang.)'},
        'tgt_same_lang_vs_nontgt_same_lang': {'scores': [], 'labels': [], 
            'desc': 'Target (Same Spk, Same Lang.) vs Non-target (Diff. Spk, Same Lang.)'}
    }
    
    # Load scores
    score_dict = {}
    with open(scores_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            score_dict[(parts[0], parts[1])] = float(parts[2])
    
    # Load and categorize trials
    with open(trials_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            utt1, utt2, label = parts[0], parts[1], parts[2]
            
            spk1, lang1 = parse_utterance_info(utt1)
            spk2, lang2 = parse_utterance_info(utt2)
            
            key = (utt1, utt2)
            if key not in score_dict:
                continue
            score = score_dict[key]
            
            is_target = (label == 'target')
            same_lang = (lang1 == lang2)
            
            if is_target:
                if same_lang:
                    categories['tgt_same_lang_vs_nontgt_diff_lang']['scores'].append(score)
                    categories['tgt_same_lang_vs_nontgt_diff_lang']['labels'].append(1)
                    categories['tgt_same_lang_vs_nontgt_same_lang']['scores'].append(score)
                    categories['tgt_same_lang_vs_nontgt_same_lang']['labels'].append(1)
                else:
                    categories['tgt_diff_lang_vs_nontgt_diff_lang']['scores'].append(score)
                    categories['tgt_diff_lang_vs_nontgt_diff_lang']['labels'].append(1)
                    categories['tgt_diff_lang_vs_nontgt_same_lang']['scores'].append(score)
                    categories['tgt_diff_lang_vs_nontgt_same_lang']['labels'].append(1)
            else:
                if same_lang:
                    categories['tgt_diff_lang_vs_nontgt_same_lang']['scores'].append(score)
                    categories['tgt_diff_lang_vs_nontgt_same_lang']['labels'].append(0)
                    categories['tgt_same_lang_vs_nontgt_same_lang']['scores'].append(score)
                    categories['tgt_same_lang_vs_nontgt_same_lang']['labels'].append(0)
                else:
                    categories['tgt_diff_lang_vs_nontgt_diff_lang']['scores'].append(score)
                    categories['tgt_diff_lang_vs_nontgt_diff_lang']['labels'].append(0)
                    categories['tgt_same_lang_vs_nontgt_diff_lang']['scores'].append(score)
                    categories['tgt_same_lang_vs_nontgt_diff_lang']['labels'].append(0)
    
    # Convert to numpy arrays
    for cat in categories:
        categories[cat]['scores'] = np.array(categories[cat]['scores'])
        categories[cat]['labels'] = np.array(categories[cat]['labels'])
    
    return categories

def plot_category_distributions(categories, output_dir, output_prefix):
    """Plot score distributions for all categories (4 subplots)"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    cat_names = list(categories.keys())
    colors_target = ['darkgreen', 'lightgreen', 'darkgreen', 'lightgreen']
    colors_nontarget = ['darkblue', 'purple', 'darkblue', 'purple']
    
    for idx, cat_name in enumerate(cat_names):
        cat = categories[cat_name]
        ax = axes[idx]
        
        scores, labels = cat['scores'], cat['labels']
        target_scores = scores[labels == 1]
        nontarget_scores = scores[labels == 0]
        
        fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
        eer, eer_threshold = compute_eer(fnr, fpr, scores)
        min_dcf = compute_min_dcf(fnr, fpr)
        
        ax.hist(nontarget_scores, bins=100, alpha=0.6, color=colors_nontarget[idx], 
                label='Non-target', density=True, edgecolor='black', linewidth=0.5)
        ax.hist(target_scores, bins=100, alpha=0.6, color=colors_target[idx], 
                label='Target', density=True, edgecolor='black', linewidth=0.5)
        
        # Mark EER threshold
        ax.axvline(eer_threshold, color='red', linestyle='--', linewidth=2, label='EER Threshold')
        
        title_parts = cat['desc'].split(' vs ')
        ax.set_title(f"{title_parts[0]}\nvs\n{title_parts[1]}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Score', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        metrics_text = f'EER: {eer*100:.2f}%\nMinDCF: {min_dcf:.3f}\nTrials: {len(scores):,}'
        ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('Speaker Verification Score Distributions by Category', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_file = os.path.join(output_dir, f'{output_prefix}_score_distributions_by_category.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Category distributions saved to: {output_file}")
    plt.close()

def plot_combined_distribution(categories, output_dir, output_prefix):
    """Plot combined score distribution with smooth curves"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract unique trials for each type
    cat1 = categories['tgt_diff_lang_vs_nontgt_diff_lang']
    cat2 = categories['tgt_diff_lang_vs_nontgt_same_lang']
    cat3 = categories['tgt_same_lang_vs_nontgt_diff_lang']
    
    tgt_diff_lang = cat1['scores'][cat1['labels'] == 1]
    tgt_same_lang = cat3['scores'][cat3['labels'] == 1]
    nontgt_diff_lang = cat1['scores'][cat1['labels'] == 0]
    nontgt_same_lang = cat2['scores'][cat2['labels'] == 0]
    
    def plot_smooth_density(data, color, label, alpha=0.5, linewidth=2.5):
        hist, bin_edges = np.histogram(data, bins=200, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed = gaussian_filter1d(hist, sigma=3)
        ax.fill_between(bin_centers, smoothed, alpha=alpha, color=color, label=label, linewidth=0)
        ax.plot(bin_centers, smoothed, color=color, linewidth=linewidth, alpha=0.9)
    
    plot_smooth_density(nontgt_diff_lang, 'darkblue', 'Non-target (Diff. Lang.)', alpha=0.4, linewidth=3)
    plot_smooth_density(nontgt_same_lang, 'mediumpurple', 'Non-target (Same Lang.)', alpha=0.4, linewidth=3)
    plot_smooth_density(tgt_diff_lang, 'lightgreen', 'Target (Diff. Lang.)', alpha=0.4, linewidth=3)
    plot_smooth_density(tgt_same_lang, 'darkgreen', 'Target (Same Lang.)', alpha=0.4, linewidth=3)
    
    # Mark EER thresholds
    colors_eer = ['blue', 'purple', 'lightgreen', 'green']
    linestyles = ['--', ':', '-.', '-']
    for idx, (cat_name, cat) in enumerate(categories.items()):
        fnr, fpr = compute_pmiss_pfa_rbst(cat['scores'], cat['labels'])
        eer, eer_threshold = compute_eer(fnr, fpr, cat['scores'])
        ax.axvline(eer_threshold, color=colors_eer[idx], linestyle=linestyles[idx], 
                  linewidth=2, alpha=0.8, zorder=10)
        ax.plot(eer_threshold, 0, 'o', color='yellow', markersize=10, 
               markeredgecolor=colors_eer[idx], markeredgewidth=2, zorder=11)
    
    ax.set_xlabel('Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax.set_title('Speaker Verification Score Distributions by Category', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xlim([-0.4, 1.0])
    ax.set_ylim([0, None])
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'{output_prefix}_score_distributions_combined.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Combined distribution saved to: {output_file}")
    plt.close()

def print_category_metrics(categories):
    """Print detailed metrics for each category"""
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS BY TRIAL PAIR CATEGORIES")
    print("="*80)
    
    for idx, (cat_name, cat) in enumerate(categories.items(), 1):
        scores, labels = cat['scores'], cat['labels']
        fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
        eer = compute_eer(fnr, fpr)
        min_dcf = compute_min_dcf(fnr, fpr)
        
        n_target = np.sum(labels)
        n_nontarget = len(labels) - n_target
        
        print(f"\n{idx}. {cat['desc']}")
        print("-" * 80)
        print(f"   Total Trials: {len(scores):,}")
        print(f"   Target Trials: {n_target:,}")
        print(f"   Non-target Trials: {n_nontarget:,}")
        print(f"   EER: {eer*100:.3f}%")
        print(f"   MinDCF (p=0.01): {min_dcf:.4f}")
        
        target_scores = scores[labels == 1]
        nontarget_scores = scores[labels == 0]
        print(f"   Target Score: {target_scores.mean():.3f} ± {target_scores.std():.3f}")
        print(f"   Non-target Score: {nontarget_scores.mean():.3f} ± {nontarget_scores.std():.3f}")
    
    print("\n" + "="*80)

# ============================================================================
# Main Function
# ============================================================================

def main(scores_file, trials_file=None):
    """Main comprehensive analysis function"""
    
    output_dir = os.path.dirname(scores_file)
    # Get a prefix from the score filename to avoid overwriting files when analyzing different epochs
    # e.g., "epoch_1_trials.kaldi.score" -> "epoch_1_trials.kaldi"
    file_prefix = os.path.basename(scores_file)
    if file_prefix.endswith('.score'):
        file_prefix = file_prefix[:-6]
    
    print("\n" + "="*80)
    print("COMPREHENSIVE SPEAKER VERIFICATION ANALYSIS")
    print("="*80)
    print(f"Score file: {scores_file}")
    if trials_file:
        print(f"Trials file: {trials_file}")
    print("="*80 + "\n")
    
    # Load scores
    print("[1/5] Loading scores...")
    scores = []
    labels = []
    with open(scores_file) as f:
        for line in f:
            parts = line.strip().split()
            scores.append(float(parts[2]))
            labels.append(parts[3] == 'target')
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    print(f"✓ Loaded {len(scores):,} trials ({np.sum(labels):,} target, {len(labels)-np.sum(labels):,} non-target)")
    
    # Compute overall metrics
    print("\n[2/5] Computing overall metrics...")
    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer = compute_eer(fnr, fpr)
    min_dcf = compute_min_dcf(fnr, fpr)
    
    print(f"✓ Overall EER: {eer*100:.3f}%")
    print(f"✓ Overall MinDCF: {min_dcf:.4f}")
    
    # Generate DET and ROC curves
    print("\n[3/5] Generating DET and ROC curves...")
    plot_det_and_roc_curves(scores, labels, output_dir, file_prefix)
    
    # Category analysis (if trials file provided)
    if trials_file and os.path.exists(trials_file):
        print("\n[4/5] Analyzing by trial pair categories...")
        categories = load_and_categorize_trials(trials_file, scores_file)
        
        print_category_metrics(categories)
        
        print("\n[5/5] Generating category distribution plots...")
        plot_category_distributions(categories, output_dir, file_prefix)
        plot_combined_distribution(categories, output_dir, file_prefix)
        
        # Save metrics to file
        metrics_file = os.path.join(output_dir, f'{file_prefix}_category_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write("DETAILED ANALYSIS BY TRIAL PAIR CATEGORIES\n")
            f.write("="*80 + "\n\n")
            for idx, (cat_name, cat) in enumerate(categories.items(), 1):
                fnr, fpr = compute_pmiss_pfa_rbst(cat['scores'], cat['labels'])
                eer = compute_eer(fnr, fpr)
                min_dcf = compute_min_dcf(fnr, fpr)
                f.write(f"{idx}. {cat['desc']}\n")
                f.write(f"   EER: {eer*100:.3f}%\n")
                f.write(f"   MinDCF: {min_dcf:.4f}\n")
                f.write(f"   Trials: {len(cat['scores']):,}\n\n")
        print(f"✓ Metrics saved to: {metrics_file}")
    else:
        print("\n[4/5] Skipping category analysis (no trials file)")
        print("[5/5] Skipping category plots")
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in: {output_dir}/")
    print("  - evaluation_curves.png (DET + ROC curves)")
    if trials_file and os.path.exists(trials_file):
        print("  - score_distributions_by_category.png (4 category plots)")
        print("  - score_distributions_combined.png (combined smooth curves)")
        print("  - category_metrics.txt (detailed metrics)")
    print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_analysis.py <scores_file> [trials_file]")
        print("\nExample:")
        print("  python comprehensive_analysis.py scores/trials.kaldi.score")
        print("  python comprehensive_analysis.py scores/trials.kaldi.score data/trials/trials.kaldi")
        sys.exit(1)
    
    scores_file = sys.argv[1]
    trials_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(scores_file, trials_file)
