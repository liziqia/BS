"""
Generate two figures:
  Figure 1: Original training set histogram (sorted by count descending)
  Figure 2: Composite figure with stacked histograms (original + GQW augmented)
             and 4 lines (recall/F1 before/after augmentation)
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = '/root/autodl-tmp/Fusion1/OmniGen/Classify/data/original'
BASELINE_REPORT = '/root/autodl-tmp/Fusion1/OmniGen/Classify/results/metrics/baseline/classification_report.json'
OURS_METRICS = '/root/autodl-tmp/Fusion1/OmniGen/Classify/ablation_synthetic_dataset1711/exp_02_weighted_random/metrics/metrics.json'
GQW_LOG = '/root/autodl-tmp/Fusion1/OmniGen/Classify/ablation_synthetic_dataset1711/exp_02_weighted_random/synthetic_data/selection_log.json'
FIGURE_DIR = '/root/autodl-tmp/Fusion1/OmniGen/Classify/results/figures'


def load_original_counts():
    counts = {}
    for d in os.listdir(DATA_DIR):
        dpath = os.path.join(DATA_DIR, d)
        if os.path.isdir(dpath):
            n = len([f for f in os.listdir(dpath)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            counts[d] = n
    return counts


def load_gtqw_allocations():
    with open(GQW_LOG, 'r') as f:
        log = json.load(f)
    alloc = {}
    alloc.update(log['original_data_counts'])
    for cls, n_syn in log['allocated_counts'].items():
        alloc[cls] = alloc.get(cls, 0) + n_syn
    syn_only = {cls: n for cls, n in log['allocated_counts'].items()}
    return alloc, syn_only


def load_metrics():
    with open(BASELINE_REPORT, 'r') as f:
        baseline_report = json.load(f)
    with open(OURS_METRICS, 'r') as f:
        ours = json.load(f)

    all_ids = {str(i) for i in range(23)}

    baseline_recall = {}
    baseline_f1 = {}
    for cls in baseline_report:
        if cls not in all_ids:
            continue
        baseline_recall[cls] = baseline_report[cls]['recall']
        baseline_f1[cls] = baseline_report[cls]['f1-score']

    ours_recall = ours['per_class_accuracy']
    ours_f1 = ours['per_class_f1']

    return baseline_recall, baseline_f1, ours_recall, ours_f1


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)

    orig_counts = load_original_counts()
    combined_counts, syn_counts = load_gtqw_allocations()
    baseline_recall, baseline_f1, ours_recall, ours_f1 = load_metrics()

    all_class_ids = sorted(orig_counts.keys(), key=lambda c: orig_counts[c], reverse=True)
    n_classes = len(all_class_ids)

    x = np.arange(n_classes)

    orig_vals = np.array([orig_counts[c] for c in all_class_ids])
    syn_vals = np.array([syn_counts.get(c, 0) for c in all_class_ids])

    baseline_r_vals = [baseline_recall.get(c, 0) for c in all_class_ids]
    baseline_f_vals = [baseline_f1.get(c, 0) for c in all_class_ids]
    ours_r_vals = [ours_recall.get(c, 0) for c in all_class_ids]
    ours_f_vals = [ours_f1.get(c, 0) for c in all_class_ids]

    # ==========================================
    # Figure 1: Original training set histogram
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, orig_vals, color='steelblue', edgecolor='white', alpha=0.85, width=0.7)
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Number of Images')
    ax.set_xticks(x)
    ax.set_xticklabels(all_class_ids)
    ax.set_ylim(0, max(orig_vals) * 1.12)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 3, str(int(h)),
                ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path1 = os.path.join(FIGURE_DIR, 'original_distribution.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure 1 saved: {path1}')

    # ==========================================
    # Figure 2: Composite figure
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(14, 9))
    ax2 = ax1.twinx()

    width = 0.7

    # ---- histogram: original (steelblue) -------------------------------------
    bars_orig = ax1.bar(x, orig_vals, width, color='steelblue', edgecolor='white',
                         alpha=0.85, label='Original', zorder=3)

    # ---- histogram: GQW augmented on top (orange) -----------------------------
    bars_syn = ax1.bar(x, syn_vals, width, bottom=orig_vals, color='darkorange',
                        edgecolor='white', alpha=0.85, label='GQW Augmented', zorder=3)

    # ---- sum count labels on top ----------------------------------------------
    for i in range(n_classes):
        total = orig_vals[i] + syn_vals[i]
        if syn_vals[i] > 0:
            ax1.text(i, total + 3, str(int(total)),
                     ha='center', va='bottom', fontsize=7, fontweight='bold')

    # ---- 4 lines --------------------------------------------------------------
    lw_line = 1.8
    ax2.plot(x, baseline_r_vals, 'b--', linewidth=lw_line, label='Recall (Baseline)',
             alpha=0.75)
    ax2.plot(x, ours_r_vals, 'b-', linewidth=lw_line, label='Recall (Ours)',
             alpha=0.95)
    ax2.plot(x, baseline_f_vals, 'g--', linewidth=lw_line, label='F1 (Baseline)',
             alpha=0.75)
    ax2.plot(x, ours_f_vals, 'g-', linewidth=lw_line, label='F1 (Ours)',
             alpha=0.95)

    # ---- vertical divider between head (5) and tail ---------------------------
    divider_x = 4.5
    ax1.axvline(x=divider_x, color='black', linestyle='-', linewidth=1.5)
    ax1.text(2, ax1.get_ylim()[1] * 0.96, 'HEAD', ha='center', fontsize=12,
             fontweight='bold', color='black')
    ax1.text(n_classes / 2 + 2, ax1.get_ylim()[1] * 0.96, 'TAIL', ha='center',
             fontsize=12, fontweight='bold', color='black')

    # ---- axis settings --------------------------------------------------------
    ax1.set_xlabel('Class ID')
    ax1.set_ylabel('Number of Images')
    ax2.set_ylabel('Recall / F1')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_class_ids)
    ax1.set_ylim(0, max(orig_vals + syn_vals) * 1.15)
    ax2.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.25)

    # ---- combined legend ------------------------------------------------------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9,
               ncol=2)

    plt.tight_layout()
    path2 = os.path.join(FIGURE_DIR, 'gqw_distribution_composite.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure 2 saved: {path2}')


if __name__ == '__main__':
    main()
