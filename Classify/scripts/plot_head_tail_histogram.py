"""
Plot head vs tail class accuracy histogram comparing all methods.

Generates two sets of outputs for each metric (Recall + F1):
  1. Full (23 classes) — standard
  2. Filtered (20 classes, excluding 16, 21, 22) — removes trivial classes
"""
import json
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HEAD_CLASS_IDS = {'2', '0', '17', '4', '6'}          # top-5 by training sample count
ALL_CLASS_IDS = {str(i) for i in range(23)}           # all 23 class IDs (0-22)
TAIL_CLASS_IDS = ALL_CLASS_IDS - HEAD_CLASS_IDS

EXCLUDED_IDS = {'16', '21', '22'}                      # trivial classes, excluded in filtered run

BASE_METRICS_DIR = '/root/autodl-tmp/Fusion1/OmniGen/Classify/results/metrics'
BASE_FIGURES_DIR = '/root/autodl-tmp/Fusion1/OmniGen/Classify/results/figures'
DATA_DIR = '/root/autodl-tmp/Fusion1/OmniGen/Classify/data/original'

OURS_METRICS_PATH = '/root/autodl-tmp/Fusion1/OmniGen/Classify/ablation_synthetic_dataset1711/exp_02_weighted_random/metrics/metrics.json'


def load_data():
    all_metrics = {}
    all_reports = {}
    per_class_acc = {}

    for method_dir in sorted(os.listdir(BASE_METRICS_DIR)):
        method_path = os.path.join(BASE_METRICS_DIR, method_dir)
        if not os.path.isdir(method_path):
            continue
        metrics_path = os.path.join(method_path, 'metrics.json')
        report_path = os.path.join(method_path, 'classification_report.json')
        if os.path.exists(metrics_path) and os.path.exists(report_path):
            with open(metrics_path, 'r') as f:
                all_metrics[method_dir] = json.load(f)
            with open(report_path, 'r') as f:
                all_reports[method_dir] = json.load(f)
            per_class_acc[method_dir] = all_metrics[method_dir]['per_class_accuracy']

    methods = sorted(all_metrics.keys())
    print(f"Found {len(methods)} methods: {methods}")
    return all_metrics, all_reports, per_class_acc, methods


def load_ours_data():
    with open(OURS_METRICS_PATH, 'r') as f:
        ours_metrics = json.load(f)
    return ours_metrics


def count_training_samples(data_dir):
    class_counts = {}
    for d in os.listdir(data_dir):
        dpath = os.path.join(data_dir, d)
        if os.path.isdir(dpath):
            count = len([f for f in os.listdir(dpath) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            class_counts[d] = count
    return class_counts


def generate_all_outputs(all_metrics, all_reports, per_class_metric, methods,
                         class_counts, suffix="", excluded_ids=None,
                         metric_label="Recall"):
    if excluded_ids is None:
        excluded_ids = set()

    working_ids = ALL_CLASS_IDS - excluded_ids
    working_head = HEAD_CLASS_IDS - excluded_ids
    working_tail = TAIL_CLASS_IDS - excluded_ids

    def _fname(base):
        name, ext = os.path.splitext(base)
        return f"{name}{suffix}{ext}"

    baseline_metric = per_class_metric['baseline']
    ours_metric = per_class_metric['ours']

    baseline_head_vals = sorted([baseline_metric[cls] for cls in working_head])
    baseline_tail_vals = sorted([baseline_metric[cls] for cls in working_tail])
    ours_head_vals = sorted([ours_metric[cls] for cls in working_head])
    ours_tail_vals = sorted([ours_metric[cls] for cls in working_tail])

    # ---- JSON ----------------------------------------------------------------
    per_class_comparison = {}
    for cls in sorted(working_ids, key=int):
        is_head = cls in working_head
        entry = {
            "group": "head" if is_head else "tail",
            "support": int(all_reports["baseline"][cls]["support"]),
        }
        for method in methods:
            if method in all_reports:
                entry[method] = {
                    "precision": round(all_reports[method][cls]["precision"], 4),
                    "recall": round(all_reports[method][cls]["recall"], 4),
                    "f1": round(all_reports[method][cls]["f1-score"], 4),
                }
        if 'ours' in per_class_metric and 'ours' not in all_reports:
            entry['ours'] = {
                "precision": round(per_class_metric.get('ours_precision', {}).get(cls, 0), 4),
                "recall": round(per_class_metric.get('ours_recall', {}).get(cls, 0), 4),
                "f1": round(per_class_metric.get('ours_f1', {}).get(cls, 0), 4),
            }
        per_class_comparison[cls] = entry

    comparison_output = {
        "description": f"Per-class {metric_label} comparison across all methods",
        "methods": methods + ['ours'],
        "head_class_ids": sorted(working_head, key=int),
        "tail_class_ids": sorted(working_tail, key=int),
        "per_class": per_class_comparison,
    }

    json_path = os.path.join(BASE_METRICS_DIR, _fname("per_class_comparison.json"))
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(comparison_output, f, indent=4, ensure_ascii=False)
    print(f"Per-class comparison JSON saved to: {json_path}")

    # ---- Figure 1: 2x2 histogram ---------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bins = np.linspace(0, 1.0, 11)

    ax = axes[0, 0]
    ax.hist(baseline_head_vals, bins=bins, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(baseline_head_vals), color='darkblue', linestyle='--', linewidth=2,
               label=f'Avg: {np.mean(baseline_head_vals):.3f}')
    ax.set_xlabel(metric_label); ax.set_ylabel('Number of Classes')
    ax.set_title(f'Baseline - Head Classes (n={len(baseline_head_vals)})')
    ax.set_xlim(0, 1.0); ax.legend(); ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 1]
    ax.hist(baseline_tail_vals, bins=bins, color='lightcoral', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(baseline_tail_vals), color='darkred', linestyle='--', linewidth=2,
               label=f'Avg: {np.mean(baseline_tail_vals):.3f}')
    ax.set_xlabel(metric_label); ax.set_ylabel('Number of Classes')
    ax.set_title(f'Baseline - Tail Classes (n={len(baseline_tail_vals)})')
    ax.set_xlim(0, 1.0); ax.legend(); ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 0]
    ax.hist(ours_head_vals, bins=bins, color='seagreen', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(ours_head_vals), color='darkgreen', linestyle='--', linewidth=2,
               label=f'Avg: {np.mean(ours_head_vals):.3f}')
    ax.set_xlabel(metric_label); ax.set_ylabel('Number of Classes')
    ax.set_title(f'Ours - Head Classes (n={len(ours_head_vals)})')
    ax.set_xlim(0, 1.0); ax.legend(); ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 1]
    ax.hist(ours_tail_vals, bins=bins, color='darkorange', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(ours_tail_vals), color='saddlebrown', linestyle='--', linewidth=2,
               label=f'Avg: {np.mean(ours_tail_vals):.3f}')
    ax.set_xlabel(metric_label); ax.set_ylabel('Number of Classes')
    ax.set_title(f'Ours - Tail Classes (n={len(ours_tail_vals)})')
    ax.set_xlim(0, 1.0); ax.legend(); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(BASE_FIGURES_DIR, _fname("head_tail_histogram.png"))
    os.makedirs(os.path.dirname(path1), exist_ok=True)
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {path1}")

    # ---- Figure 2a: head overlay histogram (independent file) -----------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(baseline_head_vals, bins=bins, color='steelblue', edgecolor='white', alpha=0.6, label='Baseline')
    ax.hist(ours_head_vals, bins=bins, color='seagreen', edgecolor='white', alpha=0.6, label='Ours')
    ax.axvline(np.mean(baseline_head_vals), color='darkblue', linestyle='--', linewidth=2)
    ax.axvline(np.mean(ours_head_vals), color='darkgreen', linestyle='--', linewidth=2)
    ax.set_xlabel(metric_label); ax.set_ylabel('Number of Classes')
    ax.set_title(f'Head Classes (n={len(baseline_head_vals)})\nBaseline avg={np.mean(baseline_head_vals):.3f}  |  Ours avg={np.mean(ours_head_vals):.3f}')
    ax.set_xlim(0, 1.0); ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path2_head = os.path.join(BASE_FIGURES_DIR, _fname("head_tail_overlay_histogram_head.png"))
    plt.savefig(path2_head, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {path2_head}")

    # ---- Figure 2b: tail overlay histogram (independent file) -----------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(baseline_tail_vals, bins=bins, color='lightcoral', edgecolor='white', alpha=0.6, label='Baseline')
    ax.hist(ours_tail_vals, bins=bins, color='darkorange', edgecolor='white', alpha=0.6, label='Ours')
    ax.axvline(np.mean(baseline_tail_vals), color='darkred', linestyle='--', linewidth=2)
    ax.axvline(np.mean(ours_tail_vals), color='saddlebrown', linestyle='--', linewidth=2)
    ax.set_xlabel(metric_label); ax.set_ylabel('Number of Classes')
    ax.set_title(f'Tail Classes (n={len(baseline_tail_vals)})\nBaseline avg={np.mean(baseline_tail_vals):.3f}  |  Ours avg={np.mean(ours_tail_vals):.3f}')
    ax.set_xlim(0, 1.0); ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path2_tail = os.path.join(BASE_FIGURES_DIR, _fname("head_tail_overlay_histogram_tail.png"))
    plt.savefig(path2_tail, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {path2_tail}")

    # ---- Figure 3: per-class bar chart ----------------------------------------
    fig, ax = plt.subplots(figsize=(16, 7))

    head_sorted = sorted(working_head, key=lambda c: class_counts.get(c, 0), reverse=True)
    tail_sorted = sorted(working_tail, key=lambda c: class_counts.get(c, 0), reverse=True)
    ordered_classes = head_sorted + tail_sorted

    x = np.arange(len(ordered_classes))
    width = 0.35

    baseline_vals = [baseline_metric[c] for c in ordered_classes]
    ours_vals = [ours_metric[c] for c in ordered_classes]

    ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='steelblue', alpha=0.85)
    ax.bar(x + width/2, ours_vals, width, label='Ours', color='darkorange', alpha=0.85)

    divider_x = len(head_sorted) - 0.5
    ax.axvline(x=divider_x, color='black', linestyle='-', linewidth=2)
    ax.text(len(head_sorted)/2 - 0.5, 1.02, 'HEAD Classes', ha='center', fontsize=12, fontweight='bold',
            transform=ax.get_xaxis_transform())
    ax.text(len(head_sorted) + len(tail_sorted)/2 - 0.5, 1.02, 'TAIL Classes', ha='center', fontsize=12, fontweight='bold',
            transform=ax.get_xaxis_transform())

    ax.set_xlabel('Class ID'); ax.set_ylabel(metric_label)
    ax.set_xticks(x); ax.set_xticklabels(ordered_classes)
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path3 = os.path.join(BASE_FIGURES_DIR, _fname("head_tail_per_class_bar.png"))
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {path3}")

    # ---- CSV table -------------------------------------------------------------
    method_order = ['ours', 'baseline']
    for m in methods:
        if m not in method_order:
            method_order.append(m)

    ordered_class_ids = sorted(working_ids, key=int)

    table_path = os.path.join(BASE_METRICS_DIR, _fname("head_tail_accuracy_table.csv"))
    with open(table_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Category"] + method_order)

        head_avgs = []
        for m in method_order:
            vals = [per_class_metric[m][c] for c in working_head]
            head_avgs.append(round(np.mean(vals), 4))
        writer.writerow(["Head_Avg"] + head_avgs)

        tail_avgs = []
        for m in method_order:
            vals = [per_class_metric[m][c] for c in working_tail]
            tail_avgs.append(round(np.mean(vals), 4))
        writer.writerow(["Tail_Avg"] + tail_avgs)

        for cls in ordered_class_ids:
            row = [f"Class_{cls}"]
            for m in method_order:
                row.append(round(per_class_metric[m][cls], 4))
            writer.writerow(row)

    print(f"Accuracy table saved to: {table_path}")

    # ---- recompute overall for working set ------------------------------------
    def _compute_overall(acc_dict):
        total_support = 0
        correct = 0.0
        for cls in working_ids:
            support = int(all_reports["baseline"][cls]["support"])
            total_support += support
            correct += acc_dict[cls] * support
        return correct / total_support if total_support > 0 else 0.0

    overall = {m: _compute_overall(per_class_metric[m]) for m in method_order}

    # ---- terminal table -------------------------------------------------------
    print()
    print(f"Metric: {metric_label}")
    print(f"Head class IDs: {sorted(working_head, key=int)}")
    print(f"Tail class IDs: {sorted(working_tail, key=int)}")
    print()
    bh, bt = np.mean(baseline_head_vals), np.mean(baseline_tail_vals)
    oh, ot = np.mean(ours_head_vals), np.mean(ours_tail_vals)
    print("=" * 65)
    print(f"SUMMARY: Head vs Tail Class {metric_label}")
    print("=" * 65)
    print(f"{'Method':<15} {'Head Avg':>10} {'Tail Avg':>10} {'Head-Tail Gap':>15} {'Overall':>10}")
    print("-" * 65)
    print(f"{'Baseline':<15} {bh:>10.4f} {bt:>10.4f} {bh-bt:>15.4f} {overall['baseline']:>10.4f}")
    print(f"{'Ours':<15} {oh:>10.4f} {ot:>10.4f} {oh-ot:>15.4f} {overall['ours']:>10.4f}")
    print("-" * 65)
    print(f"{'Improvement':<15} {oh-bh:>+10.4f} {ot-bt:>+10.4f}")
    print(f"\nHead-Tail Gap (Baseline): {bh-bt:.4f}  |  Head-Tail Gap (Ours): {oh-ot:.4f}")
    print(f"Gap reduction: {(bh-bt) - (oh-ot):.4f}")

    print("\n" + "=" * 80)
    print(f"{'Category':>12}", end="")
    for m in method_order:
        print(f"  {m:>12}", end="")
    print()
    print("-" * 80)
    print(f"{'Head_Avg':>12}", end="")
    for v in head_avgs:
        print(f"  {v:>12.4f}", end="")
    print()
    print(f"{'Tail_Avg':>12}", end="")
    for v in tail_avgs:
        print(f"  {v:>12.4f}", end="")
    print()
    for cls in ordered_class_ids:
        print(f"{'Class_'+cls:>12}", end="")
        for m in method_order:
            print(f"  {per_class_metric[m][cls]:>12.4f}", end="")
        print()
    print("=" * 80)


def run_metric(all_metrics, all_reports, methods, class_counts, metric_key, metric_label):
    print(f"\n{'#' * 60}")
    print(f"#  METRIC: {metric_label}")
    print(f"{'#' * 60}")

    per_class_metric = {}
    for m in methods:
        if metric_key == 'f1':
            per_class_metric[m] = {}
            for cls in ALL_CLASS_IDS:
                per_class_metric[m][cls] = all_reports[m][cls]['f1-score']
        else:
            per_class_metric[m] = all_metrics[m]['per_class_accuracy']

    per_class_metric['ours'] = ours_data[f'per_class_{metric_key}']

    if metric_key == 'f1':
        per_class_metric['ours_precision'] = ours_data.get('per_class_precision', {})
        per_class_metric['ours_recall'] = ours_data.get('per_class_recall', {})
        per_class_metric['ours_f1'] = ours_data['per_class_f1']

    # Pass 1: full (unsuffixed — only for recall, avoid overwriting)
    if metric_key == 'accuracy':
        print("\n  --- PASS 1: Full (23 classes) ---")
        generate_all_outputs(all_metrics, all_reports, per_class_metric, methods, class_counts,
                             metric_label=metric_label)
    else:
        print("\n  --- PASS 1: Full (23 classes, skipped for non-recall) ---")

    # Pass 2: filtered
    print(f"\n  --- PASS 2: Filtered (excluding classes {sorted(EXCLUDED_IDS)}) ---")
    generate_all_outputs(all_metrics, all_reports, per_class_metric, methods, class_counts,
                         suffix=f"_{metric_key}_filtered", excluded_ids=EXCLUDED_IDS,
                         metric_label=metric_label)

    # Pass 3: with metric suffix
    print(f"\n  --- PASS 3: Full with metric suffix ---")
    generate_all_outputs(all_metrics, all_reports, per_class_metric, methods, class_counts,
                         suffix=f"_{metric_key}", metric_label=metric_label)


if __name__ == "__main__":
    all_metrics, all_reports, per_class_acc, methods = load_data()
    ours_data = load_ours_data()
    class_counts = count_training_samples(DATA_DIR)

    print(f"\nLoaded ours metrics from: {OURS_METRICS_PATH}")
    print(f"Ours overall accuracy: {ours_data['overall_accuracy']:.4f}")
    print(f"Ours macro f1: {ours_data['macro_f1']:.4f}")

    # Recall-based figures
    run_metric(all_metrics, all_reports, methods, class_counts, 'accuracy', 'Recall')

    # F1-based figures
    run_metric(all_metrics, all_reports, methods, class_counts, 'f1', 'F1')
