"""
Generate comparison experiment CSV table for paper.
Compares Ours (GQW) against baseline methods.
"""
import json
import csv
import os
import numpy as np

HEAD_CLASS_IDS_EVAL = {'2', '0', '17', '4', '6'}
EXCLUDED_IDS = {'16', '21', '22'}

BASELINE_METRICS_DIR = '/root/autodl-tmp/Fusion1/OmniGen/Classify/results/metrics'
OURS_METRICS_PATH = '/root/autodl-tmp/Fusion1/OmniGen/Classify/ablation_synthetic_dataset1711/exp_02_weighted_random/metrics/metrics.json'
OURS_REPORT_PATH = '/root/autodl-tmp/Fusion1/OmniGen/Classify/ablation_synthetic_dataset1711/exp_02_weighted_random/metrics/classification_report.json'

OUTPUT_DIR = '/root/autodl-tmp/Fusion1/OmniGen/Classify/results/tables'

METHOD_DISPLAY_NAMES = {
    'baseline': 'Baseline',
    'weighted_loss': 'Weighted Loss',
    'decoupled': 'Decoupled Training',
    'oversample': 'Oversampling',
    'ot_dynamic': 'OT (Dynamic)',
    'omnigen': 'OmniGen Augmentation',
    'augmented_traditional': 'Augmented (Trad.)',
    'ours': 'Ours (GQW)',
}


def load_method_data(metrics_dir):
    methods = {}
    for method_dir in sorted(os.listdir(metrics_dir)):
        method_path = os.path.join(metrics_dir, method_dir)
        if not os.path.isdir(method_path):
            continue
        metrics_path = os.path.join(method_path, 'metrics.json')
        report_path = os.path.join(method_path, 'classification_report.json')
        if os.path.exists(metrics_path) and os.path.exists(report_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            with open(report_path, 'r') as f:
                report = json.load(f)
            methods[method_dir] = {
                'metrics': metrics,
                'report': report,
            }
    return methods


def compute_per_class_f1(report):
    per_class_f1 = {}
    for cls, info in report.items():
        if cls in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        per_class_f1[cls] = info.get('f1-score', 0.0)
    return per_class_f1


def main():
    all_class_ids = {str(i) for i in range(23)}
    tail_class_ids_eval = all_class_ids - HEAD_CLASS_IDS_EVAL
    tail_star_ids = tail_class_ids_eval - EXCLUDED_IDS

    methods = load_method_data(BASELINE_METRICS_DIR)

    with open(OURS_METRICS_PATH, 'r') as f:
        ours_metrics = json.load(f)
    with open(OURS_REPORT_PATH, 'r') as f:
        ours_report = json.load(f)
    methods['ours'] = {
        'metrics': ours_metrics,
        'report': ours_report,
    }

    method_order = [
        'baseline', 'weighted_loss', 'decoupled', 'oversample',
        'ot_dynamic', 'omnigen', 'augmented_traditional', 'ours',
    ]
    method_order = [m for m in method_order if m in methods]

    core_metric_names = [
        'Overall Accuracy',
        'Macro F1',
        'Weighted F1',
        'Head Avg F1',
        'Tail Avg F1',
        'Tail* Avg F1',
        'Head Avg Recall',
        'Tail Avg Recall',
        'Tail* Avg Recall',
    ]

    results = {}
    for method_key in method_order:
        data = methods[method_key]
        metrics = data['metrics']
        report = data['report']

        per_class_acc = metrics.get('per_class_accuracy', {})
        per_class_f1 = compute_per_class_f1(report)

        overall_acc = metrics.get('overall_accuracy', 0.0)

        macro_vals = [per_class_f1.get(c, 0.0) for c in all_class_ids]
        macro_f1 = np.mean(macro_vals) if macro_vals else 0.0

        total_support = 0.0
        weighted_sum = 0.0
        for cls in all_class_ids:
            if cls in report and isinstance(report[cls], dict):
                s = report[cls].get('support', 0)
                f1 = report[cls].get('f1-score', 0.0)
                weighted_sum += f1 * s
                total_support += s
        weighted_f1 = weighted_sum / total_support if total_support > 0 else 0.0

        head_acc = [per_class_acc[c] for c in HEAD_CLASS_IDS_EVAL if c in per_class_acc]
        tail_acc = [per_class_acc[c] for c in tail_class_ids_eval if c in per_class_acc]
        head_f1 = [per_class_f1[c] for c in HEAD_CLASS_IDS_EVAL if c in per_class_f1]
        tail_f1 = [per_class_f1[c] for c in tail_class_ids_eval if c in per_class_f1]

        tail_star_acc = [per_class_acc[c] for c in tail_star_ids if c in per_class_acc]
        tail_star_f1 = [per_class_f1[c] for c in tail_star_ids if c in per_class_f1]

        head_acc_avg = np.mean(head_acc) if head_acc else 0.0
        tail_acc_avg = np.mean(tail_acc) if tail_acc else 0.0
        head_f1_avg = np.mean(head_f1) if head_f1 else 0.0
        tail_f1_avg = np.mean(tail_f1) if tail_f1 else 0.0
        tail_star_acc_avg = np.mean(tail_star_acc) if tail_star_acc else 0.0
        tail_star_f1_avg = np.mean(tail_star_f1) if tail_star_f1 else 0.0

        results[method_key] = [
            overall_acc, macro_f1, weighted_f1,
            head_f1_avg, tail_f1_avg, tail_star_f1_avg,
            head_acc_avg, tail_acc_avg, tail_star_acc_avg,
        ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, 'comparison_table.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Method'] + core_metric_names)
        for method_key in method_order:
            display_name = METHOD_DISPLAY_NAMES.get(method_key, method_key)
            row = [display_name]
            for val in results[method_key]:
                row.append(f'{val:.4f}')
            writer.writerow(row)

    print(f'Comparison table saved to: {csv_path}')

    print('\n' + '=' * 130)
    print('Comparison Experiment Results')
    print('=' * 130)
    hdr = f"{'Method':<24}"
    for name in core_metric_names:
        hdr += f" {name:>14}"
    print(hdr)
    print('-' * 130)
    for method_key in method_order:
        display_name = METHOD_DISPLAY_NAMES.get(method_key, method_key)
        line = f'{display_name:<24}'
        for val in results[method_key]:
            line += f' {val:>14.4f}'
        print(line)
    print('=' * 130)


if __name__ == '__main__':
    main()