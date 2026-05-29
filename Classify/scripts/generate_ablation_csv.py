"""
Generate ablation experiment CSV tables for GQW component analysis.
"""
import json
import csv
import os
import numpy as np


HEAD_CLASS_IDS_EVAL = {'2', '0', '17', '4', '6'}  # top-5 by sample count
EXCLUDED_IDS = {'16', '21', '22'}                   # trivial classes (always ~1.0)

METRICS_FILES = {
    '完整 GQW (IV+OR+SC)': '/root/autodl-tmp/Fusion1/OmniGen/Classify/ablation_synthetic_dataset1711/exp_02_weighted_random/metrics/metrics.json',
    'IV + OR (去掉 SC)': '/root/autodl-tmp/Fusion1/OmniGen/Classify/ablation_gqw/exp_01_iv_or/metrics/metrics.json',
    'IV + SC (去掉 OR)': '/root/autodl-tmp/Fusion1/OmniGen/Classify/ablation_gqw/exp_02_iv_sc/metrics/metrics.json',
    'OR + SC (去掉 IV)': '/root/autodl-tmp/Fusion1/OmniGen/Classify/ablation_gqw/exp_03_or_sc/metrics/metrics.json',
    '均匀分配 (无 GQW)': '/root/autodl-tmp/Fusion1/OmniGen/Classify/ablation_synthetic_dataset1711/exp_04_uniform_random/metrics/metrics.json',
}

OUTPUT_DIR = '/root/autodl-tmp/Fusion1/OmniGen/Classify/ablation_gqw'


def compute_head_tail_avg(per_class_metric, head_ids, tail_ids):
    head_vals = [per_class_metric[c] for c in head_ids if c in per_class_metric]
    tail_vals = [per_class_metric[c] for c in tail_ids if c in per_class_metric]
    return np.mean(head_vals) if head_vals else 0.0, np.mean(tail_vals) if tail_vals else 0.0


def main():
    all_class_ids = {str(i) for i in range(23)}
    tail_class_ids_eval = all_class_ids - HEAD_CLASS_IDS_EVAL

    all_data = {}
    for label, path in METRICS_FILES.items():
        with open(path, 'r') as f:
            data = json.load(f)
        all_data[label] = data

    method_order = [
        '完整 GQW (IV+OR+SC)',
        'IV + OR (去掉 SC)',
        'IV + SC (去掉 OR)',
        'OR + SC (去掉 IV)',
        '均匀分配 (无 GQW)',
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # Table 1: Overall metrics comparison
    # ==========================================
    overall_metrics = [
        ('overall_accuracy', 'Overall Accuracy'),
        ('balanced_accuracy', 'Balanced Accuracy'),
        ('macro_f1', 'Macro F1'),
        ('weighted_f1', 'Weighted F1'),
        ('g_mean', 'G-Mean'),
        ('head_avg_accuracy', 'Head Avg Recall'),
        ('tail_avg_accuracy', 'Tail Avg Recall'),
    ]

    path1 = os.path.join(OUTPUT_DIR, 'ablation_overall.csv')
    with open(path1, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric'] + method_order)
        for key, label in overall_metrics:
            row = [label]
            for method in method_order:
                val = all_data[method].get(key, 0.0)
                row.append(f'{val:.4f}')
            writer.writerow(row)
    print(f'Saved: {path1}')

    # ==========================================
    # Table 2: Per-class F1 scores (full 23 classes)
    # ==========================================
    ordered_ids = sorted(all_class_ids, key=int)
    path2 = os.path.join(OUTPUT_DIR, 'ablation_per_class_f1.csv')
    with open(path2, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Class ID'] + method_order)
        for cls in ordered_ids:
            row = [f'Class {cls}']
            for method in method_order:
                f1_dict = all_data[method].get('per_class_f1', {})
                row.append(f'{f1_dict.get(cls, 0.0):.4f}')
            writer.writerow(row)
    print(f'Saved: {path2}')

    # ==========================================
    # Table 3: Head/Tail F1 averages
    # ==========================================
    path3 = os.path.join(OUTPUT_DIR, 'ablation_head_tail_f1.csv')
    with open(path3, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Group'] + method_order)

        for group_name, group_ids in [
            ('Head Avg F1 (top-5)', HEAD_CLASS_IDS_EVAL),
            ('Tail Avg F1', tail_class_ids_eval),
        ]:
            row = [group_name]
            for method in method_order:
                f1_dict = all_data[method].get('per_class_f1', {})
                vals = [f1_dict[c] for c in group_ids if c in f1_dict]
                avg = np.mean(vals) if vals else 0.0
                row.append(f'{avg:.4f}')
            writer.writerow(row)
    print(f'Saved: {path3}')

    # ==========================================
    # Table 4: Head/Tail recall averages
    # ==========================================
    path4 = os.path.join(OUTPUT_DIR, 'ablation_head_tail_recall.csv')
    with open(path4, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Group'] + method_order)

        for group_name, group_ids in [
            ('Head Avg Recall (top-5)', HEAD_CLASS_IDS_EVAL),
            ('Tail Avg Recall', tail_class_ids_eval),
        ]:
            row = [group_name]
            for method in method_order:
                acc_dict = all_data[method].get('per_class_accuracy', {})
                vals = [acc_dict[c] for c in group_ids if c in acc_dict]
                avg = np.mean(vals) if vals else 0.0
                row.append(f'{avg:.4f}')
            writer.writerow(row)
    print(f'Saved: {path4}')

    # ==========================================
    # Table 5: Head/Tail comparison with head-tail gap
    # ==========================================
    path5 = os.path.join(OUTPUT_DIR, 'ablation_head_tail_gap.csv')
    with open(path5, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Head Avg Recall', 'Tail Avg Recall', 'Tail* Avg Recall', 'Head-Tail R Gap', 'Head Avg F1', 'Tail Avg F1', 'Tail* Avg F1', 'Overall Acc', 'Macro F1'])

        for method in method_order:
            data = all_data[method]
            acc_dict = data.get('per_class_accuracy', {})
            f1_dict = data.get('per_class_f1', {})

            head_recall = np.mean([acc_dict[c] for c in HEAD_CLASS_IDS_EVAL if c in acc_dict])
            tail_recall = np.mean([acc_dict[c] for c in tail_class_ids_eval if c in acc_dict])
            tail_star_recall = np.mean([acc_dict[c] for c in tail_class_ids_eval if c in acc_dict and c not in EXCLUDED_IDS])
            head_f1 = np.mean([f1_dict[c] for c in HEAD_CLASS_IDS_EVAL if c in f1_dict])
            tail_f1 = np.mean([f1_dict[c] for c in tail_class_ids_eval if c in f1_dict])
            tail_star_f1 = np.mean([f1_dict[c] for c in tail_class_ids_eval if c in f1_dict and c not in EXCLUDED_IDS])
            gap_recall = head_recall - tail_recall

            writer.writerow([
                method,
                f'{head_recall:.4f}',
                f'{tail_recall:.4f}',
                f'{tail_star_recall:.4f}',
                f'{gap_recall:.4f}',
                f'{head_f1:.4f}',
                f'{tail_f1:.4f}',
                f'{tail_star_f1:.4f}',
                f'{data["overall_accuracy"]:.4f}',
                f'{data["macro_f1"]:.4f}',
            ])
    print(f'Saved: {path5}')

    # ==========================================
    # Print summary to terminal
    # ==========================================
    print('\n' + '=' * 130)
    print('GQW 子指标消融实验结果汇总')
    print('=' * 130)
    header = f"{'Method':<28} {'Overall':>8} {'Macro F1':>8} {'Head R':>8} {'Tail R':>8} {'Tail*R':>8} {'R Gap':>8} {'Head F1':>8} {'Tail F1':>8} {'Tail*F1':>8}"
    print(header)
    print('-' * 130)

    for method in method_order:
        data = all_data[method]
        acc_dict = data.get('per_class_accuracy', {})
        f1_dict = data.get('per_class_f1', {})

        head_r = np.mean([acc_dict[c] for c in HEAD_CLASS_IDS_EVAL if c in acc_dict])
        tail_r = np.mean([acc_dict[c] for c in tail_class_ids_eval if c in acc_dict])
        tail_star_r = np.mean([acc_dict[c] for c in tail_class_ids_eval if c in acc_dict and c not in EXCLUDED_IDS])
        head_f1 = np.mean([f1_dict[c] for c in HEAD_CLASS_IDS_EVAL if c in f1_dict])
        tail_f1 = np.mean([f1_dict[c] for c in tail_class_ids_eval if c in f1_dict])
        tail_star_f1 = np.mean([f1_dict[c] for c in tail_class_ids_eval if c in f1_dict and c not in EXCLUDED_IDS])
        gap_r = head_r - tail_r

        short_name = method[:26]
        print(f'{short_name:<28} {data["overall_accuracy"]:>8.4f} {data["macro_f1"]:>8.4f} {head_r:>8.4f} {tail_r:>8.4f} {tail_star_r:>8.4f} {gap_r:>8.4f} {head_f1:>8.4f} {tail_f1:>8.4f} {tail_star_f1:>8.4f}')

    print('=' * 130)

    # ==========================================
    # Table 6: Comprehensive summary (methods × metrics)
    # ==========================================
    path6 = os.path.join(OUTPUT_DIR, 'ablation_summary.csv')
    with open(path6, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        metric_names = [
            'Overall Accuracy', 'Balanced Accuracy', 'Macro F1', 'Weighted F1', 'G-Mean',
            'Head Avg Recall', 'Tail Avg Recall', 'Tail* Avg Recall', 'Head-Tail Recall Gap',
            'Head Avg F1', 'Tail Avg F1', 'Tail* Avg F1', 'Head-Tail F1 Gap',
        ]

        tail_star_ids = tail_class_ids_eval - EXCLUDED_IDS

        metric_fns = [
            lambda d: d['overall_accuracy'],
            lambda d: d['balanced_accuracy'],
            lambda d: d['macro_f1'],
            lambda d: d['weighted_f1'],
            lambda d: d['g_mean'],
            lambda d: np.mean([d['per_class_accuracy'][c] for c in HEAD_CLASS_IDS_EVAL if c in d['per_class_accuracy']]),
            lambda d: np.mean([d['per_class_accuracy'][c] for c in tail_class_ids_eval if c in d['per_class_accuracy']]),
            lambda d: np.mean([d['per_class_accuracy'][c] for c in tail_star_ids if c in d['per_class_accuracy']]),
            lambda d: (
                np.mean([d['per_class_accuracy'][c] for c in HEAD_CLASS_IDS_EVAL if c in d['per_class_accuracy']]) -
                np.mean([d['per_class_accuracy'][c] for c in tail_class_ids_eval if c in d['per_class_accuracy']])
            ),
            lambda d: np.mean([d['per_class_f1'][c] for c in HEAD_CLASS_IDS_EVAL if c in d['per_class_f1']]),
            lambda d: np.mean([d['per_class_f1'][c] for c in tail_class_ids_eval if c in d['per_class_f1']]),
            lambda d: np.mean([d['per_class_f1'][c] for c in tail_star_ids if c in d['per_class_f1']]),
            lambda d: (
                np.mean([d['per_class_f1'][c] for c in HEAD_CLASS_IDS_EVAL if c in d['per_class_f1']]) -
                np.mean([d['per_class_f1'][c] for c in tail_class_ids_eval if c in d['per_class_f1']])
            ),
        ]

        writer.writerow(['Method'] + metric_names)
        for method in method_order:
            row = [method]
            for fn in metric_fns:
                try:
                    val = fn(all_data[method])
                    row.append(f'{val:.4f}')
                except Exception:
                    row.append('N/A')
            writer.writerow(row)
    print(f'Saved: {path6}')

    print(f'\nAll CSVs saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
