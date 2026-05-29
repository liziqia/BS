"""
Generate appendix CSV table: per-class Recall / Precision / F1 for all comparison methods.
Output: 23 classes × 3 metrics × 6 methods
"""
import json
import csv
import os

BASELINE_METRICS_DIR = '/root/autodl-tmp/Fusion1/OmniGen/Classify/results/metrics'
OURS_METRICS_PATH = '/root/autodl-tmp/Fusion1/OmniGen/Classify/ablation_synthetic_dataset1711/exp_02_weighted_random/metrics/metrics.json'
OUTPUT_DIR = '/root/autodl-tmp/Fusion1/OmniGen/Classify/results/tables'

METHOD_ORDER = [
    ('ours', '本文方法', OURS_METRICS_PATH),
    ('baseline', '参考基准', None),
    ('decoupled', '解耦训练', None),
    ('ot_dynamic', '最优传输', None),
    ('oversample', '过采样', None),
    ('weighted_loss', '重加权', None),
]


def load_ours_data():
    with open(OURS_METRICS_PATH, 'r') as f:
        data = json.load(f)
    return {
        'recall': data.get('per_class_accuracy', data.get('per_class_recall', {})),
        'precision': data.get('per_class_precision', {}),
        'f1': data.get('per_class_f1', {}),
    }


def load_baseline_data(method_key):
    report_path = os.path.join(BASELINE_METRICS_DIR, method_key, 'classification_report.json')
    if not os.path.exists(report_path):
        return None
    with open(report_path, 'r') as f:
        report = json.load(f)
    recall = {}
    precision = {}
    f1 = {}
    for cls, info in report.items():
        if cls in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        recall[cls] = info.get('recall', 0.0)
        precision[cls] = info.get('precision', 0.0)
        f1[cls] = info.get('f1-score', 0.0)
    return {'recall': recall, 'precision': precision, 'f1': f1}


def main():
    all_method_data = {}
    base_headers = []

    for method_key, display_name, path in METHOD_ORDER:
        if method_key == 'ours':
            data = load_ours_data()
        else:
            data = load_baseline_data(method_key)
        if data is None:
            print(f'WARNING: No data found for {method_key}, skipping')
            continue
        all_method_data[display_name] = data
        base_headers.append(display_name)

    class_ids = sorted(set(
        c for d in all_method_data.values() for c in d['recall'].keys()
    ), key=int)

    metric_cn = {'recall': 'Recall', 'precision': 'Precision', 'f1': 'F1'}
    metric_keys = ['recall', 'precision', 'f1']

    rows = []
    for cls in class_ids:
        for mk in metric_keys:
            row = [f'类别{cls}', metric_cn[mk]]
            for display_name in base_headers:
                val = all_method_data[display_name].get(mk, {}).get(cls, 0.0)
                row.append(f'{val:.4f}')
            rows.append(row)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, 'appendix_per_class_table.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['舰船类别', '指标'] + base_headers)
        for row in rows:
            writer.writerow(row)

    print(f'Saved: {csv_path}')
    print(f'Total rows: {len(rows)} (23 classes × 3 metrics)')
    print(f'Methods: {base_headers}')


if __name__ == '__main__':
    main()
