import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
from scipy.stats import gmean
import numpy as np
import json
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


EXP_ID_CONFIG = {
    "baseline": "../checkpoints/baseline/model_final.pth",
    "oversample": "../checkpoints/oversample/model_final.pth",
    "weighted_loss": "../checkpoints/weighted_loss/model_final.pth",
    "decoupled": "../checkpoints/decoupled/model_final.pth",
    "ltsoups": "../checkpoints/ltsoups/model_final.pth",
    "ltrl": "../checkpoints/ltrl/model_final.pth",
    "ot_dynamic": "../checkpoints/ot_dynamic/model_final.pth",
    "omnigen": "../checkpoints/omnigen/model_final.pth",
    "real_data": None,
}


def load_model(checkpoint_path, num_classes=23, device='cuda'):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def evaluate_model(model, dataloader, dataset, class_names, device='cuda'):
    all_preds = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_indices.extend(range(batch_idx * dataloader.batch_size, 
                                     batch_idx * dataloader.batch_size + len(labels)))
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    overall_acc = accuracy_score(all_labels, all_preds)
    
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
            per_class_acc[class_name] = float(class_acc)
        else:
            per_class_acc[class_name] = 0.0
    
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    recalls = [per_class_acc[cls] for cls in class_names]
    g_mean = gmean(recalls)
    
    misclassified_details = []
    for idx in range(len(all_labels)):
        if all_preds[idx] != all_labels[idx]:
            sample_idx = all_indices[idx]
            image_path = dataset.samples[sample_idx][0]
            misclassified_details.append({
                'index': int(idx),
                'image_path': image_path,
                'true_class_id': int(all_labels[idx]),
                'true_class_name': class_names[all_labels[idx]],
                'predicted_class_id': int(all_preds[idx]),
                'predicted_class_name': class_names[all_preds[idx]],
            })
    
    return {
        'overall_accuracy': float(overall_acc),
        'per_class_accuracy': per_class_acc,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'misclassified_details': misclassified_details,
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'balanced_accuracy': float(balanced_acc),
        'g_mean': float(g_mean),
    }


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ 混淆矩阵已保存到 {save_path}")


def plot_per_class_accuracy(per_class_acc, save_path):
    classes = list(per_class_acc.keys())
    accs = list(per_class_acc.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accs, color='steelblue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ 各类别准确率图已保存到 {save_path}")


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")
    
    checkpoint_path = args.checkpoint if args.checkpoint else EXP_ID_CONFIG.get(args.exp_id)
    
    if checkpoint_path is None:
        print(f"❌ 实验 {args.exp_id} 没有可用的权重（D实验不保存权重）")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 权重文件不存在: {checkpoint_path}")
        print(f"   请先运行训练: python train_augmented.py --exp_id {args.exp_id}")
        return
    
    print(f"📂 加载模型: {checkpoint_path}")
    model, checkpoint = load_model(checkpoint_path, args.num_classes, device)
    print(f"✅ 模型加载成功 (实验: {checkpoint.get('experiment', 'unknown')})")
    
    test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(args.test_dir, test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=args.num_workers
    )
    
    class_names = test_dataset.classes
    print(f"✅ 测试集样本数: {len(test_dataset)}")
    print(f"✅ 类别数: {len(class_names)}")
    
    print("\n 开始评估...")
    results = evaluate_model(model, test_loader, test_dataset, class_names, device)
    
    metrics_dir = f'../results/metrics/{args.exp_id}'
    figures_dir = f'../results/figures/{args.exp_id}'
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    metrics_output = {
        'experiment': args.exp_id,
        'overall_accuracy': results['overall_accuracy'],
        'per_class_accuracy': results['per_class_accuracy'],
        'macro_f1': results['macro_f1'],
        'weighted_f1': results['weighted_f1'],
        'balanced_accuracy': results['balanced_accuracy'],
        'g_mean': results['g_mean'],
    }
    
    with open(os.path.join(metrics_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_output, f, indent=4)
    print(f"✅ 评估指标已保存到 {metrics_dir}/metrics.json")
    
    with open(os.path.join(metrics_dir, 'classification_report.json'), 'w') as f:
        json.dump(results['classification_report'], f, indent=4)
    print(f"✅ 分类报告已保存到 {metrics_dir}/classification_report.json")
    
    misclassified_output = {
        'experiment': args.exp_id,
        'total_misclassified': len(results['misclassified_details']),
        'total_samples': len(results['labels']),
        'misclassified_details': results['misclassified_details']
    }
    
    with open(os.path.join(metrics_dir, 'misclassified.json'), 'w') as f:
        json.dump(misclassified_output, f, indent=4)
    print(f"✅ 错分详情已保存到 {metrics_dir}/misclassified.json")
    
    cm = np.array(results['confusion_matrix'])
    plot_confusion_matrix(cm, class_names, os.path.join(figures_dir, 'confusion_matrix.png'))
    plot_per_class_accuracy(results['per_class_accuracy'], os.path.join(figures_dir, 'per_class_accuracy.png'))
    
    print(f"\n{'='*40}")
    print(f"实验: {args.exp_id}")
    print(f"整体准确率: {results['overall_accuracy']:.4f}")
    print(f"Macro F1:     {results['macro_f1']:.4f}")
    print(f"Weighted F1:  {results['weighted_f1']:.4f}")
    print(f"Balanced Acc: {results['balanced_accuracy']:.4f}")
    print(f"G-Mean:       {results['g_mean']:.4f}")
    print(f"{'='*40}")
    
    print("\n各类别准确率:")
    for class_name, acc in results['per_class_accuracy'].items():
        print(f"  {class_name}: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型评估脚本')
    parser.add_argument("--exp_id", type=str, default="ot_dynamic", help="实验 ID：baseline(A) / oversample(B1) / weighted_loss(B2) / decoupled(B3) / ltsoups(B4) / ltrl(B5) / ot_dynamic(B6) / omnigen(C) / real_data(D)")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型权重路径，默认根据exp_id自动推断")
    parser.add_argument("--test_dir", type=str, default="../data/test", help="测试集路径")
    parser.add_argument("--num_classes", type=int, default=23, help="分类类别数")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    args = parser.parse_args()
    main(args)
