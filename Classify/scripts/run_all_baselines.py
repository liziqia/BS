import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import numpy as np
import json
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, f1_score, balanced_accuracy_score
from scipy.stats import gmean
import argparse
import shutil
import random

# 导入工具函数
from utils import (
    oversample_dataset,
    compute_class_weights, create_weighted_loss,
    get_class_labels, DecoupledTrainer,
    create_lt_soups_subsets, merge_model_weights,
    create_ltrl_model, create_ot_dynamic_softmax,
    create_focal_loss, create_ldam_loss
)


# ==========================================
# 实验配置
# ==========================================

BASELINE_METHODS = {
    'baseline': {
        'label': 'Baseline (原始训练)',
        'description': '使用标准 CrossEntropyLoss，不处理类别不平衡',
    },
    'oversample': {
        'label': '随机过采样',
        'description': '对尾部类别复制样本，使每个类别不少于阈值',
        'min_samples': 150,
    },
    'weighted_loss': {
        'label': '类别平衡加权损失',
        'description': '使用 Effective Number 方法计算类别权重',
        'beta': 0.99,
    },
    'focal_loss': {
        'label': 'Focal Loss',
        'description': '降低简单样本权重，专注于困难样本',
        'gamma': 2.0,
    },
    'ldam_loss': {
        'label': 'LDAM Loss',
        'description': '标签分布感知的边际损失',
        'max_m': 0.5,
        's': 30.0,
    },
    'decoupled': {
        'label': '解耦训练',
        'description': '两阶段训练：先训练特征，再训练分类器',
        'stage1_epochs': 100,
        'stage2_epochs': 50,
        'stage2_method': 'balanced',
    },
    'ltsoups': {
        'label': 'LT-Soups',
        'description': '模型融合：训练多个不同不平衡比例的模型并融合',
        'num_models': 5,
    },
    'ltrl': {
        'label': 'LTRL 反思学习',
        'description': '知识回顾+总结+纠正',
        'lambda_ltrl': 0.1,
    },
    'ot_dynamic': {
        'label': '最优传输',
        'description': 'OT-Dynamic Softmax，动态调整类别权重',
        'epsilon': 0.1,
    },
    'omnigen': {
        'label': '本文方法 (OmniGen)',
        'description': '使用合成数据集增强（需先运行消融实验确定最优策略）',
        'synthetic_data_dir': '../ablation_synthetic_dataset/exp_01_weighted_fds/synthetic_data00',
    },
}

# 训练配置
TRAIN_CONFIG = {
    'num_epochs': 150,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_workers': 4,
    'seed': 42,
}


# ==========================================
# 工具函数
# ==========================================

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(num_classes, use_pretrained=True, weights_path=None):
    """创建ResNet50模型"""
    model = models.resnet50(weights=None)
    
    if use_pretrained and weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"    ✅ 加载预训练权重: {weights_path}")
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model


def train_single_epoch(model, dataloader, criterion, optimizer, device, epoch_num=None, total_epochs=None, batch_interval=10):
    """训练一个 epoch"""
    import time
    
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    total_batches = len(dataloader)
    epoch_start = time.time()
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # 每 batch_interval 个 batch 输出一次进度
        if batch_interval > 0 and ((batch_idx + 1) % batch_interval == 0 or batch_idx == total_batches - 1):
            batch_loss = running_loss / ((batch_idx + 1) * dataloader.batch_size)
            batch_acc = running_corrects.double() / ((batch_idx + 1) * dataloader.batch_size)
            elapsed = time.time() - epoch_start
            if epoch_num is not None and total_epochs is not None:
                print(f"    Epoch {epoch_num}/{total_epochs} [{batch_idx+1}/{total_batches}] "
                      f"Loss={batch_loss:.4f}, Acc={batch_acc:.4f}, "
                      f"Time={elapsed:.1f}s", end='\r')
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    epoch_time = time.time() - epoch_start
    
    return epoch_loss, epoch_acc.item(), epoch_time


def train_model_standard(model, dataloader, criterion, optimizer, num_epochs, device, checkpoint_dir, figures_dir):
    """标准训练流程"""
    import time
    
    history = {'train_loss': [], 'train_acc': []}
    
    total_batches = len(dataloader)
    print(f"    开始训练: {num_epochs} epochs, {total_batches} batches/epoch")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss, epoch_acc, epoch_time = train_single_epoch(
            model, dataloader, criterion, optimizer, device,
            epoch_num=epoch+1, total_epochs=num_epochs
        )
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # 每个 epoch 结束时输出
        elapsed_total = time.time() - start_time
        remaining_epochs = num_epochs - (epoch + 1)
        avg_epoch_time = elapsed_total / (epoch + 1)
        estimated_remaining = remaining_epochs * avg_epoch_time
        
        print(f"    Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, "
              f"Time={epoch_time:.1f}s, "
              f"ETA={estimated_remaining/60:.1f}min")
    
    # 保存模型和训练曲线
    save_model_and_history(model, optimizer, history, checkpoint_dir, figures_dir)
    
    return model, history


def save_model_and_history(model, optimizer, history, checkpoint_dir, figures_dir):
    """保存模型权重和训练历史"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, os.path.join(checkpoint_dir, 'model_final.pth'))
    
    # 保存训练曲线
    os.makedirs(figures_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'training_curve.png'), dpi=150)
    plt.close()


def evaluate_model(model, test_loader, class_names, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    overall_acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 各类别准确率
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[name] = float((all_preds[mask] == all_labels[mask]).sum() / mask.sum())
        else:
            per_class_acc[name] = 0.0
    
    # 尾部类别平均准确率
    tail_accs = [acc for name, acc in per_class_acc.items() if acc > 0]
    tail_avg_acc = np.mean(tail_accs) if len(tail_accs) > 0 else 0.0
    
    # G-Mean
    recalls = [per_class_acc[cls] for cls in class_names if per_class_acc[cls] > 0]
    g_mean_val = gmean(recalls) if len(recalls) > 0 else 0.0
    
    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # 计算召回率、精确率、F1
    from sklearn.metrics import precision_score, recall_score
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    precision_dict = {class_names[i]: float(per_class_precision[i]) for i in range(len(class_names))}
    recall_dict = {class_names[i]: float(per_class_recall[i]) for i in range(len(class_names))}
    f1_dict = {class_names[i]: float(per_class_f1[i]) for i in range(len(class_names))}
    
    return {
        'overall_accuracy': float(overall_acc),
        'balanced_accuracy': float(balanced_acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'tail_avg_accuracy': float(tail_avg_acc),
        'g_mean': float(g_mean_val),
        'per_class_accuracy': per_class_acc,
        'per_class_precision': precision_dict,
        'per_class_recall': recall_dict,
        'per_class_f1': f1_dict,
        'confusion_matrix': cm.tolist(),
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    import seaborn as sns
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_per_class_accuracy(per_class_acc, save_path):
    """绘制各类别准确率"""
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


def plot_head_tail_comparison(per_class_acc, class_counts, save_path, threshold=150):
    """绘制头部/尾部类别准确率对比图"""
    head_accs = []
    tail_accs = []
    head_labels = []
    tail_labels = []
    
    for cls, acc in per_class_acc.items():
        count = class_counts.get(cls, 0)
        if count >= threshold:
            head_accs.append(acc)
            head_labels.append(cls)
        else:
            tail_accs.append(acc)
            tail_labels.append(cls)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if head_accs:
        bars1 = ax1.bar(head_labels, head_accs, color='steelblue')
        ax1.set_xlabel('Head Classes')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Head Classes (≥{threshold} samples)')
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='x', rotation=45)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
    
    if tail_accs:
        bars2 = ax2.bar(tail_labels, tail_accs, color='coral')
        ax2.set_xlabel('Tail Classes')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'Tail Classes (<{threshold} samples)')
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(axis='x', rotation=45)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return {
        'head_avg': np.mean(head_accs) if head_accs else 0.0,
        'tail_avg': np.mean(tail_accs) if tail_accs else 0.0,
    }


def plot_prf_comparison(per_class_precision, per_class_recall, per_class_f1, class_names, save_path):
    """绘制Precision/Recall/F1对比图"""
    classes = list(class_names)
    precisions = [per_class_precision.get(cls, 0) for cls in classes]
    recalls = [per_class_recall.get(cls, 0) for cls in classes]
    f1s = [per_class_f1.get(cls, 0) for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recalls, width, label='Recall', color='coral')
    bars3 = ax.bar(x + width, f1s, width, label='F1', color='lightgreen')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Precision, Recall, F1')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_metrics_comparison(all_results, save_path):
    """绘制基线对比实验指标对比图"""
    exp_names = list(all_results.keys())
    
    metrics_to_plot = ['overall_accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'tail_avg_accuracy', 'g_mean']
    metric_labels = ['Overall Accuracy', 'Balanced Accuracy', 'Macro F1', 'Weighted F1', 'Tail Avg Accuracy', 'G-Mean']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_to_plot)))
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        values = [all_results[exp]['results'].get(metric, 0) if 'results' in all_results[exp] else all_results[exp].get(metric, 0) for exp in exp_names]
        
        bars = ax.bar(range(len(exp_names)), values, color=colors[idx])
        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels([name.replace('exp_', '').replace('_', ' ') for name in exp_names], 
                          rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.0)
        ax.set_title(label)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_tsne_features(model, test_loader, device, save_path):
    """绘制t-SNE特征可视化"""
    from sklearn.manifold import TSNE
    
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            features = model.conv1(inputs)
            features = model.bn1(features)
            features = model.relu(features)
            features = model.maxpool(features)
            features = model.layer1(features)
            features = model.layer2(features)
            features = model.layer3(features)
            features = model.layer4(features)
            features = model.avgpool(features)
            features = torch.flatten(features, 1)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)
    
    print("    计算t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(all_features)
    
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(all_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = all_labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=f'Class {label}', alpha=0.7, s=50)
    
    plt.title('t-SNE Feature Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_comparison_table(all_results, save_path):
    """生成基线对比实验对比表格"""
    exp_names = list(all_results.keys())
    
    metrics = ['overall_accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 
               'tail_avg_accuracy', 'g_mean']
    metric_labels = ['Overall Acc', 'Balanced Acc', 'Macro F1', 'Weighted F1', 
                     'Tail Acc', 'G-Mean']
    
    table_lines = []
    table_lines.append("=" * 120)
    table_lines.append("基线对比实验结果对比表")
    table_lines.append("=" * 120)
    
    header = f"{'方法':<30}" + "".join([f"{m:>12}" for m in metric_labels])
    table_lines.append(header)
    table_lines.append("-" * 120)
    
    for exp in exp_names:
        row = f"{exp:<30}"
        for metric in metrics:
            value = all_results[exp].get(metric, 0.0)
            row += f"{value:>12.4f}"
        table_lines.append(row)
    
    table_lines.append("=" * 120)
    
    with open(save_path.replace('.png', '.txt'), 'w') as f:
        f.write('\n'.join(table_lines))
    
    print('\n' + '\n'.join(table_lines) + '\n')


# ==========================================
# 各方法训练函数
# ==========================================

def train_baseline(model, train_loader, num_epochs, lr, device, checkpoint_dir, figures_dir):
    """A. Baseline - 原始训练"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
    
    return train_model_standard(model, train_loader, criterion, optimizer, num_epochs, device, checkpoint_dir, figures_dir)


def train_oversample(model, train_loader, num_epochs, lr, device, checkpoint_dir, figures_dir, min_samples=150):
    """B1. 过采样"""
    # 注意：过采样应该在数据准备阶段完成，这里直接使用过采样后的数据
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
    
    return train_model_standard(model, train_loader, criterion, optimizer, num_epochs, device, checkpoint_dir, figures_dir)


def train_weighted_loss(model, train_loader, num_epochs, lr, device, checkpoint_dir, figures_dir, beta=0.99):
    """B2. 类别平衡加权损失"""
    model = model.to(device)
    
    # 计算类别权重
    labels = get_class_labels(train_loader.dataset)
    class_counts = np.bincount(labels)
    class_weights = compute_class_weights(labels, beta=beta)
    criterion = create_weighted_loss(class_weights, device)
    
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
    
    return train_model_standard(model, train_loader, criterion, optimizer, num_epochs, device, checkpoint_dir, figures_dir)


def train_focal_loss(model, train_loader, num_epochs, lr, device, checkpoint_dir, figures_dir, gamma=2.0):
    """B5. Focal Loss"""
    model = model.to(device)
    
    # 计算类别权重
    labels = get_class_labels(train_loader.dataset)
    class_counts = np.bincount(labels)
    criterion = create_focal_loss(class_counts, device, gamma=gamma)
    
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
    
    return train_model_standard(model, train_loader, criterion, optimizer, num_epochs, device, checkpoint_dir, figures_dir)


def train_ldam_loss(model, train_loader, num_epochs, lr, device, checkpoint_dir, figures_dir, max_m=0.5, s=30.0):
    """B6. LDAM Loss"""
    model = model.to(device)
    
    # 计算类别样本数
    labels = get_class_labels(train_loader.dataset)
    class_counts = np.bincount(labels)
    criterion = create_ldam_loss(class_counts, device, max_m=max_m, s=s)
    
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
    
    return train_model_standard(model, train_loader, criterion, optimizer, num_epochs, device, checkpoint_dir, figures_dir)


def train_decoupled(model, train_loader, num_epochs, lr, device, checkpoint_dir, figures_dir, 
                   stage1_epochs=100, stage2_epochs=50, stage2_method='balanced'):
    """B3. 解耦训练"""
    model = model.to(device)
    trainer = DecoupledTrainer(model, device, len(train_loader.dataset.classes))
    
    # 第一阶段：训练特征表示
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    history_stage1 = trainer.train_representation(
        train_loader, optimizer, criterion, stage1_epochs, figures_dir
    )
    
    # 第二阶段：训练分类器
    history_stage2 = trainer.train_classifier(
        train_loader, stage2_epochs, stage2_method, figures_dir
    )
    
    # 合并历史
    history = {
        'stage1_loss': history_stage1['train_loss'],
        'stage1_acc': history_stage1['train_acc'],
        'stage2_loss': history_stage2['train_loss'],
        'stage2_acc': history_stage2['train_acc'],
    }
    
    save_model_and_history(model, optimizer, history, checkpoint_dir, figures_dir)
    
    return model, history


def train_ltsoups(model, train_loader, num_epochs, lr, device, checkpoint_dir, figures_dir, num_models=5):
    """B4. LT-Soups - 模型融合"""
    model = model.to(device)
    num_classes = len(train_loader.dataset.classes)
    
    # 阶段 1：训练多个模型并融合
    models_list = []
    subsets = create_lt_soups_subsets(train_loader.dataset, num_models, num_classes)
    
    for i, subset in enumerate(subsets):
        print(f"    训练第 {i+1}/{num_models} 个模型...")
        set_seed(TRAIN_CONFIG['seed'] + i)
        
        subset_loader = torch.utils.data.DataLoader(
            subset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True,
            num_workers=TRAIN_CONFIG['num_workers']
        )
        
        sub_model = create_model(num_classes, use_pretrained=True, 
                                weights_path=args.pretrained_weights)
        sub_model = sub_model.to(device)
        
        # 只训练分类器
        for param in sub_model.parameters():
            param.requires_grad = False
        for param in sub_model.fc.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.SGD(sub_model.fc.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        sub_model, _ = train_model_standard(
            sub_model, subset_loader, criterion, optimizer,
            num_epochs=num_epochs // num_models, device=device,
            checkpoint_dir=None, figures_dir=None
        )
        
        models_list.append(sub_model)
    
    # 融合权重
    print(f"    融合 {num_models} 个模型的权重...")
    merged_weights = merge_model_weights(models_list)
    model.load_state_dict(merged_weights)
    
    # 阶段 2：在完整数据集上微调分类器
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    model, history = train_model_standard(
        model, train_loader, criterion, optimizer,
        num_epochs=num_epochs // 3, device=device,
        checkpoint_dir=checkpoint_dir, figures_dir=figures_dir
    )
    
    return model, history


def train_ltrl(model, train_loader, num_epochs, lr, device, checkpoint_dir, figures_dir, lambda_ltrl=0.1):
    """B7. LTRL - 反思学习"""
    import time
    
    num_classes = len(train_loader.dataset.classes)
    
    # 获取类别样本数
    labels = get_class_labels(train_loader.dataset)
    class_counts = np.bincount(labels)
    
    ltrl_model, criterion, optimizer = create_ltrl_model(
        model, num_classes, device, class_counts, lambda_ltrl
    )
    
    history = {'train_loss': [], 'train_acc': []}
    
    total_batches = len(train_loader)
    print(f"    开始训练: {num_epochs} epochs, {total_batches} batches/epoch")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        ltrl_model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for batch_idx, (inputs, labels_batch) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            
            # 提取特征
            features = ltrl_model.extract_features(inputs)
            outputs = ltrl_model(inputs)
            
            # 计算 LTRL 损失
            loss = criterion(outputs, labels_batch, features)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels_batch.data)
            
            # 每 10 个 batch 输出一次进度
            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                batch_loss = running_loss / ((batch_idx + 1) * train_loader.batch_size)
                batch_acc = running_corrects.double() / ((batch_idx + 1) * train_loader.batch_size)
                elapsed = time.time() - epoch_start
                print(f"    Epoch {epoch+1}/{num_epochs} [{batch_idx+1}/{total_batches}] "
                      f"Loss={batch_loss:.4f}, Acc={batch_acc:.4f}, "
                      f"Time={elapsed:.1f}s", end='\r')
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        # 每个 epoch 结束时输出
        elapsed_total = time.time() - start_time
        remaining_epochs = num_epochs - (epoch + 1)
        avg_epoch_time = elapsed_total / (epoch + 1)
        estimated_remaining = remaining_epochs * avg_epoch_time
        
        print(f"    Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, "
              f"Time={epoch_time:.1f}s, "
              f"ETA={estimated_remaining/60:.1f}min")
    
    # 保存模型
    model = ltrl_model.model
    save_model_and_history(model, optimizer, history, checkpoint_dir, figures_dir)
    
    return model, history


def train_ot_dynamic(model, train_loader, num_epochs, lr, device, checkpoint_dir, figures_dir, epsilon=0.1):
    """B8. 最优传输"""
    model = model.to(device)
    num_classes = len(train_loader.dataset.classes)
    
    # 获取类别样本数
    labels = get_class_labels(train_loader.dataset)
    class_counts = np.bincount(labels)
    
    criterion = create_ot_dynamic_softmax(num_classes, device, class_counts, epsilon=epsilon)
    
    # 只训练分类器
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
    
    return train_model_standard(model, train_loader, criterion, optimizer, num_epochs, device, checkpoint_dir, figures_dir)


def train_omnigen(model, train_loader, num_epochs, lr, device, checkpoint_dir, figures_dir, synthetic_data_dir=None):
    """本文方法：使用合成数据集增强"""
    if synthetic_data_dir is None or not os.path.exists(synthetic_data_dir):
        raise ValueError("必须指定 synthetic_data_dir 参数，指向消融实验确定的最优合成数据目录")
    
    # 合并原始数据和合成数据
    import tempfile
    combined_dir = os.path.join(checkpoint_dir, 'combined_data')
    
    if os.path.exists(combined_dir):
        shutil.rmtree(combined_dir)
    shutil.copytree(train_loader.dataset.root, combined_dir)
    
    # 统计每个类别新增的合成数据数量
    synthetic_counts = {}
    
    # 合并合成数据
    for class_id in os.listdir(synthetic_data_dir):
        src_class_dir = os.path.join(synthetic_data_dir, class_id)
        dst_class_dir = os.path.join(combined_dir, class_id)
        
        if not os.path.exists(dst_class_dir):
            os.makedirs(dst_class_dir, exist_ok=True)
        
        count = 0
        for img_file in os.listdir(src_class_dir):
            src_img = os.path.join(src_class_dir, img_file)
            dst_img = os.path.join(dst_class_dir, f"syn_{img_file}")
            shutil.copy2(src_img, dst_img)
            count += 1
        
        synthetic_counts[class_id] = count
    
    # 保存新增数量统计
    import json
    stats_path = os.path.join(checkpoint_dir, 'synthetic_counts.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(synthetic_counts, f, indent=2, ensure_ascii=False)
    
    total_synthetic = sum(synthetic_counts.values())
    print(f"    合成数据新增: {total_synthetic} 张 (各类别: {synthetic_counts})")
    
    # 创建新的数据加载器
    train_dataset = datasets.ImageFolder(combined_dir, train_loader.dataset.transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_loader.batch_size, shuffle=True,
        num_workers=train_loader.num_workers
    )
    
    print(f"    合并后训练集样本数: {len(train_dataset)}")
    
    # 标准训练
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
    
    return train_model_standard(model, train_loader, criterion, optimizer, num_epochs, device, checkpoint_dir, figures_dir)


# 训练函数映射
TRAIN_FUNCTIONS = {
    'baseline': train_baseline,
    'oversample': train_oversample,
    'weighted_loss': train_weighted_loss,
    'focal_loss': train_focal_loss,
    'ldam_loss': train_ldam_loss,
    'decoupled': train_decoupled,
    'ltsoups': train_ltsoups,
    'ltrl': train_ltrl,
    'ot_dynamic': train_ot_dynamic,
    'omnigen': train_omnigen,
}


# ==========================================
# 主实验流程
# ==========================================

def run_baseline_comparison(args):
    """运行所有基线方法对比实验"""
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"开始基线方法对比实验")
    print(f"设备: {device}")
    print(f"{'='*80}\n")
    
    # 创建实验根目录
    exp_root = os.path.join(args.base_dir, 'baseline_comparison')
    os.makedirs(exp_root, exist_ok=True)
    
    # 准备测试集
    test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(args.test_dir, test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, 
        num_workers=TRAIN_CONFIG['num_workers']
    )
    class_names = test_dataset.classes
    num_classes = len(class_names)
    
    # 统计类别数量（用于head/tail对比图）
    class_counts = {}
    for class_name in os.listdir(args.original_data_dir):
        class_dir = os.path.join(args.original_data_dir, class_name)
        if os.path.isdir(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            class_counts[class_name] = count
    
    # 训练数据变换
    train_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 运行所有基线方法
    all_results = {}
    total_methods = len(BASELINE_METHODS)
    current_method = 0
    
    # 确定要运行的方法
    methods_to_run = args.methods.split(',') if args.methods else list(BASELINE_METHODS.keys())
    
    for method_name in methods_to_run:
        method_name = method_name.strip()
        if method_name not in BASELINE_METHODS:
            print(f"⚠️  未知方法: {method_name}，跳过")
            continue
        
        current_method += 1
        method_config = BASELINE_METHODS[method_name]
        
        exp_name = f"exp_{current_method:02d}_{method_name}"
        exp_dir = os.path.join(exp_root, exp_name)
        
        # 检查是否已存在
        model_path = os.path.join(exp_dir, 'checkpoint', 'model_final.pth')
        if args.skip_existing and os.path.exists(model_path):
            print(f"\n{'='*80}")
            print(f"实验 {current_method}/{len(methods_to_run)}: {method_config['label']}")
            print(f"  ✅ 已存在，跳过")
            print(f"{'='*80}")
            continue
        
        print(f"\n{'='*80}")
        print(f"实验 {current_method}/{len(methods_to_run)}: {method_config['label']}")
        print(f"  描述: {method_config['description']}")
        print(f"{'='*80}\n")
        
        try:
            # 步骤1: 准备数据
            print("  [1/4] 准备数据...")
            
            if method_name == 'oversample':
                # 过采样需要创建新的数据集
                oversampled_dir = os.path.join(exp_dir, 'oversampled_data')
                
                # 如果指定了synthetic_data_dir，让Oversample新增样本数与OmniGen一致
                synthetic_dir = args.synthetic_data_dir or BASELINE_METHODS.get('omnigen', {}).get('synthetic_data_dir')
                if synthetic_dir and os.path.exists(synthetic_dir):
                    # 统计OmniGen新增的合成数据总数
                    synthetic_count = 0
                    for class_id in os.listdir(synthetic_dir):
                        class_dir = os.path.join(synthetic_dir, class_id)
                        if os.path.isdir(class_dir):
                            synthetic_count += len([f for f in os.listdir(class_dir) 
                                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    
                    # 显式定义头部类别（与消融实验一致）
                    HEAD_CLASS_IDS = {'2', '0', '17', '4', '6', '10', '13'}
                    
                    # 计算原始尾类样本数
                    original_counts = {}
                    for class_name in os.listdir(args.original_data_dir):
                        class_dir = os.path.join(args.original_data_dir, class_name)
                        if os.path.isdir(class_dir):
                            count = len([f for f in os.listdir(class_dir) 
                                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                            original_counts[class_name] = count
                    
                    tail_counts = [c for cid, c in original_counts.items() if cid not in HEAD_CLASS_IDS]
                    
                    # 查找合适的min_samples
                    def calc_oversample_amount(min_s):
                        return sum(max(0, min_s - c) for c in tail_counts)
                    
                    min_samples = 150  # 默认值
                    max_count = max(original_counts.values()) if original_counts else 150
                    for ms in range(int(min(tail_counts)), int(max_count) + 1):
                        if calc_oversample_amount(ms) >= synthetic_count:
                            min_samples = ms
                            break
                    
                    print(f"    OmniGen新增样本数: {synthetic_count}")
                    print(f"    Oversample min_samples设置为: {min_samples} (新增约{calc_oversample_amount(min_samples)}个样本)")
                else:
                    min_samples = method_config.get('min_samples', 150)
                
                data_dir = oversample_dataset(args.original_data_dir, oversampled_dir, min_samples)
            else:
                data_dir = args.original_data_dir
            
            # 创建数据加载器
            train_dataset = datasets.ImageFolder(data_dir, train_transform)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True,
                num_workers=TRAIN_CONFIG['num_workers']
            )
            
            print(f"    训练集样本数: {len(train_dataset)}")
            print(f"    类别数: {num_classes}")
            
            # 步骤2: 创建模型
            print("  [2/4] 创建模型...")
            set_seed(TRAIN_CONFIG['seed'])
            
            model = create_model(num_classes, use_pretrained=True, 
                                weights_path=args.pretrained_weights)
            
            # 步骤3: 训练
            print("  [3/4] 训练模型...")
            checkpoint_dir = os.path.join(exp_dir, 'checkpoint')
            figures_dir = os.path.join(exp_dir, 'figures')
            
            # 获取训练函数
            train_fn = TRAIN_FUNCTIONS[method_name]
            
            # 准备训练参数
            train_kwargs = {
                'model': model,
                'train_loader': train_loader,
                'num_epochs': TRAIN_CONFIG['num_epochs'],
                'lr': TRAIN_CONFIG['learning_rate'],
                'device': device,
                'checkpoint_dir': checkpoint_dir,
                'figures_dir': figures_dir,
            }
            
            # 添加方法特定参数
            if method_name == 'oversample':
                train_kwargs['min_samples'] = min_samples
            elif method_name == 'weighted_loss':
                train_kwargs['beta'] = method_config.get('beta', 0.99)
            elif method_name == 'focal_loss':
                train_kwargs['gamma'] = method_config.get('gamma', 2.0)
            elif method_name == 'ldam_loss':
                train_kwargs['max_m'] = method_config.get('max_m', 0.5)
                train_kwargs['s'] = method_config.get('s', 30.0)
            elif method_name == 'decoupled':
                train_kwargs['stage1_epochs'] = method_config.get('stage1_epochs', 100)
                train_kwargs['stage2_epochs'] = method_config.get('stage2_epochs', 50)
                train_kwargs['stage2_method'] = method_config.get('stage2_method', 'balanced')
            elif method_name == 'ltsoups':
                train_kwargs['num_models'] = method_config.get('num_models', 5)
            elif method_name == 'ltrl':
                train_kwargs['lambda_ltrl'] = method_config.get('lambda_ltrl', 0.1)
            elif method_name == 'ot_dynamic':
                train_kwargs['epsilon'] = method_config.get('epsilon', 0.1)
            elif method_name == 'omnigen':
                # 从命令行参数或配置获取合成数据目录
                synthetic_dir = args.synthetic_data_dir or method_config.get('synthetic_data_dir')
                if synthetic_dir is None:
                    print(f"    ⚠️  跳过本文方法：未指定 --synthetic_data_dir")
                    print(f"    请先运行消融实验，然后指定最优策略的合成数据路径")
                    continue
                train_kwargs['synthetic_data_dir'] = synthetic_dir
            
            model, history = train_fn(**train_kwargs)
            
            # 步骤4: 评估
            print("  [4/4] 评估模型...")
            results = evaluate_model(model, test_loader, class_names, device)
            
            # 保存结果
            metrics_dir = os.path.join(exp_dir, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            
            with open(os.path.join(metrics_dir, 'metrics.json'), 'w') as f:
                json.dump(results, f, indent=4)
            
            # 绘制图表
            plot_confusion_matrix(
                np.array(results['confusion_matrix']), class_names,
                os.path.join(figures_dir, 'confusion_matrix.png')
            )
            plot_per_class_accuracy(
                results['per_class_accuracy'],
                os.path.join(figures_dir, 'per_class_accuracy.png')
            )
            plot_head_tail_comparison(
                results['per_class_accuracy'], class_counts,
                os.path.join(figures_dir, 'head_tail_comparison.png'),
                threshold=150
            )
            
            # 绘制Precision/Recall/F1对比图
            plot_prf_comparison(
                results['per_class_precision'],
                results['per_class_recall'],
                results['per_class_f1'],
                class_names,
                os.path.join(figures_dir, 'precision_recall_f1.png')
            )
            
            plot_tsne_features(
                model, test_loader, device,
                os.path.join(figures_dir, 'tsne_features.png')
            )
            
            # 打印结果
            print(f"\n  ✅ 实验 {current_method} 完成!")
            print(f"  整体准确率: {results['overall_accuracy']:.4f}")
            print(f"  平衡准确率: {results['balanced_accuracy']:.4f}")
            print(f"  Macro F1:   {results['macro_f1']:.4f}")
            print(f"  尾部准确率: {results['tail_avg_accuracy']:.4f}")
            print(f"  G-Mean:     {results['g_mean']:.4f}")
            
            all_results[exp_name] = {
                'method': method_name,
                'label': method_config['label'],
                'results': results,
            }
            
            # 清理GPU内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n  ❌ 实验 {current_method} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results[exp_name] = {
                'method': method_name,
                'label': method_config['label'],
                'error': str(e)
            }
            continue
    
    # 保存汇总结果
    print(f"\n{'='*80}")
    print("所有实验完成！保存汇总结果...")
    print(f"{'='*80}\n")
    
    with open(os.path.join(exp_root, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # 生成对比图表
    print("生成对比图表...")
    plot_metrics_comparison(all_results, os.path.join(exp_root, 'metrics_comparison.png'))
    generate_comparison_table(all_results, os.path.join(exp_root, 'metrics_comparison.png'))
    
    # 生成对比表格
    print("\n实验结果汇总:")
    print(f"{'方法':<30} {'整体准确率':<12} {'平衡准确率':<12} {'Macro F1':<12} {'尾部准确率':<12} {'G-Mean':<12}")
    print("-" * 100)
    
    for exp_name, exp_data in all_results.items():
        if 'error' in exp_data:
            print(f"{exp_data['label']:<30} {'ERROR':<12}")
        else:
            r = exp_data['results']
            print(f"{exp_data['label']:<30} {r['overall_accuracy']:<12.4f} {r['balanced_accuracy']:<12.4f} {r['macro_f1']:<12.4f} {r['tail_avg_accuracy']:<12.4f} {r['g_mean']:<12.4f}")
    
    print(f"\n✅ 所有结果已保存到: {exp_root}")
    print(f"✅ 汇总结果: {os.path.join(exp_root, 'all_results.json')}")
    
    return all_results


# ==========================================
# 命令行参数
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基线方法对比实验脚本')
    
    parser.add_argument('--original_data_dir', type=str, 
                       default='../data/original',
                       help='原始训练集路径')
    parser.add_argument('--test_dir', type=str,
                       default='../data/test',
                       help='测试集路径')
    parser.add_argument('--pretrained_weights', type=str,
                       default='../pretrained_weights/resnet50-0676ba61.pth',
                       help='ResNet50预训练权重路径')
    parser.add_argument('--base_dir', type=str,
                       default='..',
                       help='基础目录')
    parser.add_argument('--skip_existing', action='store_true',
                       default=True,
                       help='跳过已存在的实验（默认开启，使用 --no-skip_existing 关闭）')
    parser.add_argument('--no-skip_existing', dest='skip_existing', action='store_false',
                       help='重新运行所有实验，不跳过已存在的')
    parser.add_argument('--methods', type=str,
                       default=None,
                       help='要运行的方法列表，逗号分隔（默认运行所有方法）')
    parser.add_argument('--synthetic_data_dir', type=str,
                       default=None,
                       help='本文方法使用的合成数据目录（消融实验确定的最优策略结果）')
    parser.add_argument('--tail_threshold', type=int,
                       default=150,
                       help='尾部类别阈值')
    
    args = parser.parse_args()
    
    # 运行实验
    results = run_baseline_comparison(args)
