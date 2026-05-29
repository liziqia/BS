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

# 导入工具函数
from utils_syntheticDataset import (
    prepare_synthetic_dataset,
    identify_tail_classes,
    extract_features_resnet,
    selection_inverse_sigma_random,
    selection_random,
    compute_gqw_weights,
    allocate_generation_counts
)


# ==========================================
# 实验配置
# ==========================================

EXPERIMENT_CONFIG = {
    # 数量机制
    'quantity_methods': {
        'weighted': {'label': '加权数量生成', 'n_total': 1079},
        'uniform': {'label': '均匀分配(无机制)', 'n_total': 1079},
    },
    
    # 选择机制
    'selection_methods': {
        'inv_sigma_random': {'label': '反向σ过滤+随机选择'},
        # 'random': {'label': '随机选择(无筛选)'},
    }
}

# 训练配置
TRAIN_CONFIG = {
    'num_epochs': 250,
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
    import random
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


def train_single_experiment(
    model, train_loader, num_epochs, lr, device, 
    checkpoint_dir, figures_dir
):
    """训练单个模型"""
    import time
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
    
    history = {'train_loss': [], 'train_acc': []}
    
    total_batches = len(train_loader)
    print(f"    开始训练: {num_epochs} epochs, {total_batches} batches/epoch")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
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
    
    return model, history


def evaluate_model(model, test_loader, class_names, device, tail_class_ids=None, head_class_ids=None):
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
    
    overall_acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[name] = float((all_preds[mask] == all_labels[mask]).sum() / mask.sum())
        else:
            per_class_acc[name] = 0.0
    
    if tail_class_ids is not None:
        tail_accs = [per_class_acc[cid] for cid in tail_class_ids if cid in per_class_acc]
        tail_avg_acc = np.mean(tail_accs) if tail_accs else 0.0
    else:
        tail_avg_acc = 0.0
    
    if head_class_ids is not None:
        head_accs = [per_class_acc[cid] for cid in head_class_ids if cid in per_class_acc]
        head_avg_acc = np.mean(head_accs) if head_accs else 0.0
    else:
        head_avg_acc = 0.0
    
    recalls = [per_class_acc[cls] for cls in class_names if per_class_acc[cls] > 0]
    g_mean_val = gmean(recalls) if len(recalls) > 0 else 0.0
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    
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
        'head_avg_accuracy': float(head_avg_acc),
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


def plot_head_tail_comparison(per_class_acc, head_class_ids, tail_class_ids, save_path):
    """绘制头部/尾部类别准确率对比图"""
    head_accs = []
    tail_accs = []
    head_labels = []
    tail_labels = []
    
    for cls, acc in per_class_acc.items():
        if cls in head_class_ids:
            head_accs.append(acc)
            head_labels.append(cls)
        elif cls in tail_class_ids:
            tail_accs.append(acc)
            tail_labels.append(cls)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if head_accs:
        bars1 = ax1.bar(head_labels, head_accs, color='steelblue')
        ax1.set_xlabel('Head Classes')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Head Classes (top-{len(head_class_ids)} by sample count)')
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
        ax2.set_title(f'Tail Classes ({len(tail_class_ids)} classes)')
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


def plot_metrics_comparison(all_results, save_path):
    """绘制消融实验指标对比图"""
    exp_names = list(all_results.keys())
    
    metrics_to_plot = ['overall_accuracy', 'balanced_accuracy', 'head_avg_accuracy', 'tail_avg_accuracy']
    metric_labels = ['Overall Accuracy', 'Balanced Accuracy', 'Head Avg Accuracy', 'Tail Avg Accuracy']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_to_plot)))
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        values = [all_results[exp][metric] for exp in exp_names]
        
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
    
    # t-SNE降维
    print("    计算t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(all_features)
    
    # 绘制
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
    """生成消融实验对比表格"""
    exp_names = list(all_results.keys())
    
    metrics = ['overall_accuracy', 'balanced_accuracy', 'macro_f1', 
               'head_avg_accuracy', 'tail_avg_accuracy', 'g_mean']
    metric_labels = ['Overall Acc', 'Balanced Acc', 'Macro F1', 
                     'Head Acc', 'Tail Acc', 'G-Mean']
    
    # 生成文本表格
    table_lines = []
    table_lines.append("=" * 120)
    table_lines.append("消融实验结果对比表")
    table_lines.append("=" * 120)
    
    header = f"{'实验':<30}" + "".join([f"{m:>12}" for m in metric_labels])
    table_lines.append(header)
    table_lines.append("-" * 120)
    
    for exp in exp_names:
        row = f"{exp:<30}"
        for metric in metrics:
            value = all_results[exp].get(metric, 0.0)
            row += f"{value:>12.4f}"
        table_lines.append(row)
    
    table_lines.append("=" * 120)
    
    # 保存文本表格
    with open(save_path.replace('.png', '.txt'), 'w') as f:
        f.write('\n'.join(table_lines))
    
    # 打印到控制台
    print('\n' + '\n'.join(table_lines) + '\n')


def prepare_synthetic_dataset_resnet(
    original_data_dir,
    generated_data_dir,
    output_dir,
    tail_class_ids,
    quantity_method,
    n_total,
    n_per_class,
    target_class_count,
    resnet_model,
    resnet_transform,
    sklearn_classifier,
    selection_method,
    device,
):
    """
    使用ResNet特征的合成数据集准备（支持CUS和随机选择）
    """
    import json

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_data = {
        'quantity_method': quantity_method,
        'selection_method': selection_method,
        'n_total': n_total,
        'n_per_class': n_per_class,
        'target_class_count': target_class_count,
        'tail_class_ids': tail_class_ids,
        'original_data_counts': {},
        'allocated_counts': {},
        'selected_counts': {},
    }

    class_features = {}
    class_counts = {}

    for class_id in tail_class_ids:
        class_dir = os.path.join(original_data_dir, str(class_id))
        if not os.path.exists(class_dir):
            log_data['original_data_counts'][str(class_id)] = 0
            continue

        image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if len(image_paths) == 0:
            log_data['original_data_counts'][str(class_id)] = 0
            continue

        features = extract_features_resnet(image_paths, resnet_model, resnet_transform, device)
        class_features[class_id] = features
        class_counts[class_id] = len(image_paths)
        log_data['original_data_counts'][str(class_id)] = len(image_paths)
        print(f"    类别 {class_id}: {len(image_paths)} 张原始图像, 特征 {features.shape}")

    gen_counts = {}

    if quantity_method == 'weighted':
        gqw_weights = compute_gqw_weights(class_features, class_counts)
        gen_counts = allocate_generation_counts(len(tail_class_ids), n_total, gqw_weights)
        print(f"    GQW权重: {gqw_weights}")

    elif quantity_method == 'fixed_per_class':
        gen_counts = {cid: n_per_class for cid in tail_class_ids}

    elif quantity_method == 'fixed_total':
        if target_class_count is None:
            max_count = max(class_counts.values()) if class_counts else 0
            target_class_count = max_count
        gen_counts = {}
        for cid in tail_class_ids:
            needed = target_class_count - class_counts.get(cid, 0)
            gen_counts[cid] = max(0, needed)

    elif quantity_method == 'uniform':
        per_class = n_total // len(tail_class_ids)
        remainder = n_total % len(tail_class_ids)
        gen_counts = {}
        for i, cid in enumerate(tail_class_ids):
            gen_counts[cid] = per_class + (1 if i < remainder else 0)

    log_data['allocated_counts'] = {str(k): v for k, v in gen_counts.items()}
    print(f"    分配数量: {log_data['allocated_counts']}")

    selected_counts = {}
    filter_infos = {}

    # ========================================
    # 按分配数量选择，不足时从被过滤样本中随机补齐
    # ========================================
    shortfalls = {}

    for class_id in tail_class_ids:
        n_needed = gen_counts.get(class_id, 0)

        if n_needed <= 0:
            selected_counts[class_id] = 0
            continue

        gen_class_dir = os.path.join(generated_data_dir, str(class_id))
        if not os.path.exists(gen_class_dir):
            selected_counts[class_id] = 0
            shortfalls[class_id] = n_needed
            continue

        candidate_paths = [os.path.join(gen_class_dir, f) for f in os.listdir(gen_class_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        if len(candidate_paths) == 0:
            selected_counts[class_id] = 0
            shortfalls[class_id] = n_needed
            continue

        candidate_features = extract_features_resnet(
            candidate_paths, resnet_model, resnet_transform, device
        )

        if len(candidate_features) == 0:
            selected_counts[class_id] = 0
            shortfalls[class_id] = n_needed
            continue

        if selection_method == 'inv_sigma_random':
            original_features = class_features.get(class_id, candidate_features)
            selected_indices, filter_info = selection_inverse_sigma_random(
                candidate_features, original_features, n_needed
            )
            filter_infos[str(class_id)] = filter_info
            print(f"    类别 {class_id}: 反向σ={filter_info['sigma']}, 阈值={filter_info['threshold']:.4f}, "
                  f"离群池={filter_info['n_inverse']}/{filter_info['n_candidates']}, "
                  f"来自离群={filter_info['n_from_inverse']}, "
                  f"来自剩余={filter_info['n_from_remaining']}")

        elif selection_method == 'random':
            selected_indices = selection_random(candidate_features, n_needed)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")

        output_class_dir = os.path.join(output_dir, str(class_id))
        os.makedirs(output_class_dir, exist_ok=True)

        for idx in selected_indices:
            src_path = candidate_paths[idx]
            dst_path = os.path.join(output_class_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)

        selected_counts[class_id] = len(selected_indices)
        print(f"    类别 {class_id}: 最终选中 {len(selected_indices)}/{n_needed} 张")

    total_shortfall = sum(shortfalls.values())
    if total_shortfall > 0:
        print(f"\n  注意: 仍有缺额 {total_shortfall} 张 (候选图像不足), 缺额类别: {shortfalls}")

    print(f"  总计选择: {sum(selected_counts.values())}/{n_total} 张")

    log_data['selected_counts'] = {str(k): v for k, v in selected_counts.items()}
    log_data.setdefault('filter_info', {})
    for k, v in filter_infos.items():
        log_data['filter_info'][k] = v

    log_data['total_selected'] = int(sum(selected_counts.values()))
    log_data['shortfalls'] = {str(k): v for k, v in shortfalls.items()}

    log_path = os.path.join(output_dir, 'selection_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"  日志已保存: {log_path}")

    return selected_counts


# ==========================================
# 主实验流程
# ==========================================

def run_ablation_study(args):
    """运行8组消融实验"""
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"开始消融实验")
    print(f"设备: {device}")
    print(f"{'='*80}\n")
    
    # 创建实验根目录
    exp_root = os.path.join(args.base_dir, 'ablation_synthetic_dataset')
    os.makedirs(exp_root, exist_ok=True)
    
    # ResNet特征提取transform（统一2048维）
    resnet_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 识别尾部类别
    # 显式定义头部类别（其余均为尾部类别）
    HEAD_CLASS_IDS = {'2', '0', '17', '4', '6', '10', '13'}
    
    class_counts = {}
    for class_name in os.listdir(args.original_data_dir):
        class_dir = os.path.join(args.original_data_dir, class_name)
        if os.path.isdir(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            class_counts[class_name] = count
    
    tail_class_ids = [cid for cid in class_counts.keys() if cid not in HEAD_CLASS_IDS]
    head_class_ids = [cid for cid in class_counts.keys() if cid in HEAD_CLASS_IDS]
    
    print(f"尾部类别 ({len(tail_class_ids)}个): {tail_class_ids}")
    print(f"头部类别 ({len(head_class_ids)}个): {head_class_ids}\n")
    
    eval_head_class_ids = sorted(class_counts.keys(), key=lambda cid: class_counts[cid], reverse=True)[:5]
    eval_tail_class_ids = [cid for cid in class_counts.keys() if cid not in eval_head_class_ids]
    
    print(f"评估用头部类别 (top-5 by count): {eval_head_class_ids}")
    print(f"评估用尾部类别 ({len(eval_tail_class_ids)}个): {eval_tail_class_ids}\n")
    
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
    
    # 训练初始分类器（CUS方法需要）
    initial_classifier_path = os.path.join(exp_root, 'initial_classifier', 'model_final.pth')
    sklearn_classifier_path = os.path.join(exp_root, 'initial_classifier', 'sklearn_classifier.pkl')
    
    model_exists = os.path.exists(initial_classifier_path)
    sklearn_exists = os.path.exists(sklearn_classifier_path)
    
    if model_exists and sklearn_exists:
        print("✅ 检测到已训练的初始分类器，直接加载...\n")
        initial_model = create_model(len(class_names), use_pretrained=True, 
                                     weights_path=args.pretrained_weights)
        checkpoint = torch.load(initial_classifier_path, map_location=device)
        initial_model.load_state_dict(checkpoint['model_state_dict'])
        initial_model = initial_model.to(device)
        
        import pickle
        with open(sklearn_classifier_path, 'rb') as f:
            sklearn_classifier = pickle.load(f)
    else:
        # 加载或训练PyTorch模型
        if model_exists:
            print("✅ 检测到已训练的初始模型，直接加载...")
            initial_model = create_model(len(class_names), use_pretrained=True, 
                                         weights_path=args.pretrained_weights)
            checkpoint = torch.load(initial_classifier_path, map_location=device)
            initial_model.load_state_dict(checkpoint['model_state_dict'])
            initial_model = initial_model.to(device)
        else:
            print("训练初始分类器（用于CUS方法）...")
            set_seed(TRAIN_CONFIG['seed'])
            
            train_transform = transforms.Compose([
                transforms.Resize(512),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            original_dataset = datasets.ImageFolder(args.original_data_dir, train_transform)
            original_loader = torch.utils.data.DataLoader(
                original_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True,
                num_workers=TRAIN_CONFIG['num_workers']
            )
            
            initial_model = create_model(len(class_names), use_pretrained=True, 
                                         weights_path=args.pretrained_weights)
            initial_model, _ = train_single_experiment(
                initial_model, original_loader, 
                num_epochs=40, lr=TRAIN_CONFIG['learning_rate'],
                device=device,
                checkpoint_dir=os.path.join(exp_root, 'initial_classifier'),
                figures_dir=os.path.join(exp_root, 'initial_classifier')
            )
        
        # 提取特征并训练sklearn分类器（用于CUS）
        print("  提取特征并训练sklearn分类器...")
        from sklearn.linear_model import LogisticRegression
        import pickle
        
        # 如果原始loader不存在，需要重新创建
        if not model_exists:
            # original_loader already exists from training
            pass
        else:
            # 需要重新创建dataloader来提取特征
            eval_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            original_dataset = datasets.ImageFolder(args.original_data_dir, eval_transform)
            original_loader = torch.utils.data.DataLoader(
                original_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False,
                num_workers=0
            )
        
        all_features = []
        all_labels = []
        
        total_batches = len(original_loader)
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(original_loader):
                inputs = inputs.to(device)
                features = initial_model.conv1(inputs)
                features = initial_model.bn1(features)
                features = initial_model.relu(features)
                features = initial_model.maxpool(features)
                features = initial_model.layer1(features)
                features = initial_model.layer2(features)
                features = initial_model.layer3(features)
                features = initial_model.layer4(features)
                features = initial_model.avgpool(features)
                features = torch.flatten(features, 1)
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())
                if (batch_idx + 1) % 5 == 0:
                    print(f"    提取特征: [{batch_idx+1}/{total_batches}] {len(all_labels)} 样本", end='\r')
        print(f"    提取特征: [{total_batches}/{total_batches}] {len(all_labels)} 样本 完成")
        
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)
        
        sklearn_classifier = LogisticRegression(max_iter=1000)
        sklearn_classifier.fit(all_features, all_labels)
        
        # 保存sklearn分类器
        os.makedirs(os.path.join(exp_root, 'initial_classifier'), exist_ok=True)
        with open(sklearn_classifier_path, 'wb') as f:
            pickle.dump(sklearn_classifier, f)
        
        print("  ✅ 初始分类器处理完成并保存，sklearn分类器特征维度: 2048 (ResNet)\n")
    
    train_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    all_results = {}
    total_experiments = 4
    current_exp = 0
    
    for q_method, q_config in EXPERIMENT_CONFIG['quantity_methods'].items():
        for s_method, s_config in EXPERIMENT_CONFIG['selection_methods'].items():
            current_exp += 1
            
            exp_name = f"exp_{current_exp:02d}_{q_method}_{s_method}"
            exp_dir = os.path.join(exp_root, exp_name)
            
            print(f"\n{'='*80}")
            print(f"实验 {current_exp}/{total_experiments}: {exp_name}")
            print(f"  数量机制: {q_config['label']}")
            print(f"  选择机制: {s_config['label']}")
            print(f"{'='*80}\n")
            
            try:
                print(f"  [1/4] 准备合成数据集 (ResNet特征 + {s_config['label']})...")
                output_dir = os.path.join(exp_dir, 'synthetic_data')
                
                gen_counts = prepare_synthetic_dataset_resnet(
                    original_data_dir=args.original_data_dir,
                    generated_data_dir=args.generated_data_dir,
                    output_dir=output_dir,
                    tail_class_ids=tail_class_ids,
                    quantity_method=q_method,
                    n_total=q_config.get('n_total', 500),
                    n_per_class=q_config.get('n_per_class', 25),
                    target_class_count=q_config.get('target_class_count'),
                    resnet_model=initial_model,
                    resnet_transform=resnet_transform,
                    sklearn_classifier=sklearn_classifier,
                    selection_method=s_method,
                    device=device,
                )
                
                with open(os.path.join(exp_dir, 'gen_counts.json'), 'w') as f:
                    json.dump(gen_counts, f, indent=4)
                
                print("  [2/4] 创建训练数据集...")
                combined_dir = os.path.join(exp_dir, 'combined_data')
                
                if os.path.exists(combined_dir):
                    shutil.rmtree(combined_dir)
                shutil.copytree(args.original_data_dir, combined_dir)
                
                for class_id in os.listdir(output_dir):
                    src_class_dir = os.path.join(output_dir, class_id)
                    if not os.path.isdir(src_class_dir):
                        continue
                    dst_class_dir = os.path.join(combined_dir, class_id)
                    if not os.path.exists(dst_class_dir):
                        os.makedirs(dst_class_dir, exist_ok=True)
                    for img_file in os.listdir(src_class_dir):
                        src_img = os.path.join(src_class_dir, img_file)
                        dst_img = os.path.join(dst_class_dir, f"syn_{img_file}")
                        shutil.copy2(src_img, dst_img)
                
                print("  [3/4] 训练分类模型...")
                set_seed(TRAIN_CONFIG['seed'])
                
                combined_dataset = datasets.ImageFolder(combined_dir, train_transform)
                combined_loader = torch.utils.data.DataLoader(
                    combined_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True,
                    num_workers=TRAIN_CONFIG['num_workers']
                )
                
                model = create_model(len(class_names), use_pretrained=True,
                                    weights_path=args.pretrained_weights)
                
                checkpoint_dir = os.path.join(exp_dir, 'checkpoint')
                figures_dir = os.path.join(exp_dir, 'figures')
                
                model, history = train_single_experiment(
                    model, combined_loader,
                    num_epochs=TRAIN_CONFIG['num_epochs'],
                    lr=TRAIN_CONFIG['learning_rate'],
                    device=device,
                    checkpoint_dir=checkpoint_dir,
                    figures_dir=figures_dir
                )
                
                print("  [4/4] 评估模型...")
                results = evaluate_model(model, test_loader, class_names, device,
                                        tail_class_ids=eval_tail_class_ids,
                                        head_class_ids=eval_head_class_ids)
                
                metrics_dir = os.path.join(exp_dir, 'metrics')
                os.makedirs(metrics_dir, exist_ok=True)
                
                with open(os.path.join(metrics_dir, 'metrics.json'), 'w') as f:
                    json.dump(results, f, indent=4)
                
                plot_confusion_matrix(
                    np.array(results['confusion_matrix']), class_names,
                    os.path.join(figures_dir, 'confusion_matrix.png')
                )
                plot_per_class_accuracy(
                    results['per_class_accuracy'],
                    os.path.join(figures_dir, 'per_class_accuracy.png')
                )
                plot_head_tail_comparison(
                    results['per_class_accuracy'], eval_head_class_ids, eval_tail_class_ids,
                    os.path.join(figures_dir, 'head_tail_comparison.png')
                )
                plot_tsne_features(
                    model, test_loader, device,
                    os.path.join(figures_dir, 'tsne_features.png')
                )
                
                print(f"\n  ✅ {exp_name} 完成: Overall={results['overall_accuracy']:.4f} BAcc={results['balanced_accuracy']:.4f} Head={results['head_avg_accuracy']:.4f} Tail={results['tail_avg_accuracy']:.4f} F1={results['macro_f1']:.4f}")
                
                all_results[exp_name] = results
                
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\n  ❌ {exp_name} 失败: {str(e)}")
                import traceback
                traceback.print_exc()
                all_results[exp_name] = {'error': str(e)}
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
    print(f"{'实验':<50} {'Overall':<10} {'Balanced':<12} {'Head':<10} {'Tail':<10} {'Macro F1':<12}")
    print("-" * 110)
    
    for exp_name, results in all_results.items():
        if 'error' in results:
            print(f"{exp_name:<50} {'ERROR':<10}")
        else:
            print(f"{exp_name:<50} {results['overall_accuracy']:<10.4f} {results['balanced_accuracy']:<12.4f} {results['head_avg_accuracy']:<10.4f} {results['tail_avg_accuracy']:<10.4f} {results['macro_f1']:<12.4f}")
    
    print(f"\n✅ 所有结果已保存到: {exp_root}")
    print(f"✅ 汇总结果: {os.path.join(exp_root, 'all_results.json')}")
    
    return all_results


# ==========================================
# 命令行参数
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合成数据集消融实验脚本')
    
    parser.add_argument('--original_data_dir', type=str, 
                       default='../data/original',
                       help='原始训练集路径')
    parser.add_argument('--generated_data_dir', type=str,
                       default='../../Ship/DataGen',
                       help='生成图像路径（每个类别一个子文件夹）')
    parser.add_argument('--test_dir', type=str,
                       default='../data/test',
                       help='测试集路径')
    parser.add_argument('--pretrained_weights', type=str,
                       default='../pretrained_weights/resnet50-0676ba61.pth',
                       help='ResNet50预训练权重路径')
    parser.add_argument('--base_dir', type=str,
                       default='..',
                       help='基础目录')
    
    args = parser.parse_args()
    
    # 运行实验
    results = run_ablation_study(args)
