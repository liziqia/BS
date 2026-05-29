import torch
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from scipy import stats

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 配置
# ==========================================

ORIGINAL_DATA_DIR = '../data/original'
PRETRAINED_WEIGHTS = '../pretrained_weights/resnet50-0676ba61.pth'
OUTPUT_DIR = '../cosine_similarity_analysis'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 工具函数
# ==========================================

def extract_features(image_paths, model, transform, device, batch_size=32):
    """提取 ResNet 特征"""
    model.eval()
    features = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            img = transform(img)
            batch_images.append(img)
        
        batch = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            feat = model.conv1(batch)
            feat = model.bn1(feat)
            feat = model.relu(feat)
            feat = model.maxpool(feat)
            feat = model.layer1(feat)
            feat = model.layer2(feat)
            feat = model.layer3(feat)
            feat = model.layer4(feat)
            feat = model.avgpool(feat)
            feat = torch.flatten(feat, 1)
            features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def compute_cosine_similarity_to_center(features):
    """计算所有样本与特征中心的余弦相似度"""
    center = np.mean(features, axis=0)
    
    # 归一化
    features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
    center_norm = center / np.linalg.norm(center)
    
    # 余弦相似度
    cosine_sim = np.dot(features_norm, center_norm)
    
    return cosine_sim


def test_distribution(cosine_sim, method='kstest'):
    """测试余弦相似度服从什么分布"""
    results = {}
    
    # 1. 正态分布检验
    if method == 'shapiro':
        stat, p_value = stats.shapiro(cosine_sim)
        results['normal_shapiro'] = {'stat': stat, 'p': p_value}
    
    # 2. K-S 检验：正态分布
    stat, p_value = stats.kstest(cosine_sim, 'norm', args=(np.mean(cosine_sim), np.std(cosine_sim)))
    results['normal_ks'] = {'stat': stat, 'p': p_value}
    
    # 3. K-S 检验：Beta 分布（余弦相似度通常在 [-1, 1] 之间）
    # 先转换到 [0, 1]
    sim_min = np.min(cosine_sim)
    sim_max = np.max(cosine_sim)
    sim_range = sim_max - sim_min
    if sim_range < 1e-10:
        results['beta_ks'] = {'stat': 0, 'p': 1.0, 'params': {'a': 1, 'b': 1}, 'note': 'range too small'}
    else:
        sim_scaled = (cosine_sim - sim_min) / sim_range
        # 裁剪到 (0, 1) 开区间，避免 Beta 拟合错误
        eps = 1e-6
        sim_scaled = np.clip(sim_scaled, eps, 1 - eps)
        # 拟合 Beta 分布
        try:
            a, b, loc, scale = stats.beta.fit(sim_scaled, floc=0, fscale=1)
            stat, p_value = stats.kstest(sim_scaled, 'beta', args=(a, b, 0, 1))
            results['beta_ks'] = {'stat': stat, 'p': p_value, 'params': {'a': a, 'b': b}}
        except Exception as e:
            results['beta_ks'] = {'stat': 0, 'p': 1.0, 'params': {'a': 1, 'b': 1}, 'note': str(e)}
    
    # 4. K-S 检验：截断正态分布
    from scipy.stats import truncnorm
    a_trunc = (np.min(cosine_sim) - np.mean(cosine_sim)) / np.std(cosine_sim)
    b_trunc = (np.max(cosine_sim) - np.mean(cosine_sim)) / np.std(cosine_sim)
    stat, p_value = stats.kstest(cosine_sim, 'truncnorm', args=(a_trunc, b_trunc, np.mean(cosine_sim), np.std(cosine_sim)))
    results['truncnorm_ks'] = {'stat': stat, 'p': p_value}
    
    # 5. 经验分位数分析
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    results['percentiles'] = {f'p{p}': np.percentile(cosine_sim, p) for p in percentiles}
    
    return results


def plot_distribution(cosine_sim, class_id, save_path, dist_results=None):
    """绘制余弦相似度分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 直方图
    ax1 = axes[0, 0]
    ax1.hist(cosine_sim, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    ax1.axvline(x=np.mean(cosine_sim), color='red', linestyle='--', label=f'Mean={np.mean(cosine_sim):.4f}')
    ax1.axvline(x=np.median(cosine_sim), color='green', linestyle='--', label=f'Median={np.median(cosine_sim):.4f}')
    ax1.set_xlabel('Cosine Similarity to Center')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Class {class_id}: Cosine Similarity Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q 图（正态）
    ax2 = axes[0, 1]
    stats.probplot(cosine_sim, dist='norm', plot=ax2)
    ax2.set_title(f'Class {class_id}: Q-Q Plot vs Normal')
    ax2.grid(True, alpha=0.3)
    
    # 3. 置信区间可视化
    ax3 = axes[1, 0]
    sorted_sim = np.sort(cosine_sim)
    n = len(sorted_sim)
    x = np.arange(n)
    ax3.plot(x, sorted_sim, 'b-', alpha=0.7)
    
    # 标记不同置信区间
    for conf, color, label in [(0.95, 'green', '95% CI'), (0.99, 'orange', '99% CI'), (0.997, 'red', '99.7% CI')]:
        lower = np.percentile(cosine_sim, (1 - conf) / 2 * 100)
        upper = np.percentile(cosine_sim, (1 + conf) / 2 * 100)
        ax3.axhline(y=lower, color=color, linestyle='--', alpha=0.5)
        ax3.axhline(y=upper, color=color, linestyle='--', alpha=0.5)
        ax3.fill_between(x, lower, upper, alpha=0.1, color=color, label=label)
    
    ax3.set_xlabel('Sorted Index')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title(f'Class {class_id}: Confidence Intervals')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 箱线图统计
    ax4 = axes[1, 1]
    ax4.boxplot(cosine_sim, vert=False)
    ax4.set_xlabel('Cosine Similarity')
    ax4.set_title(f'Class {class_id}: Box Plot')
    ax4.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = (
        f'Mean: {np.mean(cosine_sim):.4f}\n'
        f'Std: {np.std(cosine_sim):.4f}\n'
        f'Min: {np.min(cosine_sim):.4f}\n'
        f'Max: {np.max(cosine_sim):.4f}\n'
        f'Q1: {np.percentile(cosine_sim, 25):.4f}\n'
        f'Q3: {np.percentile(cosine_sim, 75):.4f}'
    )
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_noise_filtering(cosine_sim, class_id):
    """分析不同阈值对噪声过滤的影响"""
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    results = {}
    for thresh in thresholds:
        # 低于阈值的样本比例（会被过滤）
        filtered_ratio = np.mean(cosine_sim < thresh)
        # 保留的样本比例
        kept_ratio = 1 - filtered_ratio
        
        results[thresh] = {
            'filtered_ratio': filtered_ratio,
            'kept_ratio': kept_ratio,
            'n_filtered': int(filtered_ratio * len(cosine_sim)),
            'n_kept': int(kept_ratio * len(cosine_sim))
        }
    
    return results


# ==========================================
# 主流程
# ==========================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载模型
    model = models.resnet50(weights=None)
    if os.path.exists(PRETRAINED_WEIGHTS):
        model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location='cpu'))
        print(f"Loaded pretrained weights from {PRETRAINED_WEIGHTS}")
    
    # 移除 fc 层
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 遍历所有类别
    all_results = {}
    
    for class_name in sorted(os.listdir(ORIGINAL_DATA_DIR)):
        class_dir = os.path.join(ORIGINAL_DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(image_paths) < 10:
            print(f"Class {class_name}: Skip (only {len(image_paths)} images)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Analyzing Class {class_name} ({len(image_paths)} images)")
        print(f"{'='*60}")
        
        # 提取特征
        features = extract_features(image_paths, model, transform, device)
        print(f"  Feature shape: {features.shape}")
        
        # 计算余弦相似度
        cosine_sim = compute_cosine_similarity_to_center(features)
        print(f"  Cosine similarity range: [{np.min(cosine_sim):.4f}, {np.max(cosine_sim):.4f}]")
        print(f"  Mean: {np.mean(cosine_sim):.4f}, Std: {np.std(cosine_sim):.4f}")
        
        # 测试分布
        dist_results = test_distribution(cosine_sim)
        print(f"  Normal KS test: p={dist_results['normal_ks']['p']:.6f}")
        print(f"  Beta KS test: p={dist_results['beta_ks']['p']:.6f}")
        print(f"  Percentiles: {dist_results['percentiles']}")
        
        # 绘制分布图
        plot_path = os.path.join(OUTPUT_DIR, f'class_{class_name}_cosine.png')
        plot_distribution(cosine_sim, class_name, plot_path, dist_results)
        print(f"  Distribution plot saved to {plot_path}")
        
        # 分析噪声过滤
        filter_results = analyze_noise_filtering(cosine_sim, class_name)
        print(f"  Filtering analysis:")
        for thresh, res in filter_results.items():
            print(f"    Threshold {thresh}: filter {res['n_filtered']} ({res['filtered_ratio']*100:.1f}%), keep {res['n_kept']}")
        
        all_results[class_name] = {
            'n_samples': len(image_paths),
            'mean': float(np.mean(cosine_sim)),
            'std': float(np.std(cosine_sim)),
            'min': float(np.min(cosine_sim)),
            'max': float(np.max(cosine_sim)),
            'percentiles': {k: float(v) for k, v in dist_results['percentiles'].items()},
            'normal_ks_p': float(dist_results['normal_ks']['p']),
            'beta_ks_p': float(dist_results['beta_ks']['p']),
            'beta_params': dist_results['beta_ks']['params'],
            'filtering': {str(k): v for k, v in filter_results.items()}
        }
    
    # 保存汇总结果
    import json
    summary_path = os.path.join(OUTPUT_DIR, 'cosine_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    
    # 打印汇总表格
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Class':<10} {'Samples':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Normal-p':<10}")
    print("-" * 80)
    
    for class_name, res in sorted(all_results.items()):
        print(f"{class_name:<10} {res['n_samples']:<10} {res['mean']:<10.4f} {res['std']:<10.4f} "
              f"{res['min']:<10.4f} {res['max']:<10.4f} {res['normal_ks_p']:<10.4f}")
    
    # 生成汇总图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 各类别均值分布
    means = [r['mean'] for r in all_results.values()]
    axes[0, 0].bar(range(len(means)), means, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Mean Cosine Similarity')
    axes[0, 0].set_title('Mean Cosine Similarity by Class')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 各类别标准差分布
    stds = [r['std'] for r in all_results.values()]
    axes[0, 1].bar(range(len(stds)), stds, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Std of Cosine Similarity')
    axes[0, 1].set_title('Std of Cosine Similarity by Class')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 正态性检验 p 值
    p_values = [r['normal_ks_p'] for r in all_results.values()]
    axes[1, 0].hist(p_values, bins=20, color='orange', edgecolor='black')
    axes[1, 0].axvline(x=0.05, color='red', linestyle='--', label='alpha=0.05')
    axes[1, 0].set_xlabel('Normal KS Test p-value')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Normality Test p-values')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 过滤阈值分析（取平均值）
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    avg_filtered = []
    for thresh in thresholds:
        filtered_ratios = []
        for res in all_results.values():
            if str(thresh) in res['filtering']:
                filtered_ratios.append(res['filtering'][str(thresh)]['filtered_ratio'])
        avg_filtered.append(np.mean(filtered_ratios) * 100)
    
    axes[1, 1].plot(thresholds, avg_filtered, 'bo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Avg Filtered Ratio (%)')
    axes[1, 1].set_title('Avg Filtered Ratio by Threshold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(thresholds)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cosine_summary_plots.png'), dpi=150, bbox_inches='tight')
    print(f"\nSummary plots saved to {os.path.join(OUTPUT_DIR, 'cosine_summary_plots.png')}")


if __name__ == '__main__':
    main()
