import torch
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from scipy import stats
from sklearn.covariance import LedoitWolf

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 配置
# ==========================================

ORIGINAL_DATA_DIR = '../data/original'
PRETRAINED_WEIGHTS = '../pretrained_weights/resnet50-0676ba61.pth'
OUTPUT_DIR = '../feature_distribution_analysis'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 工具函数
# ==========================================

def extract_features(image_paths, model, transform, device, batch_size=32):
    """提取 ResNet 特征"""
    model.eval()
    features = []
    
    from PIL import Image
    
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


def test_normality(features, method='shapiro'):
    """
    测试特征是否服从正态分布
    注意：高维数据需要降维后测试
    """
    from sklearn.decomposition import PCA
    
    # 用 PCA 降维到 1D 进行测试
    pca = PCA(n_components=1)
    features_1d = pca.fit_transform(features).flatten()
    
    if method == 'shapiro':
        # Shapiro-Wilk 检验（适合小样本）
        if len(features_1d) > 5000:
            # 采样
            features_1d = np.random.choice(features_1d, 5000, replace=False)
        stat, p_value = stats.shapiro(features_1d)
    elif method == 'kstest':
        # Kolmogorov-Smirnov 检验
        stat, p_value = stats.kstest(features_1d, 'norm')
    elif method == 'anderson':
        # Anderson-Darling 检验
        result = stats.anderson(features_1d, dist='norm')
        stat = result.statistic
        p_value = result.significance_level[2]  # 5% 显著性水平
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return stat, p_value


def test_mahalanobis_chi2(original_features, candidate_features):
    """
    测试马氏距离是否服从卡方分布
    """
    mean = np.mean(original_features, axis=0)
    
    # 使用 Ledoit-Wolf 估计协方差（更稳定）
    lw = LedoitWolf()
    cov = lw.fit(original_features).covariance_
    
    # 正则化
    cov += 1e-6 * np.eye(cov.shape[0])
    
    try:
        cov_inv = np.linalg.inv(cov)
    except:
        cov_inv = np.linalg.pinv(cov)
    
    # 计算原始样本的马氏距离平方
    diff = original_features - mean
    mahal_sq = np.sum(diff @ cov_inv * diff, axis=1)
    
    # 计算候选样本的马氏距离平方
    diff_cand = candidate_features - mean
    mahal_sq_cand = np.sum(diff_cand @ cov_inv * diff_cand, axis=1)
    
    # K-S 检验：测试是否服从卡方分布
    dof = original_features.shape[1]
    stat, p_value = stats.kstest(mahal_sq, 'chi2', args=(dof,))
    
    return stat, p_value, mahal_sq, mahal_sq_cand


def plot_distribution(mahal_sq, dof, class_id, save_path):
    """绘制马氏距离分布与卡方分布对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 直方图 + 卡方分布曲线
    ax1 = axes[0]
    ax1.hist(mahal_sq, bins=50, density=True, alpha=0.6, color='skyblue', label='Empirical')
    
    x = np.linspace(0, np.max(mahal_sq) * 1.2, 200)
    ax1.plot(x, stats.chi2.pdf(x, dof), 'r-', linewidth=2, label=f'Chi2(df={dof})')
    ax1.set_xlabel('Mahalanobis Distance Squared')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Class {class_id}: Distribution vs Chi2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q 图
    ax2 = axes[1]
    stats.probplot(mahal_sq, dist=stats.chi2, sparams=(dof,), plot=ax2)
    ax2.set_title(f'Class {class_id}: Q-Q Plot vs Chi2')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig


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
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 遍历所有类别
    results = {}
    
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
        
        # 测试正态性（PCA 降维后）
        shapiro_stat, shapiro_p = test_normality(features, method='shapiro')
        print(f"  Shapiro-Wilk test: stat={shapiro_stat:.4f}, p={shapiro_p:.6f}")
        print(f"  -> {'Approximately normal' if shapiro_p > 0.05 else 'NOT normal'} (alpha=0.05)")
        
        # 测试马氏距离的卡方分布
        # 用自身作为候选（测试理论假设）
        chi2_stat, chi2_p, mahal_sq, _ = test_mahalanobis_chi2(features, features)
        print(f"  Mahalanobis Chi2 test: stat={chi2_stat:.4f}, p={chi2_p:.6f}")
        print(f"  -> {'Approximately Chi2' if chi2_p > 0.05 else 'NOT Chi2'} (alpha=0.05)")
        
        # 绘制分布图
        dof = features.shape[1]
        plot_path = os.path.join(OUTPUT_DIR, f'class_{class_name}_distribution.png')
        plot_distribution(mahal_sq, dof, class_name, plot_path)
        print(f"  Distribution plot saved to {plot_path}")
        
        results[class_name] = {
            'n_samples': len(image_paths),
            'shapiro_stat': float(shapiro_stat),
            'shapiro_p': float(shapiro_p),
            'chi2_stat': float(chi2_stat),
            'chi2_p': float(chi2_p),
            'is_normal': bool(shapiro_p > 0.05),
            'is_chi2': bool(chi2_p > 0.05),
        }
    
    # 保存汇总结果
    import json
    summary_path = os.path.join(OUTPUT_DIR, 'distribution_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    
    # 打印汇总表格
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Class':<10} {'Samples':<10} {'Shapiro-p':<12} {'Normal?':<10} {'Chi2-p':<12} {'Chi2?':<10}")
    print("-" * 80)
    
    normal_count = 0
    chi2_count = 0
    total = 0
    
    for class_name, res in sorted(results.items()):
        total += 1
        if res['is_normal']:
            normal_count += 1
        if res['is_chi2']:
            chi2_count += 1
        
        print(f"{class_name:<10} {res['n_samples']:<10} {res['shapiro_p']:<12.4f} "
              f"{'Yes' if res['is_normal'] else 'No':<10} {res['chi2_p']:<12.4f} "
              f"{'Yes' if res['is_chi2'] else 'No':<10}")
    
    print("-" * 80)
    print(f"Normal: {normal_count}/{total} ({normal_count/total*100:.1f}%)")
    print(f"Chi2: {chi2_count}/{total} ({chi2_count/total*100:.1f}%)")
    
    # 生成汇总图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 正态性 p 值分布
    p_values_normal = [r['shapiro_p'] for r in results.values()]
    axes[0].hist(p_values_normal, bins=20, color='skyblue', edgecolor='black')
    axes[0].axvline(x=0.05, color='red', linestyle='--', label='alpha=0.05')
    axes[0].set_xlabel('Shapiro-Wilk p-value')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Normality Test p-values')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 卡方检验 p 值分布
    p_values_chi2 = [r['chi2_p'] for r in results.values()]
    axes[1].hist(p_values_chi2, bins=20, color='lightgreen', edgecolor='black')
    axes[1].axvline(x=0.05, color='red', linestyle='--', label='alpha=0.05')
    axes[1].set_xlabel('Chi2 Test p-value')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Mahalanobis Chi2 Test p-values')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'summary_pvalues.png'), dpi=150, bbox_inches='tight')
    print(f"\nSummary plot saved to {os.path.join(OUTPUT_DIR, 'summary_pvalues.png')}")


if __name__ == '__main__':
    main()
