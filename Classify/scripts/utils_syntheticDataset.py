import torch
import torch.nn as nn
import numpy as np
import os
import shutil
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import mahalanobis


def selection_inverse_sigma_random(candidate_features, original_features, n_select, sigma=1.0):
    """
    反向σ过滤 + 随机选择
    保留余弦相似度低于阈值的样本（原来被过滤的），从中随机选择
    
    直觉：3σ过滤"好样本"降低了效果 → 被过滤的"离群样本"反而可能是有价值的多样性来源
    
    参数:
        candidate_features: 候选图像特征 [M, D]
        original_features: 原始图像特征 [N, D]
        n_select: 选择数量
        sigma: σ倍数（默认1.0，阈值=μ-1σ，离群池约16%）
    
    返回:
        selected_indices: 选中的索引列表
        filter_info: 过滤信息字典
    """
    # 1. 计算原始特征中心
    center = np.mean(original_features, axis=0)
    
    # 2. 计算余弦相似度
    orig_norm = original_features / (np.linalg.norm(original_features, axis=1, keepdims=True) + 1e-10)
    center_norm = center / (np.linalg.norm(center) + 1e-10)
    orig_cosine = np.dot(orig_norm, center_norm)
    orig_mean = np.mean(orig_cosine)
    orig_std = np.std(orig_cosine)
    
    candidate_norm = candidate_features / (np.linalg.norm(candidate_features, axis=1, keepdims=True) + 1e-10)
    cosine_sim = np.dot(candidate_norm, center_norm)
    
    # 3. 阈值（σ越小阈值越高，被"判定为离群"的越多）
    threshold = orig_mean - sigma * orig_std
    
    # 4. 反向：选低于阈值的（原来会被过滤掉的）
    inverse_mask = cosine_sim < threshold
    inverse_indices = np.where(inverse_mask)[0]
    remaining_indices = np.where(~inverse_mask)[0]
    
    filter_info = {
        'n_candidates': len(candidate_features),
        'n_inverse': len(inverse_indices),
        'n_remaining': len(remaining_indices),
        'threshold': float(threshold),
        'sigma': sigma,
        'orig_mean': float(orig_mean),
        'orig_std': float(orig_std),
        'cosine_sim_min': float(np.min(cosine_sim)),
        'cosine_sim_max': float(np.max(cosine_sim)),
        'cosine_sim_mean': float(np.mean(cosine_sim)),
    }
    
    n_selected = 0
    selected = []
    
    # 优先从反向池随机选
    if len(inverse_indices) > 0:
        n_from_inverse = min(n_select, len(inverse_indices))
        selected = np.random.choice(inverse_indices, size=n_from_inverse, replace=False).tolist()
        n_selected = len(selected)
    
    # 不足则从剩余池补
    if n_selected < n_select and len(remaining_indices) > 0:
        n_from_remaining = min(n_select - n_selected, len(remaining_indices))
        fill = np.random.choice(remaining_indices, size=n_from_remaining, replace=False).tolist()
        selected = selected + fill
        n_selected = len(selected)
    
    filter_info['n_from_inverse'] = min(n_select, len(inverse_indices)) if len(inverse_indices) > 0 else 0
    filter_info['n_from_remaining'] = n_selected - filter_info['n_from_inverse']
    
    return selected, filter_info



#############################################
# 一、加权数量生成机制 (GQW)
#############################################

def extract_features_clip(image_paths, model, preprocess, device, batch_size=32):
    """
    使用CLIP提取图像特征
    
    参数:
        image_paths: 图像路径列表
        model: CLIP模型
        preprocess: CLIP预处理函数
        device: 设备
        batch_size: 批次大小
    
    返回:
        features: numpy数组 [N, feature_dim]
    """
    import clip
    
    model.eval()
    features = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        
        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                images.append(preprocess(image))
            except:
                continue
        
        if len(images) == 0:
            continue
        
        images = torch.stack(images).to(device)
        
        with torch.no_grad():
            feature = model.encode_image(images)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature.cpu().numpy())
    
    if len(features) == 0:
        return np.array([])
    
    return np.concatenate(features, axis=0)


def extract_features_resnet(image_paths, model, transform, device, batch_size=32):
    """
    使用ResNet提取图像特征（backbone输出，不含fc）
    
    参数:
        image_paths: 图像路径列表
        model: ResNet模型
        transform: 预处理transform
        device: 设备
        batch_size: 批次大小
    
    返回:
        features: numpy数组 [N, 2048]
    """
    model.eval()
    features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []

        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                images.append(transform(image))
            except:
                continue

        if len(images) == 0:
            continue

        images = torch.stack(images).to(device)

        with torch.no_grad():
            x = model.conv1(images)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            features.append(x.cpu().numpy())

    if len(features) == 0:
        return np.array([])

    return np.concatenate(features, axis=0)


def compute_intra_variation(features):
    """
    计算类内差异度 IV
    使用平均余弦距离
    
    参数:
        features: [N, D] 特征矩阵
    
    返回:
        iv: 标量，类内差异度
    """
    if len(features) < 2:
        return 0.0
    
    # 计算余弦距离矩阵
    dist_matrix = pairwise_distances(features, metric='cosine')
    
    # 平均距离（排除对角线）
    n = len(features)
    iv = dist_matrix.sum() / (n * (n - 1))
    
    return float(iv)


def compute_inter_overlap_knn(features_current, features_others, k=5):
    """
    计算类间重合度 OR（使用KNN方法）
    
    参数:
        features_current: 当前类别特征 [N1, D]
        features_others: 其他类别特征 [N2, D]
        k: KNN的k值
    
    返回:
        or_score: 标量，重合度（被错误分类的比例）
    """
    if len(features_current) == 0 or len(features_others) == 0:
        return 0.0
    
    # 合并特征
    X = np.concatenate([features_current, features_others], axis=0)
    y = np.concatenate([np.zeros(len(features_current)), np.ones(len(features_others))])
    
    # 训练KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    
    # 预测当前类别样本
    preds = knn.predict(features_current)
    
    # 计算错误率
    error_rate = (preds == 1).sum() / len(features_current)
    
    return float(error_rate)


def compute_sample_sufficiency(class_counts):
    """
    计算样本充足度 SC
    
    参数:
        class_counts: 各类别样本数列表
    
    返回:
        sc_scores: 各类别的样本充足度列表
    """
    max_count = max(class_counts)
    if max_count == 0:
        return [1.0] * len(class_counts)
    
    return [count / max_count for count in class_counts]


def compute_gqw_weights(class_features_dict, class_counts, 
                        alpha=0.4, beta=0.4, gamma=0.2,
                        w_min=0.6, w_max=1.4):
    """
    计算GQW权重
    
    参数:
        class_features_dict: {class_id: features_array}
        class_counts: {class_id: count}
        alpha, beta, gamma: 权重系数
        w_min, w_max: 归一化范围
    
    返回:
        gqw_weights: {class_id: weight}
    """
    class_ids = list(class_features_dict.keys())
    n_classes = len(class_ids)
    
    # 计算各子指标
    iv_scores = []
    or_scores = []
    sc_scores = compute_sample_sufficiency([class_counts[cid] for cid in class_ids])
    
    for i, cid in enumerate(class_ids):
        # 类内差异
        iv = compute_intra_variation(class_features_dict[cid])
        iv_scores.append(iv)
        
        # 类间重合
        other_features = np.concatenate(
            [class_features_dict[other_cid] for other_cid in class_ids if other_cid != cid],
            axis=0
        )
        or_score = compute_inter_overlap_knn(class_features_dict[cid], other_features)
        or_scores.append(or_score)
    
    # 归一化到[0, 1]
    def normalize(scores):
        min_val = min(scores)
        max_val = max(scores)
        if max_val - min_val < 1e-8:
            return [0.5] * len(scores)
        return [(s - min_val) / (max_val - min_val) for s in scores]
    
    iv_norm = normalize(iv_scores)
    or_norm = normalize(or_scores)
    sc_norm = sc_scores  # 已经在[0, 1]范围内
    
    # 计算GQW
    gqw_raw = []
    for i in range(n_classes):
        gqw = alpha * iv_norm[i] + beta * or_norm[i] + gamma * (1 - sc_norm[i])
        gqw_raw.append(gqw)
    
    # 归一化到[w_min, w_max]
    gqw_norm = normalize(gqw_raw)
    gqw_final = [w * (w_max - w_min) + w_min for w in gqw_norm]
    
    return {cid: w for cid, w in zip(class_ids, gqw_final)}


def allocate_generation_counts(n_tail_classes, n_total, gqw_weights):
    """
    分配生成数量（单超参数版本）
    
    参数:
        n_tail_classes: 尾部类别数量
        n_total: 总生成数量
        gqw_weights: {class_id: weight}
    
    返回:
        gen_counts: {class_id: count}
    """
    class_ids = list(gqw_weights.keys())
    weights = [gqw_weights[cid] for cid in class_ids]
    
    # 步骤1: 计算原始权重
    n_raw = weights
    
    # 步骤2: 全局归一化
    total_raw = sum(n_raw)
    factor = n_total / total_raw
    n_scaled = [n * factor for n in n_raw]
    
    # 步骤3: 向下取整
    n_floor = [int(n) for n in n_scaled]
    fractions = [n - f for n, f in zip(n_scaled, n_floor)]
    
    # 步骤4: 调整舍入误差
    current_total = sum(n_floor)
    diff = n_total - current_total
    
    # 按小数部分排序
    indices = sorted(range(len(fractions)), key=lambda i: fractions[i], reverse=True)
    
    for i in range(abs(diff)):
        idx = indices[i]
        if diff > 0:
            n_floor[idx] += 1
        else:
            n_floor[idx] -= 1
    
    return {cid: count for cid, count in zip(class_ids, n_floor)}


#############################################
# 二、生成图像选择机制
#############################################

def selection_fds(candidate_features, original_features, n_select, lambda_param=0.5):
    """
    方案A: 基于特征多样性的选择 (FDS)
    使用贪心MMR策略
    
    参数:
        candidate_features: 候选图像特征 [M, D]
        original_features: 原始图像特征 [N, D]
        n_select: 选择数量
        lambda_param: 平衡参数
    
    返回:
        selected_indices: 选中的索引列表
    """
    if len(candidate_features) <= n_select:
        return list(range(len(candidate_features)))
    
    # 原始特征均值
    mu_orig = original_features.mean(axis=0)
    
    selected = []
    remaining = list(range(len(candidate_features)))
    
    for _ in range(n_select):
        best_idx = -1
        best_score = -float('inf')
        
        for idx in remaining:
            # 与原始分布的相似度
            sim_to_orig = 1 - pairwise_distances(
                candidate_features[idx:idx+1], mu_orig.reshape(1, -1), metric='cosine'
            )[0][0]
            
            # 与已选样本的平均相似度
            if len(selected) > 0:
                selected_feats = candidate_features[selected]
                sim_to_selected = pairwise_distances(
                    candidate_features[idx:idx+1], selected_feats, metric='cosine'
                ).mean()
            else:
                sim_to_selected = 0
            
            # MMR分数
            score = lambda_param * sim_to_orig - (1 - lambda_param) * sim_to_selected
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return selected


def selection_cus_cosine(candidate_features, original_features, classifier, n_select, threshold=0.75):
    """
    基于余弦相似度过滤的 CUS 选择机制
    先用余弦相似度过滤低质量样本，再对剩余样本进行不确定性选择
    
    参数:
        candidate_features: 候选图像特征 [M, D]
        original_features: 原始图像特征 [N, D]
        classifier: 分类器
        n_select: 选择数量
        threshold: 余弦相似度阈值（默认 0.75，过滤明显语义错误的样本）
    
    返回:
        selected_indices: 选中的索引列表
        filter_info: 过滤信息字典
    """
    # 1. 计算原始特征中心
    center = np.mean(original_features, axis=0)
    
    # 2. 计算候选样本与中心的余弦相似度
    candidate_norm = candidate_features / (np.linalg.norm(candidate_features, axis=1, keepdims=True) + 1e-10)
    center_norm = center / (np.linalg.norm(center) + 1e-10)
    cosine_sim = np.dot(candidate_norm, center_norm)
    
    # 3. 计算原始样本的余弦相似度统计量（仅用于日志）
    orig_norm = original_features / (np.linalg.norm(original_features, axis=1, keepdims=True) + 1e-10)
    orig_cosine_sim = np.dot(orig_norm, center_norm)
    orig_mean = np.mean(orig_cosine_sim)
    orig_std = np.std(orig_cosine_sim)
    
    # 4. 过滤低相似度样本
    valid_mask = cosine_sim >= threshold
    valid_indices = np.where(valid_mask)[0]
    
    filter_info = {
        'n_candidates': len(candidate_features),
        'n_valid': len(valid_indices),
        'n_filtered': len(candidate_features) - len(valid_indices),
        'filter_ratio': (len(candidate_features) - len(valid_indices)) / len(candidate_features),
        'threshold': float(threshold),
        'orig_mean': float(orig_mean),
        'orig_std': float(orig_std),
        'cosine_sim_min': float(np.min(cosine_sim)),
        'cosine_sim_max': float(np.max(cosine_sim)),
        'cosine_sim_mean': float(np.mean(cosine_sim)),
    }
    
    if len(valid_indices) == 0:
        # 如果没有有效样本，放宽阈值
        threshold -= 0.05
        valid_mask = cosine_sim >= threshold
        valid_indices = np.where(valid_mask)[0]
        filter_info['threshold_relaxed'] = float(threshold)
        filter_info['n_valid_after_relax'] = len(valid_indices)
    
    if len(valid_indices) <= n_select:
        # 如果有效样本不足，直接返回所有有效样本
        return valid_indices.tolist(), filter_info
    
    # 5. 对有效样本进行 CUS 选择
    valid_features = candidate_features[valid_indices]
    probs = classifier.predict_proba(valid_features)
    entropies = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    
    # 选择熵最高的
    sorted_indices = np.argsort(entropies)[::-1][:n_select]
    
    # 映射回原始索引
    return [valid_indices[i] for i in sorted_indices], filter_info


def selection_random(candidate_features, n_select):
    """
    随机选择机制
    
    参数:
        candidate_features: 候选图像特征 [M, D]
        n_select: 选择数量
    
    返回:
        selected_indices: 随机选中的索引列表
    """
    n_candidates = len(candidate_features)
    return np.random.choice(n_candidates, size=min(n_select, n_candidates), replace=False).tolist()


def selection_cus_3sigma(candidate_features, original_features, classifier, n_select, confidence=0.997):
    """
    基于 3σ 原则过滤的 CUS 选择机制
    使用马氏距离过滤异常样本（低质量/噪声），再对剩余样本进行不确定性选择
    
    参数:
        candidate_features: 候选图像特征 [M, D]
        original_features: 原始图像特征 [N, D]
        classifier: 分类器
        n_select: 选择数量
        confidence: 置信度（默认 0.997 对应 3σ）
    
    返回:
        selected_indices: 选中的索引列表
    """
    from scipy.stats import chi2
    
    # 1. 计算原始特征的统计量
    mean = np.mean(original_features, axis=0)
    cov = np.cov(original_features, rowvar=False)
    
    # 正则化协方差矩阵（防止奇异）
    cov += 1e-6 * np.eye(cov.shape[0])
    
    try:
        cov_inv = np.linalg.inv(cov)
    except:
        # 如果求逆失败，使用伪逆
        cov_inv = np.linalg.pinv(cov)
    
    # 2. 计算马氏距离
    diff = candidate_features - mean
    mahal_dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    
    # 3. 计算 3σ 阈值（卡方分布）
    # 马氏距离的平方服从卡方分布，自由度为特征维度
    dof = candidate_features.shape[1]
    threshold_sq = chi2.ppf(confidence, dof)
    threshold = np.sqrt(threshold_sq)
    
    print(f"    3σ 阈值: {threshold:.2f}, 过滤前样本数: {len(candidate_features)}")
    
    # 4. 过滤异常样本
    valid_mask = mahal_dist <= threshold
    valid_indices = np.where(valid_mask)[0]
    
    print(f"    过滤后样本数: {len(valid_indices)} (过滤掉 {len(candidate_features) - len(valid_indices)} 个异常样本)")
    
    if len(valid_indices) == 0:
        # 如果没有有效样本，放宽阈值
        print("    警告: 无有效样本，放宽阈值至 95%")
        threshold_sq = chi2.ppf(0.95, dof)
        threshold = np.sqrt(threshold_sq)
        valid_mask = mahal_dist <= threshold
        valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) <= n_select:
        # 如果有效样本不足，直接返回所有有效样本
        return valid_indices.tolist()
    
    # 5. 对有效样本进行 CUS 选择
    valid_features = candidate_features[valid_indices]
    probs = classifier.predict_proba(valid_features)
    entropies = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    
    # 选择熵最高的
    sorted_indices = np.argsort(entropies)[::-1][:n_select]
    
    # 映射回原始索引
    return [valid_indices[i] for i in sorted_indices]


def selection_cus(candidate_features, classifier, n_select):
    """
    方案B: 基于分类器不确定性的选择 (CUS)
    使用熵作为不确定性度量
    
    参数:
        candidate_features: 候选图像特征 [M, D]
        classifier: 分类器（需要有predict_proba方法）
        n_select: 选择数量
    
    返回:
        selected_indices: 选中的索引列表（按不确定性降序）
    """
    # 预测概率
    probs = classifier.predict_proba(candidate_features)
    
    # 计算熵
    entropies = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    
    # 选择熵最高的
    sorted_indices = np.argsort(entropies)[::-1]
    
    return sorted_indices[:n_select].tolist()


def selection_dms(candidate_features, original_features, target_mean, target_cov, n_select):
    """
    方案C: 基于分布匹配的选择 (DMS)
    使用马氏距离简化实现
    
    参数:
        candidate_features: 候选图像特征 [M, D]
        original_features: 原始图像特征 [N, D]
        target_mean: 目标分布均值 [D]
        target_cov: 目标分布协方差 [D, D]
        n_select: 选择数量
    
    返回:
        selected_indices: 选中的索引列表
    """
    if len(candidate_features) <= n_select:
        return list(range(len(candidate_features)))
    
    # 计算协方差逆矩阵
    try:
        cov_inv = np.linalg.inv(target_cov + 1e-6 * np.eye(target_cov.shape[0]))
    except:
        cov_inv = np.eye(target_cov.shape[0])
    
    # 计算每张候选图像的马氏距离
    distances = []
    for i in range(len(candidate_features)):
        diff = candidate_features[i] - target_mean
        dist = np.sqrt(diff @ cov_inv @ diff)
        distances.append(dist)
    
    # 选择距离最近的
    sorted_indices = np.argsort(distances)
    
    return sorted_indices[:n_select].tolist()


def selection_none(candidate_features, n_select):
    """
    不筛选：随机选择
    
    参数:
        candidate_features: 候选图像特征 [M, D]
        n_select: 选择数量
    
    返回:
        selected_indices: 随机选择的索引列表
    """
    indices = np.random.permutation(len(candidate_features))
    return indices[:n_select].tolist()


def selection_density_ratio(candidate_features, original_features, n_select, sigma=3.0, pca_dim=None, k_knn=None):
    """
    PCA降维 + 3σ余弦过滤 + KNN密度比采样
    使选出的合成子集分布趋近原始分布
    
    参数:
        candidate_features: 候选图像特征 [M, D]
        original_features: 原始图像特征 [N, D]
        n_select: 选择数量
        sigma: 余弦过滤的σ倍数
        pca_dim: PCA目标维度（None则自适应）
        k_knn: KNN的k值（None则自适应）
    
    返回:
        selected_indices: 选中的索引列表
        filter_info: 过滤信息字典
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    
    M, D = candidate_features.shape
    N = original_features.shape[0]
    
    # 自适应参数
    if pca_dim is None:
        pca_dim = min(30, N - 1, D)
    if k_knn is None:
        k_knn = min(10, max(3, N // 5))
    
    # ============================================
    # 阶段1: PCA降维
    # ============================================
    all_features = np.concatenate([original_features, candidate_features], axis=0)
    pca = PCA(n_components=pca_dim, random_state=42)
    all_pca = pca.fit_transform(all_features)
    orig_pca = all_pca[:N]
    cand_pca = all_pca[N:]
    
    # ============================================
    # 阶段2: 3σ 余弦相似度过滤（在原始高维空间）
    # ============================================
    center = np.mean(original_features, axis=0)
    
    orig_norm = original_features / (np.linalg.norm(original_features, axis=1, keepdims=True) + 1e-10)
    center_norm = center / (np.linalg.norm(center) + 1e-10)
    orig_cosine = np.dot(orig_norm, center_norm)
    orig_mean = np.mean(orig_cosine)
    orig_std = np.std(orig_cosine)
    
    candidate_norm = candidate_features / (np.linalg.norm(candidate_features, axis=1, keepdims=True) + 1e-10)
    cosine_sim = np.dot(candidate_norm, center_norm)
    
    threshold = orig_mean - sigma * orig_std
    valid_mask = cosine_sim >= threshold
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        threshold = orig_mean - 2 * orig_std
        valid_mask = cosine_sim >= threshold
        valid_indices = np.where(valid_mask)[0]
    
    filter_info = {
        'n_candidates': M,
        'n_valid': len(valid_indices),
        'n_filtered': M - len(valid_indices),
        'filter_ratio': (M - len(valid_indices)) / max(M, 1),
        'threshold': float(threshold),
        'sigma': sigma,
        'orig_mean': float(orig_mean),
        'orig_std': float(orig_std),
        'cosine_sim_min': float(np.min(cosine_sim)),
        'cosine_sim_max': float(np.max(cosine_sim)),
        'cosine_sim_mean': float(np.mean(cosine_sim)),
        'pca_dim': pca_dim,
        'k_knn': k_knn,
        'pca_variance_ratio': float(np.sum(pca.explained_variance_ratio_)),
        'filtered_indices': np.where(~valid_mask)[0].tolist(),
    }
    
    if len(valid_indices) <= n_select:
        # 过滤后数量不足 → 不做KNN，全部通过过滤的保留，剩余随机补齐
        return valid_indices.tolist(), filter_info
    
    # 过滤后数量充足 → KNN密度比采样
    # ============================================
    valid_pca = cand_pca[valid_indices]
    cand_pca_full = cand_pca
    
    # 候选点在原始数据中的密度
    nn_orig = NearestNeighbors(n_neighbors=min(k_knn, N))
    nn_orig.fit(orig_pca)
    dist_orig, _ = nn_orig.kneighbors(valid_pca)
    d_orig = np.mean(dist_orig, axis=1)
    
    # 候选点在候选池中的密度
    nn_synth = NearestNeighbors(n_neighbors=min(k_knn, M))
    nn_synth.fit(cand_pca_full)
    dist_synth, _ = nn_synth.kneighbors(valid_pca)
    d_synth = np.mean(dist_synth, axis=1)
    
    # 密度比权重
    eps = 1e-10
    weights = d_synth / (d_orig + eps)
    
    # 归一化
    weights = weights / np.sum(weights)
    
    # 加权随机采样
    n_to_select = min(n_select, len(valid_indices))
    selected_local = np.random.choice(
        len(valid_indices), size=n_to_select, replace=False, p=weights
    )
    
    filter_info['density_weights_min'] = float(np.min(weights))
    filter_info['density_weights_max'] = float(np.max(weights))
    filter_info['density_weights_std'] = float(np.std(weights))
    
    return [valid_indices[i] for i in selected_local], filter_info


#############################################
# 三、数据集准备工具
#############################################

def identify_tail_classes(class_counts, threshold=None):
    """
    识别尾部类别
    
    参数:
        class_counts: {class_id: count}
        threshold: 阈值，小于此值为尾部类别；None则使用中位数
    
    返回:
        tail_class_ids: 尾部类别ID列表
    """
    counts = list(class_counts.values())
    
    if threshold is None:
        threshold = np.median(counts)
    
    return [cid for cid, count in class_counts.items() if count < threshold]


def prepare_synthetic_dataset(
    original_data_dir,
    generated_data_dir,
    output_dir,
    tail_class_ids,
    quantity_method='weighted',  # 'weighted', 'fixed_per_class', 'fixed_total'
    selection_method='fds',      # 'fds', 'cus', 'dms', 'none'
    n_total=500,
    n_per_class=25,
    target_class_count=None,
    clip_model=None,
    clip_preprocess=None,
    classifier=None,
    device='cuda',
    head_class_ids=None
):
    """
    准备合成数据集
    
    参数:
        original_data_dir: 原始数据集路径
        generated_data_dir: 生成图像路径（每个类别一个子文件夹）
        output_dir: 输出路径（保存新增数据）
        tail_class_ids: 尾部类别ID列表
        quantity_method: 数量分配方法
        selection_method: 选择方法
        n_total: 总生成数量（weighted方法使用）
        n_per_class: 每类固定数量（fixed_per_class方法使用）
        target_class_count: 目标每类总数（fixed_total方法使用）
        clip_model: CLIP模型
        clip_preprocess: CLIP预处理
        classifier: 分类器（CUS方法需要）
        device: 设备
        head_class_ids: 头部类别ID（DMS方法需要）
    
    返回:
        gen_counts: {class_id: 实际生成数量}
    """
    import clip
    import json
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 日志字典
    log_data = {
        'quantity_method': quantity_method,
        'selection_method': selection_method,
        'n_total': n_total,
        'n_per_class': n_per_class,
        'target_class_count': target_class_count,
        'tail_class_ids': tail_class_ids,
        'original_data_counts': {},
        'gqw_weights': {},
        'allocated_counts': {},
        'candidate_counts': {},
        'selected_images': {},
        'selected_counts': {}
    }
    
    # 提取原始数据集特征
    print("  提取原始数据集特征...")
    class_features = {}
    class_counts = {}
    
    for class_id in tail_class_ids:
        class_dir = os.path.join(original_data_dir, str(class_id))
        if not os.path.exists(class_dir):
            print(f"    警告: 原始数据目录不存在 {class_dir}")
            log_data['original_data_counts'][str(class_id)] = 0
            continue
        
        image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(image_paths) == 0:
            print(f"    警告: 原始数据目录为空 {class_dir}")
            log_data['original_data_counts'][str(class_id)] = 0
            continue
        
        features = extract_features_clip(image_paths, clip_model, clip_preprocess, device)
        class_features[class_id] = features
        class_counts[class_id] = len(image_paths)
        log_data['original_data_counts'][str(class_id)] = len(image_paths)
        print(f"    类别 {class_id}: {len(image_paths)} 张原始图像")
    
    # 计算生成数量
    print(f"  计算生成数量 (method={quantity_method})...")
    gen_counts = {}
    
    if quantity_method == 'weighted':
        gqw_weights = compute_gqw_weights(class_features, class_counts)
        gen_counts = allocate_generation_counts(len(tail_class_ids), n_total, gqw_weights)
        log_data['gqw_weights'] = {str(k): round(v, 4) for k, v in gqw_weights.items()}
        print(f"    GQW权重: {log_data['gqw_weights']}")
    
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
    
    log_data['allocated_counts'] = {str(k): v for k, v in gen_counts.items()}
    print(f"    分配数量: {log_data['allocated_counts']}")
    
    # 选择生成图像
    print(f"  选择生成图像 (method={selection_method})...")
    
    # 计算目标分布（DMS需要）
    target_mean = None
    target_cov = None
    if selection_method == 'dms' and head_class_ids:
        head_features = []
        for cid in head_class_ids:
            class_dir = os.path.join(original_data_dir, str(cid))
            if os.path.exists(class_dir):
                image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if len(image_paths) > 0:
                    feats = extract_features_clip(image_paths, clip_model, clip_preprocess, device)
                    head_features.append(feats)
        
        if len(head_features) > 0:
            all_head = np.concatenate(head_features, axis=0)
            target_mean = all_head.mean(axis=0)
            target_cov = np.cov(all_head.T)
    
    # 为每个类别选择图像
    selected_counts = {}
    
    for class_id in tail_class_ids:
        n_needed = gen_counts.get(class_id, 0)
        log_data['selected_images'][str(class_id)] = []
        
        if n_needed <= 0:
            selected_counts[class_id] = 0
            print(f"    类别 {class_id}: 分配数量为0，跳过")
            continue
        
        # 加载候选图像
        gen_class_dir = os.path.join(generated_data_dir, str(class_id))
        if not os.path.exists(gen_class_dir):
            selected_counts[class_id] = 0
            log_data['candidate_counts'][str(class_id)] = 0
            print(f"    类别 {class_id}: 候选目录不存在 {gen_class_dir}")
            continue
        
        candidate_paths = [os.path.join(gen_class_dir, f) for f in os.listdir(gen_class_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        log_data['candidate_counts'][str(class_id)] = len(candidate_paths)
        
        if len(candidate_paths) == 0:
            selected_counts[class_id] = 0
            print(f"    类别 {class_id}: 候选目录为空")
            continue
        
        print(f"    类别 {class_id}: {len(candidate_paths)} 张候选图像, 需要选择 {n_needed} 张")
        
        # 提取候选特征
        candidate_features = extract_features_clip(
            candidate_paths, clip_model, clip_preprocess, device
        )
        
        if len(candidate_features) == 0:
            selected_counts[class_id] = 0
            print(f"    类别 {class_id}: 特征提取失败")
            continue
        
        # 选择
        if selection_method == 'fds':
            selected_indices = selection_fds(
                candidate_features, class_features[class_id], n_needed
            )
        elif selection_method == 'cus':
            selected_indices = selection_cus(
                candidate_features, classifier, n_needed
            )
        elif selection_method == 'dms':
            if target_mean is not None:
                selected_indices = selection_dms(
                    candidate_features, class_features[class_id],
                    target_mean, target_cov, n_needed
                )
            else:
                selected_indices = selection_none(candidate_features, n_needed)
        else:  # none
            selected_indices = selection_none(candidate_features, n_needed)
        
        # 复制选中的图像
        output_class_dir = os.path.join(output_dir, str(class_id))
        os.makedirs(output_class_dir, exist_ok=True)
        
        selected_names = []
        for idx in selected_indices:
            src_path = candidate_paths[idx]
            dst_path = os.path.join(output_class_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            selected_names.append(os.path.basename(src_path))
        
        selected_counts[class_id] = len(selected_indices)
        log_data['selected_images'][str(class_id)] = selected_names
        print(f"    类别 {class_id}: 已选择 {len(selected_indices)} 张")
    
    log_data['selected_counts'] = {str(k): v for k, v in selected_counts.items()}
    
    # 保存日志
    log_path = os.path.join(output_dir, 'selection_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"  日志已保存: {log_path}")
    
    print(f"  合成数据集准备完成: {output_dir}")
    print(f"  各类别生成数量: {selected_counts}")
    
    return selected_counts
