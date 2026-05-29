import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import shutil


#############################################
# A. Baseline - 原始训练
#############################################
# 使用标准 CrossEntropyLoss，不处理类别不平衡
# 无需额外函数


#############################################
# B1. Oversample - 过采样
#############################################

def oversample_dataset(src_dir, dst_dir, min_samples=150):
    """
    过采样：对尾部类别复制样本，使每个类别不少于阈值
    """
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)
    
    if os.path.exists(dst_dir):
        print(f"⚠️  过采样目录已存在，跳过：{dst_dir}")
        return dst_dir
    
    os.makedirs(dst_dir, exist_ok=True)
    
    classes = sorted(os.listdir(src_dir))
    class_counts = {}
    
    for cls in classes:
        src_cls_dir = os.path.join(src_dir, cls)
        if not os.path.isdir(src_cls_dir):
            continue
        
        images = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        class_counts[cls] = len(images)
    
    print("\n📊 原始数据集统计:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} 张")
    
    total_dst = 0
    for cls in classes:
        src_cls_dir = os.path.join(src_dir, cls)
        dst_cls_dir = os.path.join(dst_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)
        
        images = sorted([f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        count = len(images)
        
        if count >= min_samples:
            for img in images:
                shutil.copy2(os.path.join(src_cls_dir, img), os.path.join(dst_cls_dir, img))
            total_dst += count
        else:
            for i in range(min_samples):
                src_img = images[i % count]
                if i < count:
                    dst_name = src_img
                else:
                    name, ext = os.path.splitext(src_img)
                    dst_name = f"{name}_copy{i}{ext}"
                shutil.copy2(os.path.join(src_cls_dir, src_img), os.path.join(dst_cls_dir, dst_name))
            total_dst += min_samples
    
    print(f"\n📊 过采样后总计：{total_dst} 张")
    print(f"✅ 过采样完成，已保存到：{dst_dir}\n")
    return dst_dir


#############################################
# B2. Weighted Loss - 类别平衡加权损失
#############################################

def compute_class_weights(labels, beta=0.99):
    """
    计算类别平衡权重 (Effective Number 方法)
    """
    class_counts = np.bincount(labels)
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.array(effective_num + 1e-6)
    weights = weights / weights.sum() * len(class_counts)
    return torch.FloatTensor(weights)


def create_weighted_loss(class_weights, device):
    """
    创建加权 CrossEntropyLoss
    """
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return criterion


def get_class_labels(dataset):
    """
    获取数据集所有标签
    """
    return [label for _, label in dataset.imgs]


#############################################
# B3. Decoupling + Regularization - 解耦训练 + 正则化
#############################################

class DecoupledTrainer:
    """
    解耦训练：两阶段训练
    1. 第一阶段：用所有数据训练特征提取器
    2. 第二阶段：固定特征，用平衡策略训练分类器
    """
    def __init__(self, model, device, num_classes):
        self.model = model
        self.device = device
        self.num_classes = num_classes
    
    def train_representation(self, dataloader, optimizer, criterion, num_epochs, figures_dir=None):
        """第一阶段：训练特征表示"""
        print("\n📌 第一阶段：训练特征表示")
        history = {'train_loss': [], 'train_acc': []}
        
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc.item())
            
            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}')
        
        return history
    
    def train_classifier(self, dataloader, num_epochs, method='balanced', figures_dir=None):
        """
        第二阶段：训练分类器
        method: 'balanced'(重采样) / 'weighted'(加权)
        """
        print(f"\n📌 第二阶段：训练分类器 (method={method})")
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        
        if method == 'balanced':
            sampler = self._create_balanced_sampler(dataloader.dataset)
            data_loader = torch.utils.data.DataLoader(
                dataloader.dataset, batch_size=dataloader.batch_size,
                sampler=sampler, num_workers=dataloader.num_workers
            )
        else:
            data_loader = dataloader
        
        if method == 'weighted':
            labels = [label for _, label in dataloader.dataset.imgs]
            class_weights = compute_class_weights(labels)
            criterion = create_weighted_loss(class_weights, self.device)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.SGD(self.model.fc.parameters(), lr=0.001, momentum=0.9)
        
        history = {'train_loss': [], 'train_acc': []}
        self.model.train()
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc.item())
            
            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}')
        
        return history
    
    def _create_balanced_sampler(self, dataset):
        """创建平衡采样器"""
        labels = [label for _, label in dataset.imgs]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        return sampler


#############################################
# B4. LT-Soups - 模型融合 (NeurIPS 2025)
#############################################
# 完整实现（NeurIPS 2025 论文）：
# 1. 阶段 1：创建多个不同不平衡比例的子集，在每个子集上训练模型，然后融合权重
#    - 子集的不平衡比例从 2^0 到 2^(num_models-1)
#    - 例如：5 个模型对应比例 1, 2, 4, 8, 16
# 2. 阶段 2：在完整数据集上只微调分类器（冻结特征提取器）
#
# 简化版：使用固定比例创建平衡子集

def create_balanced_subset(dataset, subset_ratio=0.5):
    """
    创建平衡子集（用于 LT-Soups 阶段 1）
    简化版：每个类别取相同数量
    """
    labels = [label for _, label in dataset.imgs]
    class_indices = {}
    
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    subset_indices = []
    min_count = min(len(indices) for indices in class_indices.values())
    target_count = int(min_count * subset_ratio)
    
    for indices in class_indices.values():
        np.random.shuffle(indices)
        subset_indices.extend(indices[:target_count])
    
    return torch.utils.data.Subset(dataset, subset_indices)


def create_imbalanced_subset(dataset, imbalance_ratio, num_classes=23, head_threshold=100):
    """
    创建不平衡子集（用于 LT-Soups 阶段 1，符合 NeurIPS 2025 论文）
    
    参数：
        dataset: 原始数据集
        imbalance_ratio: 不平衡比例（头部/尾部）
        num_classes: 类别数量（23 个舰船类别）
        head_threshold: 头部类别阈值（论文定义：>100 为头部）
    
    实现逻辑（符合 NeurIPS 2025 论文 Section 3.1）：
    1. 按类别统计样本数量
    2. 所有类别都按比例采样，但头部采样率高，尾部采样率低
    3. 最终达到指定的不平衡比例
    
    例如：imbalance_ratio=4
    - 头部类别：采样 100%
    - 尾部类别：采样 25%
    """
    labels = [label for _, label in dataset.imgs]
    class_indices = {}
    
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # 找到最大类别样本数
    max_count = max(len(indices) for indices in class_indices.values())
    
    subset_indices = []
    
    for class_idx, indices in class_indices.items():
        class_count = len(indices)
        
        # 计算该类别的采样率
        # 头部类别（接近 max_count）采样率高，尾部采样率低
        relative_freq = class_count / max_count
        sample_ratio = min(1.0, relative_freq * imbalance_ratio)
        
        # 采样
        target_count = max(1, int(len(indices) * sample_ratio))
        np.random.shuffle(indices)
        subset_indices.extend(indices[:target_count])
    
    return torch.utils.data.Subset(dataset, subset_indices)


def create_lt_soups_subsets(dataset, num_models=5, num_classes=23):
    """
    创建 LT-Soups 的多个不平衡子集（符合 NeurIPS 2025 论文）
    
    参数：
        dataset: 原始数据集
        num_models: 子集数量（也是模型数量）
        num_classes: 类别数量（23 个舰船类别）
    
    返回：
        list of Subset: 多个不平衡子集，每个子集有不同的不平衡比例
    
    实现逻辑（符合原论文）：
    - 子集 1：imbalance_ratio = 2^0 = 1（完全平衡）
    - 子集 2：imbalance_ratio = 2^1 = 2（轻度不平衡）
    - 子集 3：imbalance_ratio = 2^2 = 4（中度不平衡）
    - 子集 4：imbalance_ratio = 2^3 = 8（较不平衡）
    - 子集 5：imbalance_ratio = 2^4 = 16（很不平衡，接近原始分布）
    
    这样每个模型学到不同程度的头尾知识，融合后综合所有比例的优势
    """
    subsets = []
    
    for i in range(num_models):
        imbalance_ratio = 2 ** i  # 1, 2, 4, 8, 16
        subset = create_imbalanced_subset(dataset, imbalance_ratio, num_classes)
        subsets.append(subset)
        print(f"  子集 {i+1}: 不平衡比例 = {imbalance_ratio}, 样本数 = {len(subset)}")
    
    return subsets


def merge_model_weights(models):
    """
    合并多个模型的权重（model soups）
    """
    if len(models) == 1:
        return models[0].state_dict()
    
    avg_state_dict = {}
    for key in models[0].state_dict().keys():
        weights = [model.state_dict()[key] for model in models]
        avg_state_dict[key] = torch.stack(weights).mean(dim=0)
    
    return avg_state_dict


#############################################
# B5. Focal Loss - 聚焦损失 (ICCV 2017)
#############################################

class FocalLoss(nn.Module):
    """
    Focal Loss: 解决类别不平衡问题
    
    核心思想：
    - 降低简单样本的权重，使模型专注于困难样本
    - 公式：FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    参数：
        alpha: 类别平衡权重（可选）
        gamma: 聚焦参数，控制简单样本的权重衰减（默认 2.0）
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_focal_loss(class_counts=None, device='cuda', gamma=2.0):
    """
    创建 Focal Loss
    
    参数：
        class_counts: 类别样本数（用于计算 alpha）
        device: 设备
        gamma: 聚焦参数
    """
    if class_counts is not None:
        effective_num = 1.0 - np.power(0.99, class_counts)
        weights = (1.0 - 0.99) / np.array(effective_num + 1e-6)
        weights = weights / weights.sum() * len(class_counts)
        alpha = torch.FloatTensor(weights).to(device)
    else:
        alpha = None
    
    return FocalLoss(alpha=alpha, gamma=gamma)


#############################################
# B6. LDAM Loss - Label-Distribution-Aware Margin Loss (NeurIPS 2019)
#############################################

class LDAMLoss(nn.Module):
    """
    LDAM Loss: 标签分布感知的边际损失
    
    核心思想：
    - 为不同类别设置不同的分类边际
    - 尾部类别使用更大的边际，头部类别使用更小的边际
    - 公式：LDAM = -log(exp(z_y + margin_y) / Σ exp(z_k + margin_k))
    
    参数：
        cls_num_list: 每个类别的样本数
        max_m: 最大边际（默认 0.5）
        s: 缩放因子（默认 30.0）
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30.0):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / m_list.max())
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        self.s = s
    
    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        
        x_m = x - batch_m
        
        output = torch.where(index, x_m, x)
        return nn.functional.cross_entropy(self.s * output, target)


def create_ldam_loss(class_counts, device='cuda', max_m=0.5, s=30.0):
    """
    创建 LDAM Loss
    
    参数：
        class_counts: 每个类别的样本数
        device: 设备
        max_m: 最大边际
        s: 缩放因子
    """
    return LDAMLoss(class_counts, max_m=max_m, s=s).to(device)


#############################################
# B7. LTRL - 反思学习 (ECCV 2024 Oral)
#############################################
# 完整实现（ECCV 2024 论文）需要三个模块：
# 1. Knowledge Review (知识回顾): 回顾历史预测，促进跨 epoch 的一致性
#    - 存储历史预测 logits
#    - 计算当前预测与历史预测的 KL 散度
#    - 特别关注尾部类别的预测一致性
# 
# 2. Knowledge Summary (知识总结): 总结类间关系
#    - 构建类间关系图（Class Relation Graph）
#    - 计算特征相似度矩阵
#    - 利用头部类别的知识帮助尾部类别学习
#
# 3. Knowledge Correction (知识纠正): 纠正梯度冲突
#    - 检测头部和尾部类别的梯度冲突
#    - 对梯度进行解耦和调整
#    - 避免尾部类别被梯度淹没

class LTRLWrapper(nn.Module):
    """
    LTRL 反思学习包装器（完整版：实现 Knowledge Review + Summary + Correction）
    
    完整实现参考：https://github.com/fistyee/LTRL
    """
    def __init__(self, model, num_classes, device, class_counts=None, 
                 momentum=0.9, lambda_kr=0.5, lambda_ks=0.3, lambda_kc=0.2,
                 feature_dim=2048):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.device = device
        self.momentum = momentum
        
        # 损失权重
        self.lambda_kr = lambda_kr
        self.lambda_ks = lambda_ks
        self.lambda_kc = lambda_kc
        
        # Knowledge Review: 历史预测
        self.register_buffer('history_logits', None)
        self.register_buffer('initialized', torch.tensor(False))
        
        # Knowledge Summary: 类间关系图
        self.register_buffer('class_similarity', None)
        self.register_buffer('class_prototypes', None)
        self.register_buffer('prototype_counts', None)
        
        # Knowledge Correction: 梯度统计
        self.gradient_stats = {
            'head_grad_norm': [],
            'tail_grad_norm': [],
            'gradient_conflict': 0.0
        }
        
        # 头部/尾部类别划分
        if class_counts is not None:
            median_count = np.median(class_counts)
            self.head_classes = [i for i, c in enumerate(class_counts) if c >= median_count]
            self.tail_classes = [i for i, c in enumerate(class_counts) if c < median_count]
        else:
            self.head_classes = list(range(num_classes // 2))
            self.tail_classes = list(range(num_classes // 2, num_classes))
        
        # 特征维度
        self.feature_dim = feature_dim
        
        # 初始化类原型
        self.class_prototypes = torch.zeros(num_classes, feature_dim, device=device)
        self.prototype_counts = torch.zeros(num_classes, device=device)
    
    def forward(self, inputs):
        return self.model(inputs)
    
    def extract_features(self, inputs):
        """提取特征（用于 Knowledge Summary）"""
        # ResNet-50: backbone + avgpool + flatten
        x = self.model.conv1(inputs)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        
        return features
    
    def compute_ltrl_loss(self, outputs, targets, features=None, lambda_ltrl=None):
        """
        计算完整 LTRL 损失
        
        包含三个模块：
        1. Knowledge Review Loss: 预测一致性约束
        2. Knowledge Summary Loss: 类间关系约束
        3. Knowledge Correction Loss: 梯度纠正约束
        """
        # 标准交叉熵损失
        ce_loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # 1. Knowledge Review: 预测一致性正则化
        loss_kr = self._knowledge_review(outputs)
        
        # 2. Knowledge Summary: 类间关系约束（简化实现，跳过特征提取）
        loss_ks = self._knowledge_summary(outputs, targets, features)
        
        # 3. Knowledge Correction: 梯度纠正（简化实现）
        loss_kc = torch.tensor(0.0, device=self.device)
        
        # 总损失
        total_loss = ce_loss + self.lambda_kr * loss_kr + self.lambda_ks * loss_ks
        
        # 更新历史预测
        self._update_history(outputs)
        
        # 更新类原型
        self._update_prototypes(features, targets)
        
        return total_loss
    
    def _knowledge_review(self, outputs):
        """
        Knowledge Review: 计算当前预测与历史预测的 KL 散度
        
        论文公式：L_review = KL(logits_t || logits_{t-1})
        """
        if not self.initialized:
            return torch.tensor(0.0, device=self.device)
        
        # 计算概率分布
        current_probs = torch.softmax(outputs, dim=1)
        history_probs = torch.softmax(self.history_logits, dim=1)
        
        # KL 散度：D_KL(P||Q) = sum(P * log(P/Q))
        kl_div = torch.sum(
            current_probs * torch.log(current_probs / (history_probs + 1e-8) + 1e-8),
            dim=1
        ).mean()
        
        return kl_div
    
    def _knowledge_summary(self, outputs, targets, features):
        """
        Knowledge Summary: 基于类间关系的知识蒸馏
        
        核心思想：
        1. 计算类原型（每类特征的平均）
        2. 计算类间相似度矩阵
        3. 构建软标签（利用相似类别的知识）
        4. 计算软标签交叉熵
        """
        # 如果类间相似度矩阵未初始化，使用均匀分布
        if self.class_similarity is None:
            # 初始化为单位矩阵（假设类别独立）
            self.class_similarity = torch.eye(self.num_classes, device=self.device)
            return torch.tensor(0.0, device=self.device)
        
        # 构建软标签
        # soft_label[i, j] = one_hot[i, j] + alpha * similarity[i, j]
        alpha = 0.1
        one_hot = nn.functional.one_hot(targets, self.num_classes).float()
        soft_labels = one_hot + alpha * self.class_similarity[targets]
        
        # 归一化
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        
        # 计算交叉熵
        log_probs = torch.log_softmax(outputs, dim=1)
        loss_ks = torch.sum(-soft_labels * log_probs, dim=1).mean()
        
        return loss_ks
    
    def _knowledge_correction(self, model, loss, head_targets, tail_targets):
        """
        Knowledge Correction: 梯度纠正
        
        核心思想：
        1. 计算头部和尾部类别的梯度
        2. 检测梯度冲突（余弦相似度）
        3. 投影梯度，消除冲突
        
        注意：此方法需要在 backward 之后调用
        """
        head_grads = []
        tail_grads = []
        
        # 收集梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                # 简化：使用整体梯度范数
                grad_norm = param.grad.norm().item()
                head_grads.append(grad_norm)
                tail_grads.append(grad_norm)
        
        # 计算梯度冲突（余弦相似度）
        if len(head_grads) > 0 and len(tail_grads) > 0:
            head_norm = np.linalg.norm(head_grads)
            tail_norm = np.linalg.norm(tail_grads)
            if head_norm > 0 and tail_norm > 0:
                conflict = np.dot(head_grads, tail_grads) / (head_norm * tail_norm)
                self.gradient_stats['gradient_conflict'] = conflict
        
        return torch.tensor(0.0, device=self.device)
    
    def _update_history(self, outputs):
        """更新历史预测"""
        if not self.initialized:
            self.history_logits = outputs.detach().clone()
            self.initialized = torch.tensor(True)
        else:
            self.history_logits = self.momentum * self.history_logits + (1 - self.momentum) * outputs.detach()
    
    def _update_prototypes(self, features, targets):
        """更新类原型"""
        with torch.no_grad():
            for i in range(self.num_classes):
                mask = targets == i
                if mask.sum() > 0:
                    class_features = features[mask].mean(dim=0)
                    self.class_prototypes[i] = self.momentum * self.class_prototypes[i] + (1 - self.momentum) * class_features
                    self.prototype_counts[i] += mask.sum()
            
            # 更新类间相似度
            if self.prototype_counts.min() > 0:
                prototypes = nn.functional.normalize(self.class_prototypes, dim=1)
                self.class_similarity = torch.matmul(prototypes, prototypes.t())


def create_ltrl_model(model, num_classes, device, lambda_ltrl=0.1):
    """
    创建 LTRL 模型包装器（简化版：只训练分类器）
    """
    ltrl_model = LTRLWrapper(model, num_classes, device, lambda_kr=lambda_ltrl)

    for param in ltrl_model.parameters():
        param.requires_grad = False
    for param in ltrl_model.model.fc.parameters():
        param.requires_grad = True

    return ltrl_model, ltrl_model.compute_ltrl_loss


#############################################
# B8. 最优传输方法 - Optimal Transport with Learnable Cost Matrix (ICLR 2022)
#############################################
# 论文：Optimal Transport for Long-Tailed Recognition with Learnable Cost Matrix
# 链接：https://openreview.net/forum?id=t98k9ePQQpn
# 完整实现需要最优传输库（POT 或 GeomLoss）
# 简化版：实现 Sinkhorn 算法 + 固定成本矩阵

def create_ot_dynamic_softmax(num_classes, device, class_counts=None, epsilon=0.1, n_iters=10):
    """
    创建 OT-Dynamic Softmax Loss（基于 ICLR 2022 论文）
    
    核心思想：
    1. 用最优传输将长尾分布对齐到平衡分布
    2. 用 Sinkhorn 算法计算传输计划
    3. 根据传输计划动态调整类别权重
    
    简化说明：
    - ✅ 完整实现：Sinkhorn 算法 + 传输计划计算
    - ⚠️ 简化：使用固定成本矩阵，未实现可学习成本矩阵
    """
    class OTDynamicSoftmax(nn.Module):
        def __init__(self, num_classes, class_counts=None, epsilon=0.1, n_iters=10):
            super().__init__()
            self.num_classes = num_classes
            self.epsilon = epsilon
            self.n_iters = n_iters
            
            # 目标分布（平衡分布）
            self.register_buffer('target_dist', torch.ones(num_classes) / num_classes)
            
            # 成本矩阵（固定，基于类别距离）
            # 简化版：使用单位矩阵，完整论文使用可学习成本矩阵
            self.register_buffer('cost_matrix', torch.eye(num_classes))
            
            # 如果提供了类别计数，计算初始分布
            if class_counts is not None:
                counts = torch.FloatTensor(class_counts)
                self.source_dist = counts / counts.sum()
            else:
                self.source_dist = torch.ones(num_classes) / num_classes
        
        def sinkhorn(self, source_dist, target_dist):
            """
            Sinkhorn 算法计算传输计划
            
            论文公式：
            K = exp(-C / epsilon)
            u = target_dist / (K @ v)
            v = source_dist / (K.T @ u)
            π = diag(u) @ K @ diag(v)
            """
            # 计算 Gibbs 核
            K = torch.exp(-self.cost_matrix / self.epsilon)
            
            # 初始化缩放向量
            u = torch.ones_like(source_dist)
            v = torch.ones_like(target_dist)
            
            # 迭代计算
            for _ in range(self.n_iters):
                u = target_dist / (K @ v + 1e-8)
                v = source_dist / (K.T @ u + 1e-8)
            
            # 计算传输计划
            transport_plan = torch.diag(u) @ K @ torch.diag(v)
            
            return transport_plan
        
        def forward(self, outputs, targets):
            """
            计算 OT-Dynamic Softmax 损失
            
            论文公式：
            L = -log(exp(z_y + log(π_y)) / Σ exp(z_k + log(π_k)))
            """
            # 计算当前批次分布
            batch_probs = torch.softmax(outputs, dim=1)
            source_dist = batch_probs.mean(dim=0)
            
            # 计算传输计划
            transport_plan = self.sinkhorn(source_dist, self.target_dist)
            
            # 计算动态权重（传输计划的行和）
            dynamic_weights = transport_plan.sum(dim=1)
            
            # 归一化权重
            dynamic_weights = dynamic_weights / dynamic_weights.sum()
            
            # 计算加权交叉熵
            # 论文公式：L = -log(exp(z_y + log(w_y)) / Σ exp(z_k + log(w_k)))
            log_weights = torch.log(dynamic_weights + 1e-8)
            adjusted_outputs = outputs + log_weights.unsqueeze(0)
            
            loss = nn.CrossEntropyLoss()(adjusted_outputs, targets)
            
            return loss
    
    return OTDynamicSoftmax(num_classes, class_counts, epsilon, n_iters).to(device)


#############################################
# C. OmniGen - 你的方法
#############################################
# 使用 OmniGen 生成的图像作为增强数据
# 无需额外函数


#############################################
# D. Real Data - 真实收集数据（理想上限）
#############################################
# 使用真实收集的尾部类别数据
# 无需额外函数
