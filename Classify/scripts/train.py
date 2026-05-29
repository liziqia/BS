import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
import numpy as np
import random
import os
import json
import argparse
import shutil
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import (
    compute_class_weights, create_weighted_loss, oversample_dataset, 
    get_class_labels, DecoupledTrainer, create_balanced_subset, 
    merge_model_weights, LTRLWrapper, create_ot_dynamic_softmax,
    create_ltrl_model
)


EXP_ID_CONFIG = {
    # A. Baseline - 原始训练
    "baseline": {
        "data_dir": "../data/original",
        "checkpoint_dir": "../checkpoints/baseline",
        "method": "none",
    },
    # B1. Oversample - 过采样
    "oversample": {
        "src_dir": "../data/original",
        "data_dir": "../data/oversample",
        "checkpoint_dir": "../checkpoints/oversample",
        "method": "oversample",
        "min_samples": 150,
    },
    # B2. Weighted Loss - 类别平衡加权损失
    "weighted_loss": {
        "data_dir": "../data/original",
        "checkpoint_dir": "../checkpoints/weighted_loss",
        "method": "weighted_loss",
        "beta": 0.99,
    },
    # B3. Decoupling + Regularization - 解耦训练 + 正则化
    "decoupled": {
        "data_dir": "../data/original",
        "checkpoint_dir": "../checkpoints/decoupled",
        "method": "decoupled",
    },
    # B4. LT-Soups - 模型融合
    "ltsoups": {
        "data_dir": "../data/original",
        "checkpoint_dir": "../checkpoints/ltsoups",
        "method": "ltsoups",
        "num_models": 3,
    },
    # B5. LTRL - 反思学习
    "ltrl": {
        "data_dir": "../data/original",
        "checkpoint_dir": "../checkpoints/ltrl",
        "method": "ltrl",
        "lambda_ltrl": 0.1,
    },
    # B6. 最优传输 - OT-dynamic Softmax
    "ot_dynamic": {
        "data_dir": "../data/original",
        "checkpoint_dir": "../checkpoints/ot_dynamic",
        "method": "ot_dynamic",
    },
    # C. OmniGen - 你的方法
    "omnigen": {
        "data_dir": "../data/omnigen",
        "checkpoint_dir": "../checkpoints/omnigen",
        "method": "none",
    },
    # D. Real Data - 真实收集数据（理想上限）
    "real_data": {
        "data_dir": "../data/real_data",
        "checkpoint_dir": None,
        "method": "none",
    },
}


def oversample_dataset(src_dir, dst_dir, min_samples=150):
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)
    
    if os.path.exists(dst_dir):
        print(f"⚠️  过采样目录已存在，跳过: {dst_dir}")
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
    
    print(f"\n📊 过采样后总计: {total_dst} 张")
    print(f"✅ 过采样完成，已保存到: {dst_dir}\n")
    return dst_dir


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_model(num_classes, use_pretrained=True, weights_dir='../pretrained_weights'):
    if use_pretrained:
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, 'resnet50-0676ba61.pth')
        if not os.path.exists(weights_path):
            cache_path = os.path.expanduser('~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')
            if os.path.exists(cache_path):
                print(f"📋 从缓存复制权重到 {weights_path}...")
                import shutil
                shutil.copy(cache_path, weights_path)
            else:
                print(f"⬇️  下载预训练权重到 {weights_path}...")
                torch.hub.download_url_to_file(
                    'https://download.pytorch.org/models/resnet50-0676ba61.pth',
                    weights_path
                )
        model = models.resnet50(weights=None)
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"✅ 使用预训练的 ResNet50 权重（来自 {weights_path}）")
    else:
        model = models.resnet50(weights=None)
        print("⚠️  使用随机初始化的 ResNet50")
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model


def plot_training_history(history, save_path, start_epoch=0):
    """
    绘制训练历史曲线
    
    参数：
        history: 包含 train_loss, train_acc 等的字典
        save_path: 保存路径
        start_epoch: 起始 epoch（用于断点续训时正确显示 epoch 编号）
    """
    plt.figure(figsize=(12, 5))
    
    # 创建正确的 epoch 轴
    num_epochs = len(history['train_loss'])
    epoch_axis = np.arange(start_epoch + 1, start_epoch + num_epochs + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(epoch_axis, history['train_loss'], 'b-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epoch_axis, history['train_acc'], 'r-', label='Train Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_model(model, dataloaders, criterion, optimizer, num_epochs=50, device='cuda', 
                checkpoint_dir=None, save_every=None, figures_dir=None, 
                resume_checkpoint=None, start_epoch=0, resume_history=None):
    """
    训练模型
    
    参数：
        model: 模型
        dataloaders: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 总 epoch 数
        device: 设备
        checkpoint_dir: checkpoint 保存目录
        save_every: 每隔多少 epoch 保存一次
        figures_dir: 图表保存目录
        resume_checkpoint: 从中断的 checkpoint 路径（用于断点续训）
        start_epoch: 起始 epoch（用于断点续训）
        resume_history: 之前的历史（用于断点续训）
    """
    # 记录原始起点（用于绘图）
    original_start_epoch = start_epoch
    
    # 如果是断点续训，加载 checkpoint
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n📌 从 checkpoint 恢复训练：{resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        
        # 加载历史
        if resume_history:
            history = resume_history
            print(f"✅ 已加载历史训练数据，从 epoch {start_epoch} 继续")
        else:
            history = {
                'train_loss': [], 'train_acc': [], 'epoch_time': [],
            }
            print(f"✅ 从 epoch {start_epoch} 继续训练（无历史数据）")
    else:
        history = {
            'train_loss': [], 'train_acc': [], 'epoch_time': [],
        }
        start_epoch = 0
    
    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        model.train()
        
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        epoch_time = time.time() - epoch_start_time
        history['epoch_time'].append(epoch_time)
        
        print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {epoch_time:.2f}s')
        
        if figures_dir:
            # 使用原始起点画图（断点续训时会从 0 开始显示完整历史）
            plot_training_history(history, os.path.join(figures_dir, 'training_curve.png'), start_epoch=original_start_epoch)
        
        if checkpoint_dir and save_every and (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'train_acc': epoch_acc.item(),
                'history': history,
            }, ckpt_path)
            print(f'  ✅ 保存 checkpoint: {ckpt_path}')
    
    return model, history


def main(args):
    set_seed(args.seed)
    print(f"✅ 实验ID: {args.exp_id}")
    print(f"✅ 随机种子已设置为: {args.seed}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")
    
    config = EXP_ID_CONFIG.get(args.exp_id, {})
    data_dir = args.data_dir if args.data_dir else config.get("data_dir", f"../data/{args.exp_id}")
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else config.get("checkpoint_dir", f"../checkpoints/{args.exp_id}")
    
    method = config.get("method", "none")
    if method == "oversample":
        src_dir = config.get("src_dir", "../data/original")
        min_samples = config.get("min_samples", 150)
        data_dir = oversample_dataset(src_dir, data_dir, min_samples)
    
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('../results/logs', exist_ok=True)
    os.makedirs('../results/figures', exist_ok=True)
    os.makedirs('../results/metrics', exist_ok=True)
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    image_datasets = {
        'train': datasets.ImageFolder(data_dir, data_transforms['train']),
    }
    
    num_classes = len(image_datasets['train'].classes)
    
    print(f"✅ 训练数据路径：{data_dir}")
    print(f"✅ 训练集类别数：{num_classes}")
    print(f"✅ 训练集样本数：{len(image_datasets['train'])}")
    
    def seed_worker(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'], 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g
        )
    }
    
    model = create_model(num_classes=num_classes, use_pretrained=args.pretrained)
    model = model.to(device)
    
    figures_dir = f'../results/figures/{args.exp_id}'
    os.makedirs(figures_dir, exist_ok=True)
    
    if method == "decoupled":
        # B3. 解耦训练 + 正则化（只训 fc 层，与其他方法一致）
        stage1_epochs = args.num_epochs // 2
        stage2_epochs = args.num_epochs - stage1_epochs
        
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        
        trainer = DecoupledTrainer(model, device, num_classes)
        
        optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        history_stage1 = trainer.train_representation(
            dataloaders['train'], optimizer, criterion, 
            num_epochs=stage1_epochs, figures_dir=figures_dir
        )
        
        history_stage2 = trainer.train_classifier(
            dataloaders['train'], num_epochs=stage2_epochs, 
            method='balanced', figures_dir=figures_dir
        )
        
        history = {
            'stage1_loss': history_stage1['train_loss'],
            'stage1_acc': history_stage1['train_acc'],
            'stage2_loss': history_stage2['train_loss'],
            'stage2_acc': history_stage2['train_acc'],
        }
        
        print(f"\n✅ 解耦训练完成 (阶段 1: {stage1_epochs}轮，阶段 2: {stage2_epochs}轮)")
    
    elif method == "ltsoups":
        # B4. LT-Soups - 模型融合 (NeurIPS 2025)
        # 两阶段训练：
        # 阶段 1：创建多个不同不平衡比例的子集，训练模型并融合
        # 阶段 2：在完整数据集上微调分类器
        
        num_models = config.get("num_models", 5)
        
        print(f"\n📌 LT-Soups: 两阶段训练")
        print(f"  阶段 1: 训练 {num_models} 个模型并融合")
        print(f"  阶段 2: 在完整数据集上微调分类器")
        
        # 阶段 1：训练多个模型并融合
        models = []
        subsets = create_lt_soups_subsets(image_datasets['train'], num_models, num_classes)
        
        for i, subset in enumerate(subsets):
            print(f"\n  训练第 {i+1}/{num_models} 个模型 (不平衡比例=2^{i})...")
            set_seed(args.seed + i)  # 不同种子
            
            subset_loader = torch.utils.data.DataLoader(
                subset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers
            )
            
            sub_model = create_model(num_classes=num_classes, use_pretrained=args.pretrained)
            sub_model = sub_model.to(device)
            
            # 只训练分类器
            for param in sub_model.parameters():
                param.requires_grad = False
            for param in sub_model.fc.parameters():
                param.requires_grad = True
            
            optimizer = optim.SGD(sub_model.fc.parameters(), lr=args.lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            
            sub_model, _ = train_model(
                sub_model, {'train': subset_loader}, criterion, optimizer,
                num_epochs=args.num_epochs // num_models, device=device,
                checkpoint_dir=None, figures_dir=None
            )
            
            models.append(sub_model)
        
        # 融合权重
        print(f"\n  融合 {num_models} 个模型的权重...")
        merged_weights = merge_model_weights(models)
        model.load_state_dict(merged_weights)
        
        # 阶段 2：在完整数据集上微调分类器
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        
        optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        print(f"\n  阶段 2: 在完整数据集上微调分类器...")
        model, history = train_model(
            model, dataloaders, criterion, optimizer,
            num_epochs=args.num_epochs // 3, device=device,
            checkpoint_dir=checkpoint_dir, save_every=args.save_every,
            figures_dir=figures_dir
        )
        
        print(f"\n✅ LT-Soups 完成 (融合 {num_models} 个模型 + 阶段 2 微调)")
    
    elif method == "ltrl":
        # B5. LTRL - 反思学习（简化版：只训 fc 层 + Knowledge Review）
        lambda_ltrl = config.get("lambda_ltrl", 0.1)
        
        print(f"\n📌 LTRL: 反思学习 (lambda={lambda_ltrl})")
        
        ltrl_model, criterion = create_ltrl_model(
            model, num_classes, device, lambda_ltrl
        )
        
        optimizer = optim.SGD(ltrl_model.model.fc.parameters(), lr=args.lr, momentum=0.9)
        
        model, history = train_model(
            ltrl_model, dataloaders, criterion, optimizer,
            num_epochs=args.num_epochs, device=device,
            checkpoint_dir=checkpoint_dir, save_every=args.save_every,
            figures_dir=figures_dir
        )
        
        model = ltrl_model.model
        print(f"\n✅ LTRL 反思学习完成")
    
    elif method == "ot_dynamic":
        # B6. 最优传输 - OT-dynamic Softmax
        labels = get_class_labels(image_datasets['train'])
        class_counts = np.bincount(labels)
        
        criterion = create_ot_dynamic_softmax(num_classes, device, class_counts)
        
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        
        optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
        
        print(f"\n📌 最优传输：使用 OT-dynamic Softmax")
        
        model, history = train_model(
            model, dataloaders, criterion, optimizer,
            num_epochs=args.num_epochs, device=device,
            checkpoint_dir=checkpoint_dir, save_every=args.save_every,
            figures_dir=figures_dir
        )
        
        print(f"\n✅ 最优传输训练完成")
    
    else:
        # A/B1/B2/C/D: 标准训练
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        
        optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
        
        if method == "weighted_loss":
            labels = get_class_labels(image_datasets['train'])
            class_weights = compute_class_weights(labels, beta=config.get("beta", 0.99))
            criterion = create_weighted_loss(class_weights, device)
            print(f"✅ 使用加权损失函数 (beta={config.get('beta', 0.99)})")
        else:
            criterion = nn.CrossEntropyLoss()
        
        # 断点续训逻辑
        resume_checkpoint = None
        resume_history = None
        if args.resume:
            if os.path.exists(args.resume):
                resume_checkpoint = args.resume
                # 尝试加载历史
                history_path = os.path.join(checkpoint_dir, 'training_history.json')
                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        resume_history = json.load(f)
                    print(f"✅ 已加载训练历史：{history_path}")
            else:
                print(f"⚠️  指定的 checkpoint 不存在：{args.resume}，将从头开始训练")
        
        print(f"\n 开始训练实验：{args.exp_id}")
        model, history = train_model(
            model, dataloaders, criterion, optimizer,
            num_epochs=args.num_epochs, device=device,
            checkpoint_dir=checkpoint_dir, save_every=args.save_every,
            figures_dir=figures_dir,
            resume_checkpoint=resume_checkpoint,
            resume_history=resume_history
        )
    
    if checkpoint_dir:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'seed': args.seed,
            'experiment': args.exp_id
        }, os.path.join(checkpoint_dir, 'model_final.pth'))
        
        print(f"\n✅ 训练完成！模型已保存到 {checkpoint_dir}/model_final.pth")
        
        config = {
            'seed': args.seed,
            'experiment_name': args.exp_id,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'num_epochs': args.num_epochs,
            'data_dir': data_dir,
            'num_classes': num_classes,
            'model': 'ResNet50',
            'pretrained': args.pretrained,
        }
        
        with open(os.path.join(checkpoint_dir, 'train_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ 训练配置已保存到 {checkpoint_dir}/train_config.json")
        
        with open(os.path.join(checkpoint_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"✅ 训练历史已保存到 {checkpoint_dir}/training_history.json")
    else:
        print(f"\n✅ 训练完成！未保存权重（checkpoint_dir=None）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='增强数据训练脚本（实验 B/C/D）')
    parser.add_argument("--exp_id", type=str, default="ot_dynamic", help="实验 ID：baseline(A) / oversample(B1) / weighted_loss(B2) / decoupled(B3) / ltsoups(B4) / ltrl(B5) / ot_dynamic(B6) / omnigen(C) / real_data(D)")
    parser.add_argument("--data_dir", type=str, default=None, help="训练数据路径，默认根据 exp_id 自动推断")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.002, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=250, help="训练轮数")
    parser.add_argument("--save_every", type=int, default=50, help="每隔多少轮保存一个 checkpoint，None 则只在最后保存")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="权重保存路径，默认根据 exp_id 自动推断（D 实验默认不保存）")
    parser.add_argument("--resume", type=str, default=None, help="从中断的 checkpoint 路径恢复训练（断点续训）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--pretrained", type=bool, default=True, help="是否使用预训练权重")
    args = parser.parse_args()
    main(args)
