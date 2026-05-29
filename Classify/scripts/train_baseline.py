import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
import numpy as np
import random
import os
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], 'r-', label='Train Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_model(model, dataloaders, criterion, optimizer, num_epochs=50, device='cuda', checkpoint_dir=None, save_every=None, figures_dir=None):
    history = {
        'train_loss': [], 'train_acc': [],
    }
    
    for epoch in range(num_epochs):
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
        
        print(f'train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        if figures_dir:
            plot_training_history(history, os.path.join(figures_dir, 'training_curve.png'))
        
        if checkpoint_dir and save_every and (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'train_acc': epoch_acc.item(),
            }, ckpt_path)
            print(f'  ✅ 保存 checkpoint: {ckpt_path}')
    
    return model, history


def main(args):
    set_seed(args.seed)
    print(f"✅ 随机种子已设置为: {args.seed}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")
    
    checkpoint_dir = args.checkpoint_dir
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
        'train': datasets.ImageFolder(args.data_dir, data_transforms['train']),
    }
    
    num_classes = len(image_datasets['train'].classes)
    
    print(f"✅ 训练数据路径: {args.data_dir}")
    print(f"✅ 训练集类别数: {num_classes}")
    print(f"✅ 训练集样本数: {len(image_datasets['train'])}")
    
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
    
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    figures_dir = '../results/figures/baseline'
    os.makedirs(figures_dir, exist_ok=True)
    
    print("\n 开始训练...")
    model, history = train_model(
        model, dataloaders, criterion, optimizer, 
        num_epochs=args.num_epochs, device=device,
        checkpoint_dir=checkpoint_dir, save_every=args.save_every,
        figures_dir=figures_dir
    )
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'seed': args.seed,
        'experiment': 'baseline'
    }, os.path.join(checkpoint_dir, 'model_final.pth'))
    
    print(f"\n✅ 训练完成！模型已保存到 {checkpoint_dir}/model_final.pth")
    
    config = {
        'seed': args.seed,
        'experiment_name': 'baseline',
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_epochs': args.num_epochs,
        'data_dir': args.data_dir,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基线训练脚本（实验A）')
    parser.add_argument("--data_dir", type=str, default="../data/original", help="训练数据路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--save_every", type=int, default=100, help="每隔多少轮保存一个checkpoint，None则只在最后保存")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints/baseline", help="权重保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--pretrained", type=bool, default=True, help="是否使用预训练权重")
    args = parser.parse_args()
    main(args)
