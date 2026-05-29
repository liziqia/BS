#!/usr/bin/env python3
"""
快速检查数据目录
"""
import os

original_dir = "/root/autodl-tmp/Fusion1/OmniGen/Classify/data/original"
datagen_dir = "/root/autodl-tmp/Fusion1/OmniGen/Ship/DataGen"

print("="*70)
print("数据目录检查")
print("="*70)

# 检查 original
print(f"\n1. Original 数据集：{original_dir}")
if os.path.exists(original_dir):
    classes = sorted([d for d in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, d))])
    print(f"   ✅ 存在，{len(classes)} 个类别")
    total = sum(len([f for f in os.listdir(os.path.join(original_dir, c)) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]) 
               for c in classes)
    print(f"   总图像数：{total}")
else:
    print(f"   ❌ 不存在")

# 检查 DataGen
print(f"\n2. DataGen 数据集：{datagen_dir}")
if os.path.exists(datagen_dir):
    classes = sorted([d for d in os.listdir(datagen_dir) if os.path.isdir(os.path.join(datagen_dir, d))])
    print(f"   ✅ 存在，{len(classes)} 个类别")
    total = sum(len([f for f in os.listdir(os.path.join(datagen_dir, c)) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]) 
               for c in classes)
    print(f"   总图像数：{total}")
else:
    print(f"   ❌ 不存在")

print("\n" + "="*70)
