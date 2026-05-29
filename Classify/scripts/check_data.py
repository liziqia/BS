#!/usr/bin/env python3
import os

# 检查数据目录
dirs_to_check = [
    "/root/autodl-tmp/Fusion1/OmniGen/Classify/data",
    "/root/autodl-tmp/Fusion1/OmniGen/Classify/data/original",
    "/root/autodl-tmp/Fusion1/OmniGen/Ship/DataGen",
]

for dir_path in dirs_to_check:
    print(f"\n{'='*60}")
    print(f"检查：{dir_path}")
    print('='*60)
    
    if os.path.exists(dir_path):
        items = os.listdir(dir_path)
        print(f"✅ 存在，包含 {len(items)} 个项目")
        
        # 如果是目录，显示子目录
        subdirs = [d for d in items if os.path.isdir(os.path.join(dir_path, d))]
        if subdirs:
            print(f"   子目录：{sorted(subdirs)[:10]}{'...' if len(subdirs) > 10 else ''}")
        
        # 统计图像文件
        img_count = 0
        for item in items:
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                count = len([f for f in os.listdir(item_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
                img_count += count
        print(f"   图像文件总数：约 {img_count} 张")
    else:
        print(f"❌ 不存在")

print("\n")
