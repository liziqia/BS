#!/usr/bin/env python3
"""
执行数据合并：Original + DataGen -> OmniGen
"""
import os
import shutil

# 路径配置
ORIGINAL_DIR = "/root/autodl-tmp/Fusion1/OmniGen/Classify/data/original"
DATAGEN_DIR = "/root/autodl-tmp/Fusion1/OmniGen/Ship/DataGen"
OUTPUT_DIR = "/root/autodl-tmp/Fusion1/OmniGen/Classify/data/omnigen"

print("="*70)
print("📊 数据集合并：Original + DataGen → OmniGen")
print("="*70)

# 检查目录
print("\n检查输入目录...")
for name, path in [("Original", ORIGINAL_DIR), ("DataGen", DATAGEN_DIR)]:
    if os.path.exists(path):
        classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        print(f"✅ {name}: {path} ({len(classes)} 类别)")
    else:
        print(f"❌ {name}: {path} (不存在)")
        exit(1)

# 创建输出目录
if os.path.exists(OUTPUT_DIR):
    print(f"\n⚠️  输出目录已存在，将被覆盖：{OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n创建输出目录：{OUTPUT_DIR}")

# 获取所有类别
original_classes = set([d for d in os.listdir(ORIGINAL_DIR) if os.path.isdir(os.path.join(ORIGINAL_DIR, d))])
datagen_classes = set([d for d in os.listdir(DATAGEN_DIR) if os.path.isdir(os.path.join(DATAGEN_DIR, d))])
all_classes = sorted(set(original_classes | datagen_classes))

print(f"\n合并信息:")
print(f"   总类别数：{len(all_classes)}")
print(f"   Original 独有：{len(original_classes - datagen_classes)}")
print(f"   DataGen 独有：{len(datagen_classes - original_classes)}")
print(f"   两者共有：{len(original_classes & datagen_classes)}")

# 合并数据
print(f"\n开始合并...")
merged_counts = {}
total_original = 0
total_datagen = 0

for cls in all_classes:
    dst_cls_dir = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(dst_cls_dir, exist_ok=True)
    
    orig_count = 0
    datagen_count = 0
    
    # 复制 Original 数据
    if cls in original_classes:
        src_cls_dir = os.path.join(ORIGINAL_DIR, cls)
        images = [f for f in os.listdir(src_cls_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        for img in images:
            shutil.copy2(os.path.join(src_cls_dir, img), 
                        os.path.join(dst_cls_dir, img))
        orig_count = len(images)
        total_original += orig_count
    
    # 复制 DataGen 数据
    if cls in datagen_classes:
        src_cls_dir = os.path.join(DATAGEN_DIR, cls)
        images = [f for f in os.listdir(src_cls_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        for i, img in enumerate(images):
            name, ext = os.path.splitext(img)
            dst_name = f"datagen_{i:04d}_{name}{ext}"
            shutil.copy2(os.path.join(src_cls_dir, img), 
                        os.path.join(dst_cls_dir, dst_name))
        datagen_count = len(images)
        total_datagen += datagen_count
    
    merged_counts[cls] = orig_count + datagen_count

# 打印统计
print(f"\n📊 合并完成！")
print(f"   Original 样本数：{total_original}")
print(f"   DataGen 样本数：{total_datagen}")
print(f"   合并后总样本数：{sum(merged_counts.values())}")
print(f"   平均每个类别：{sum(merged_counts.values()) / len(merged_counts):.1f} 张")

print(f"\n前 10 个类别详情:")
for i, cls in enumerate(list(merged_counts.keys())[:10]):
    orig = orig_count if cls in original_classes else 0
    datagen = datagen_count if cls in datagen_classes else 0
    total = merged_counts[cls]
    print(f"   {i+1:2d}. {cls:20s}: 原始={orig:4d}, DataGen={datagen:4d}, 总计={total:4d}")

print(f"\n尾部类别（样本最少的 5 个）:")
sorted_counts = sorted(merged_counts.items(), key=lambda x: x[1])
for i, (cls, count) in enumerate(sorted_counts[:5]):
    has_orig = cls in original_classes
    has_datagen = cls in datagen_classes
    print(f"   {i+1}. {cls:20s}: 总计={count:4d} (Original: {'✓' if has_orig else '✗'}, DataGen: {'✓' if has_datagen else '✗'})")

print(f"\n✅ 已保存到：{OUTPUT_DIR}")
print("\n" + "="*70)
print("下一步：运行训练")
print("="*70)
print("cd /root/autodl-tmp/Fusion1/OmniGen/Classify/scripts")
print("python train.py --exp_id omnigen")
print("="*70 + "\n")
