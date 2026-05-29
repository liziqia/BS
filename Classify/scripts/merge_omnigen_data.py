#!/usr/bin/env python3
"""
合并原始数据集和 DataGen 生成的数据到 omnigen 数据集
"""

import os
import shutil
import argparse
from pathlib import Path


def check_directory(dir_path, name):
    """检查目录是否存在并返回类别信息"""
    if not os.path.exists(dir_path):
        print(f"❌ {name} 目录不存在：{dir_path}")
        return None, None
    
    classes = sorted([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])
    
    if not classes:
        print(f"❌ {name} 目录中没有找到任何类别子目录：{dir_path}")
        return None, None
    
    # 统计每个类别的样本数
    class_counts = {}
    for cls in classes:
        cls_dir = os.path.join(dir_path, cls)
        count = len([f for f in os.listdir(cls_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
        class_counts[cls] = count
    
    print(f"\n✅ {name}")
    print(f"   路径：{dir_path}")
    print(f"   类别数：{len(classes)}")
    print(f"   总样本数：{sum(class_counts.values())}")
    
    return classes, class_counts


def merge_datasets(original_dir, datagen_dir, output_dir, overwrite=False):
    """
    合并两个数据集
    
    参数：
        original_dir: 原始数据集目录
        datagen_dir: DataGen 生成的数据目录
        output_dir: 输出目录（omnigen 数据集）
        overwrite: 是否覆盖已存在的输出目录
    """
    print("=" * 70)
    print("📊 数据集合并工具：Original + DataGen → OmniGen")
    print("=" * 70)
    
    # 检查输入目录
    original_classes, original_counts = check_directory(original_dir, "原始数据集")
    datagen_classes, datagen_counts = check_directory(datagen_dir, "DataGen 数据集")
    
    if original_classes is None or datagen_classes is None:
        print("\n❌ 无法继续，请检查输入目录")
        return False
    
    # 处理输出目录
    if os.path.exists(output_dir):
        if overwrite:
            print(f"\n⚠️  输出目录已存在，将被覆盖：{output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"\n⚠️  输出目录已存在，跳过：{output_dir}")
            print(f"   如需覆盖，请添加 --overwrite 参数")
            return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并所有类别
    all_classes = sorted(set(original_classes + datagen_classes))
    print(f"\n📌 合并信息:")
    print(f"   总类别数：{len(all_classes)}")
    print(f"   输出路径：{output_dir}")
    
    merged_counts = {}
    total_original = 0
    total_datagen = 0
    
    for cls in all_classes:
        cls_has_original = cls in original_classes
        cls_has_datagen = cls in datagen_classes
        
        # 创建类别目录
        dst_cls_dir = os.path.join(output_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)
        
        # 复制原始数据
        if cls_has_original:
            src_cls_dir = os.path.join(original_dir, cls)
            images = [f for f in os.listdir(src_cls_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            
            for img in images:
                shutil.copy2(
                    os.path.join(src_cls_dir, img),
                    os.path.join(dst_cls_dir, img)
                )
            
            orig_count = original_counts.get(cls, 0)
            total_original += orig_count
        else:
            orig_count = 0
        
        # 复制 DataGen 数据
        if cls_has_datagen:
            src_cls_dir = os.path.join(datagen_dir, cls)
            images = [f for f in os.listdir(src_cls_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            
            for i, img in enumerate(images):
                # 重命名以避免冲突：original_img.jpg -> datagen_001_img.jpg
                name, ext = os.path.splitext(img)
                dst_name = f"datagen_{i:04d}_{name}{ext}"
                
                shutil.copy2(
                    os.path.join(src_cls_dir, img),
                    os.path.join(dst_cls_dir, dst_name)
                )
            
            datagen_count = datagen_counts.get(cls, 0)
            total_datagen += datagen_count
        else:
            datagen_count = 0
        
        merged_counts[cls] = orig_count + datagen_count
    
    # 打印统计信息
    print(f"\n📊 合并完成！统计信息:")
    print(f"   原始数据样本数：{total_original}")
    print(f"   DataGen 样本数：{total_datagen}")
    print(f"   合并后总样本数：{sum(merged_counts.values())}")
    print(f"   平均每个类别：{sum(merged_counts.values()) / len(merged_counts):.1f} 张")
    
    # 显示前 10 个类别的详细信息
    print(f"\n📋 前 10 个类别详情:")
    for i, cls in enumerate(list(merged_counts.keys())[:10]):
        orig = original_counts.get(cls, 0)
        datagen = datagen_counts.get(cls, 0)
        total = merged_counts[cls]
        print(f"   {i+1:2d}. {cls:20s}: 原始={orig:4d}, DataGen={datagen:4d}, 总计={total:4d}")
    
    # 显示尾部类别（样本最少的 5 个）
    print(f"\n📋 尾部类别（样本最少的 5 个）:")
    sorted_counts = sorted(merged_counts.items(), key=lambda x: x[1])
    for i, (cls, count) in enumerate(sorted_counts[:5]):
        orig = original_counts.get(cls, 0)
        datagen = datagen_counts.get(cls, 0)
        print(f"   {i+1}. {cls:20s}: 原始={orig:4d}, DataGen={datagen:4d}, 总计={count:4d}")
    
    print(f"\n✅ 合并完成！已保存到：{output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="合并原始数据集和 DataGen 数据到 omnigen 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 默认配置（推荐）
  python merge_omnigen_data.py
  
  # 自定义路径
  python merge_omnigen_data.py --original ../data/original --datagen ../../Ship/DataGen --output ../data/omnigen
  
  # 覆盖已存在的输出目录
  python merge_omnigen_data.py --overwrite
        """
    )
    
    parser.add_argument(
        "--original", 
        type=str, 
        default="/root/autodl-tmp/Fusion1/OmniGen/Classify/data/original",
        help="原始数据集目录 (默认：/root/autodl-tmp/Fusion1/OmniGen/Classify/data/original)"
    )
    parser.add_argument(
        "--datagen", 
        type=str, 
        default="/root/autodl-tmp/Fusion1/OmniGen/Ship/DataGen",
        help="DataGen 数据集目录 (默认：/root/autodl-tmp/Fusion1/OmniGen/Ship/DataGen)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="/root/autodl-tmp/Fusion1/OmniGen/Classify/data/omnigen",
        help="输出数据集目录 (默认：/root/autodl-tmp/Fusion1/OmniGen/Classify/data/omnigen)"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="覆盖已存在的输出目录"
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    original_dir = os.path.abspath(args.original)
    datagen_dir = os.path.abspath(args.datagen)
    output_dir = os.path.abspath(args.output)
    
    # 执行合并
    success = merge_datasets(original_dir, datagen_dir, output_dir, args.overwrite)
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 下一步：使用 omnigen 数据集进行训练")
        print("=" * 70)
        print(f"cd scripts")
        print(f"python train.py --exp_id omnigen")
        print("=" * 70 + "\n")
    else:
        exit(1)


if __name__ == "__main__":
    main()
