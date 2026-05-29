"""
生成 FGSC 数据集的训练/测试 JSON 标注文件
支持三种模式：
1. train 内部配对（相邻图像）
2. train → target 配对（输入→输出）
3. test 独立生成
"""

import os
import json
import argparse
from pathlib import Path


# FGSC 舰船类别映射
SHIP_CLASSES = {
    '0': 'Amphibious Assault Ship',
    '1': 'Nimitz-class Aircraft Carrier',
    '2': 'Arleigh Burke-class Destroyer',
    '3': 'Hyuga-class Helicopter Destroyer',
    '4': 'Yamato-class Battleship',
    '5': 'Blue Ridge-class Command Ship',
    '6': 'Type 075 Amphibious Assault Ship',
    '7': 'Wasp-class Amphibious Assault Ship',
    '8': 'America-class Amphibious Assault Ship',
    '9': 'San Antonio-class Amphibious Transport Dock',
    '10': 'Virginia-class Nuclear Submarine',
    '11': 'Mercy-class Hospital Ship',
    '12': 'Gepard-class Frigate',
    '13': 'Type 054A Frigate',
    '14': 'Container Ship',
    '15': 'Roll-on/Roll-off Ship',
    '16': 'Bridge Construction Vessel',
    '17': 'Semi-submersible Ship',
    '18': 'Oil Tanker',
    '19': 'Bulk Carrier',
    '20': 'Air-Cushioned Landing Craft',
    '21': 'Liquefied Natural Gas Carrier',
    '22': 'Ultra-large Container Ship'
}


def generate_train_internal_json(image_dir, output_json, num_references=1):
    """
    train 内部配对：相邻图像作为输入输出对
    """
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist!")
        return
    
    data = []
    
    for class_dir in sorted(image_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_id = class_dir.name
        class_name = SHIP_CLASSES.get(class_id, f'ship class {class_id}')
        
        all_images = sorted(list(class_dir.glob('*.jpg')))
        
        if len(all_images) == 0:
            continue
        
        for idx, img_file in enumerate(all_images):
            relative_path = str(img_file.relative_to(image_dir))
            
            other_images = [img for img in all_images if img != img_file]
            if len(other_images) == 0:
                continue
            
            num_refs = min(num_references, len(other_images))
            reference_images = other_images[:num_refs]
            ref_paths = [str(ref.relative_to(image_dir)) for ref in reference_images]
            
            image_tokens = " ".join([f"<|image_{i+1}|>" for i in range(len(ref_paths))])
            
            if num_refs == 1:
                instruction = f"Generate a variation of this {class_name}. {image_tokens}"
            else:
                instruction = f"Generate a variation of this {class_name}. {image_tokens}"
            
            entry = {
                "instruction": instruction,
                "input_images": ref_paths,
                "output_image": str(relative_path)
            }
            
            data.append(entry)
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(data)} training entries (internal) to {output_json}")
    return len(data)


def generate_train_target_json(train_dir, target_dir, output_json, num_references=1):
    """
    train → target 配对：train 图像作为输入，target 图像作为输出
    """
    train_dir = Path(train_dir)
    target_dir = Path(target_dir)
    
    if not train_dir.exists():
        print(f"Error: Train directory {train_dir} does not exist!")
        return
    
    if not target_dir.exists():
        print(f"Error: Target directory {target_dir} does not exist!")
        return
    
    data = []
    
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_id = class_dir.name
        class_name = SHIP_CLASSES.get(class_id, f'ship class {class_id}')
        
        train_images = sorted(list(class_dir.glob('*.jpg')))
        target_class_dir = target_dir / class_id
        
        if not target_class_dir.exists():
            print(f"Warning: Target class directory {target_class_dir} does not exist!")
            continue
        
        target_images = sorted(list(target_class_dir.glob('*.jpg')))
        
        if len(train_images) == 0 or len(target_images) == 0:
            continue
        
        # 为每张 target 图像配对 train 图像作为输入
        for target_img in target_images:
            target_relative = str(target_img.relative_to(target_dir))
            
            # 从 train 中选择参考图像
            num_refs = min(num_references, len(train_images))
            reference_images = train_images[:num_refs]
            ref_paths = [str(ref.relative_to(train_dir)) for ref in reference_images]
            
            image_tokens = " ".join([f"<|image_{i+1}|>" for i in range(len(ref_paths))])
            
            instruction = f"Generate a {class_name} similar to the reference. {image_tokens}"
            
            entry = {
                "instruction": instruction,
                "input_images": ref_paths,
                "output_image": target_relative
            }
            
            data.append(entry)
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(data)} training entries (train→target) to {output_json}")
    return len(data)


def generate_test_json(image_dir, output_json, num_references=1):
    """
    生成测试集 JSON
    """
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist!")
        return
    
    data = []
    
    for class_dir in sorted(image_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_id = class_dir.name
        class_name = SHIP_CLASSES.get(class_id, f'ship class {class_id}')
        
        all_images = sorted(list(class_dir.glob('*.jpg')))
        
        if len(all_images) == 0:
            continue
        
        num_refs = min(num_references, len(all_images))
        reference_images = all_images[:num_refs]
        
        for img_file in all_images:
            relative_path = str(img_file.relative_to(image_dir))
            ref_paths = [str(ref.relative_to(image_dir)) for ref in reference_images]
            
            image_tokens = " ".join([f"<|image_{i+1}|>" for i in range(len(ref_paths))])
            
            if num_refs == 1:
                instruction = f"Generate an image of a {class_name} similar to the reference image. {image_tokens}"
            else:
                instruction = f"Generate an image of a {class_name} similar to the reference images. {image_tokens}"
            
            entry = {
                "instruction": instruction,
                "input_images": ref_paths,
                "output_image": str(relative_path)
            }
            
            data.append(entry)
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(data)} test entries to {output_json}")
    return len(data)


def generate_class_json(image_dir, output_json, class_id, class_name, num_references=1):
    """
    生成单个类别的 JSON
    """
    class_dir = image_dir / class_id
    
    if not class_dir.exists():
        print(f"Warning: Class directory {class_dir} does not exist!")
        return 0
    
    data = []
    
    all_images = sorted(list(class_dir.glob('*.jpg')))
    
    if len(all_images) < 2:
        print(f"Warning: Need at least 2 images in {class_dir}, got {len(all_images)}")
        return 0
    
    for i in range(0, len(all_images) - 1, 2):
        input_img = all_images[i]
        output_img = all_images[i + 1]
        
        input_relative = str(input_img.relative_to(image_dir.parent))
        output_relative = str(output_img.relative_to(image_dir.parent))
        
        image_tokens = " ".join([f"<|image_{j+1}|>" for j in range(num_references)])
        
        if num_references == 1:
            instruction = f"Generate an image of a {class_name} similar to the reference image. {image_tokens}"
        else:
            instruction = f"Generate an image of a {class_name} similar to the reference images. {image_tokens}"
        
        entry = {
            "instruction": instruction,
            "input_images": [input_relative],
            "output_image": output_relative
        }
        
        data.append(entry)
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(data)} entries for class {class_id} ({class_name}) to {output_json}")
    return len(data)


def main():
    parser = argparse.ArgumentParser(description="Generate JSON annotations for FGSC dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/autodl-tmp/Fusion1/OmniGen/Ship/data/images/FGSC",
        help="Root directory of processed FGSC dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/Fusion1/OmniGen/Ship/data/annotations",
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--num_references",
        type=int,
        default=1,
        help="Number of reference images per class"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["internal", "target", "test", "all"],
        help="Generation mode: internal (train内部配对), target (train→target), test, or all"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    train_dir = data_dir / "train"
    target_dir = data_dir / "target"
    test_dir = data_dir / "test"
    
    if args.mode in ["internal", "all"]:
        if train_dir.exists():
            train_json = output_dir / "fgsc_train.json"
            generate_train_internal_json(train_dir, train_json, num_references=args.num_references)
        else:
            print(f"Warning: Train directory {train_dir} does not exist!")
    
    if args.mode in ["target", "all"]:
        if train_dir.exists() and target_dir.exists():
            train_target_json = output_dir / "fgsc_train_target.json"
            generate_train_target_json(train_dir, target_dir, train_target_json, num_references=args.num_references)
        else:
            print(f"Warning: Train or Target directory does not exist!")
    
    if args.mode in ["test", "all"]:
        if test_dir.exists():
            test_json = output_dir / "fgsc_test.json"
            generate_test_json(test_dir, test_json, num_references=args.num_references)
        else:
            print(f"Warning: Test directory {test_dir} does not exist!")
    
    if args.mode == "all":
        print("\nGenerating per-class JSON files...")
        for class_id, class_name in SHIP_CLASSES.items():
            if train_dir.exists():
                class_train_json = output_dir / f"class_{class_id}_train.json"
                generate_class_json(train_dir, class_train_json, class_id, class_name, num_references=args.num_references)
            
            if test_dir.exists():
                class_test_json = output_dir / f"class_{class_id}_test.json"
                generate_class_json(test_dir, class_test_json, class_id, class_name, num_references=args.num_references)


if __name__ == "__main__":
    main()
