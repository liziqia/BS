"""
从现有训练JSON中筛选16个类别，每个类别选取前4对和后4对（共8对）数据，合成新的训练JSON
"""

import json
import os
from pathlib import Path

# 排除的7个数量最多的类别
EXCLUDE_CLASSES = {'0', '2', '4', '6', '10', '13', '17'}

# 舰船类别映射
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


def extract_class_data(json_path):
    """读取JSON并按类别分组"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    class_data = {}
    for entry in data:
        input_images = entry.get('input_images', [])
        if not input_images:
            continue
        
        # 从路径中提取类别ID: train/15/xxx.jpg -> 15
        img_path = input_images[0]
        class_id = img_path.split('/')[1]
        
        if class_id not in class_data:
            class_data[class_id] = []
        class_data[class_id].append(entry)
    
    return class_data


def select_pairs(entries, num_pairs=8):
    """选取前num_pairs/2对和后num_pairs/2对"""
    half = num_pairs // 2
    if len(entries) <= num_pairs:
        return entries
    
    selected = entries[:half] + entries[-half:]
    return selected


def main():
    annotations_dir = '/root/autodl-tmp/Fusion1/OmniGen/Ship/data/annotations'
    output_path = '/root/autodl-tmp/Fusion1/OmniGen/Ship/data/annotations/train_16classes.json'
    
    selected_data = []
    total_selected = 0
    
    for class_id in sorted(SHIP_CLASSES.keys(), key=lambda x: int(x)):
        if class_id in EXCLUDE_CLASSES:
            continue
        
        class_json = os.path.join(annotations_dir, f'class_{class_id}_train.json')
        if not os.path.exists(class_json):
            print(f"Warning: {class_json} not found, skipping class {class_id}")
            continue
        
        with open(class_json, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        
        class_name = SHIP_CLASSES.get(class_id, f'Class {class_id}')
        selected = select_pairs(entries, num_pairs=8)
        selected_data.extend(selected)
        
        print(f"Class {class_id} ({class_name}): {len(entries)} -> {len(selected)} pairs")
        total_selected += len(selected)
    
    # 保存新JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nTotal selected: {total_selected} pairs")
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
