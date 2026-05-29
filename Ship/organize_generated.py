"""
将 batch_generated 中的 LoRA 生成图像按类别复制到 DataGen 文件夹
跳过 DataGen 中已存在的文件，不覆盖
"""

import os
import shutil

SOURCE_DIR = "/root/autodl-tmp/Fusion1/OmniGen/Ship/results/batch_generated"
DEST_DIR = "/root/autodl-tmp/Fusion1/OmniGen/Ship/DataGen"


def extract_category_id(filename):
    """
    从文件名中提取类别 ID
    例如：1_3_30_10961_lora.png -> 1
    """
    parts = filename.split('_')
    if parts[0].isdigit():
        return int(parts[0])
    return None


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    
    for cat_id in range(23):
        cat_dest_dir = os.path.join(DEST_DIR, str(cat_id))
        os.makedirs(cat_dest_dir, exist_ok=True)
    
    files = os.listdir(SOURCE_DIR)
    lora_files = [f for f in files if f.endswith('_lora.png')]
    
    print(f"Found {len(lora_files)} LoRA generated images in {SOURCE_DIR}")
    
    copied_count = {cat: 0 for cat in range(23)}
    skipped_count = 0
    
    for filename in lora_files:
        cat_id = extract_category_id(filename)
        
        if cat_id is None or cat_id >= 23:
            print(f"  Skip (unknown category): {filename}")
            continue
        
        cat_dest_dir = os.path.join(DEST_DIR, str(cat_id))
        dest_path = os.path.join(cat_dest_dir, filename)
        
        if os.path.exists(dest_path):
            skipped_count += 1
            continue
        
        src_path = os.path.join(SOURCE_DIR, filename)
        shutil.copy2(src_path, dest_path)
        copied_count[cat_id] += 1
    
    print("\nCopy complete:")
    for cat_id in range(23):
        if copied_count[cat_id] > 0:
            print(f"  Category {cat_id}: {copied_count[cat_id]} images")
    
    total = sum(copied_count.values())
    print(f"\nTotal: {total} images copied to {DEST_DIR}")
    print(f"Skipped: {skipped_count} (already exist)")


if __name__ == "__main__":
    main()
