"""
FGSC 数据集预处理脚本
将图像统一调整到 512x512 分辨率：
- 如果图像大于 512x512，进行中心裁剪
- 如果图像小于 512x512，先缩放使最长边为 512，然后用黑色填充短边
- 保持目录结构和文件名不变
- 非图像文件直接复制
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
TARGET_SIZE = 512


def preprocess_image(img_path, output_path):
    """
    预处理单张图像：
    统一将长边调整到 512，然后用黑色填充到 512x512
    避免裁剪导致关键部分丢失
    """
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    max_dim = max(width, height)
    
    # 统一缩放：将长边调整到 512
    scale = TARGET_SIZE / max_dim
    new_width = int(width * scale)
    new_height = int(height * scale)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # 创建黑色背景并粘贴图像到中心
    new_img = Image.new('RGB', (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    paste_x = (TARGET_SIZE - new_width) // 2
    paste_y = (TARGET_SIZE - new_height) // 2
    new_img.paste(img, (paste_x, paste_y))
    result_img = new_img
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存图像，保持原始格式
    result_img.save(output_path)


def process_dataset(input_dir, output_dir):
    """
    处理整个数据集目录
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        return
    
    # 统计信息
    total_files = 0
    processed_images = 0
    copied_files = 0
    errors = 0
    
    # 收集所有文件
    all_files = list(input_path.rglob('*'))
    
    print(f"Found {len(all_files)} files in {input_dir}")
    print(f"Processing images to {TARGET_SIZE}x{TARGET_SIZE}...")
    
    for file_path in tqdm(all_files, desc="Processing"):
        if file_path.is_file():
            total_files += 1
            
            # 计算相对路径
            relative_path = file_path.relative_to(input_path)
            target_file = output_path / relative_path
            
            # 检查是否为图像文件
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    preprocess_image(str(file_path), str(target_file))
                    processed_images += 1
                except Exception as e:
                    print(f"\nError processing {file_path}: {e}")
                    errors += 1
            else:
                # 非图像文件，直接复制
                try:
                    os.makedirs(os.path.dirname(target_file), exist_ok=True)
                    shutil.copy2(str(file_path), str(target_file))
                    copied_files += 1
                except Exception as e:
                    print(f"\nError copying {file_path}: {e}")
                    errors += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total files: {total_files}")
    print(f"Processed images: {processed_images}")
    print(f"Copied non-image files: {copied_files}")
    print(f"Errors: {errors}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess FGSC dataset images to 512x512")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/autodl-tmp/Fusion1/OmniGen/Ship/FGSC",
        help="Input directory containing FGSC dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/Fusion1/OmniGen/Ship/data/images/FGSC",
        help="Output directory for processed images"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=512,
        help="Target image size (default: 512)"
    )
    
    args = parser.parse_args()
    
    global TARGET_SIZE
    TARGET_SIZE = args.target_size
    
    process_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
