"""
按分配比例批量生成舰船图像推理脚本
根据 selection_log.json 中的 allocated_counts 等比例缩放生成数量
"""

import os
import argparse
import torch
import random
import json
import math
from PIL import Image
from pathlib import Path
from glob import glob

from Ship import OmniGenPipeline

HEAD_CATEGORIES = {0, 2, 4, 6, 10, 13, 17}

CATEGORY_NAMES = {
    0: "Amphibious_Assault_Ship",
    1: "Nimitz-class_Aircraft_Carrier",
    2: "Arleigh_Burke-class_Destroyer",
    3: "Hyuga-class_Helicopter_Destroyer",
    4: "Yamato-class_Battleship",
    5: "Blue_Ridge-class_Command_Ship",
    6: "Type_075_Amphibious_Assault_Ship",
    7: "Wasp-class_Amphibious_Assault_Ship",
    8: "America-class_Amphibious_Assault_Ship",
    9: "San_Antonio-class_Amphibious_Transport_Dock",
    10: "Virginia-class_Nuclear_Submarine",
    11: "Mercy-class_Hospital_Ship",
    12: "Gepard-class_Frigate",
    13: "Type_054A_Frigate",
    14: "Container_Ship",
    15: "Roll-on_Roll-off_Ship",
    16: "Bridge_Construction_Vessel",
    17: "Semi-submersible_Ship",
    18: "Oil_Tanker",
    19: "Bulk_Carrier",
    20: "Air-Cushioned_Landing_Craft",
    21: "Liquefied_Natural_Gas_Carrier",
    22: "Ultra-large_Container_Ship"
}

DIVERSE_PROMPTS = [
    "This is a remote sensing image of a {ship_type}. Generate a diversified image base on this.<|image_1|>",
    "A satellite view of {ship_type}. Create a diverse representation based on this reference.<|image_1|>",
    "Remote sensing imagery showing {ship_type}. Generate a varied image from this perspective.<|image_1|>",
    "An aerial photograph of {ship_type}. Produce a diversified output image.<|image_1|>",
    "This overhead view captures a {ship_type}. Generate a diverse interpretation.<|image_1|>",
    "Top-down satellite image of {ship_type}. Create a varied visualization.<|image_1|>",
    "A bird's eye view of {ship_type}. Generate a diversified representation.<|image_1|>",
    "Remote sensing data of {ship_type}. Produce a diverse image based on this input.<|image_1|>",
    "This is an aerial image showing {ship_type}. Generate a varied output.<|image_1|>",
    "Satellite imagery of {ship_type}. Create a diversified visual interpretation.<|image_1|>",
    "Based on this reference image of {ship_type}, generate a diversified version with different lighting and perspective.<|image_1|>",
    "This remote sensing image shows a {ship_type}. Generate a diverse version with varied scene composition.<|image_1|>",
    "An overhead view of {ship_type}. Create a diversified image with different water and background conditions.<|image_1|>",
    "This is a satellite image of {ship_type}. Generate a varied representation maintaining the ship characteristics.<|image_1|>",
    "Aerial imagery of {ship_type}. Produce a diversified output with adjusted brightness and contrast.<|image_1|>",
    "This view captures a {ship_type}. Generate a diverse interpretation with different environmental conditions.<|image_1|>",
    "Remote sensing photograph of {ship_type}. Create a varied image with modified scene layout.<|image_1|>",
    "This satellite observation shows {ship_type}. Generate a diversified version with altered viewing angle.<|image_1|>",
    "An aerial shot of {ship_type}. Produce a diverse representation with different surrounding context.<|image_1|>",
    "This is a top-down view of {ship_type}. Generate a varied image maintaining ship features while changing scene elements.<|image_1|>"
]


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [d for d in os.listdir(checkpoint_dir) 
                   if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.replace('-', '').isdigit()]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x))
    return os.path.join(checkpoint_dir, checkpoints[-1])


def get_reference_images(data_dir, category_id):
    category_dir = os.path.join(data_dir, str(category_id))
    if not os.path.exists(category_dir):
        print(f"Warning: Category directory {category_dir} does not exist")
        return []
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(category_dir, ext)))
    return sorted(image_paths)


def load_allocation(json_path, scale_ratio=0.2):
    """
    从 selection_log.json 读取 allocated_counts，按比例缩放
    
    参数:
        json_path: selection_log.json 路径
        scale_ratio: 缩放比例（默认 0.2，即 51→10, 80→16）
    
    返回:
        category_counts: {category_id: num_images_to_generate}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    allocated = data['allocated_counts']
    category_counts = {}
    
    total_allocated = sum(allocated.values())
    
    for cat_id_str, count in allocated.items():
        scaled = max(1, int(round(count * scale_ratio)))
        category_counts[int(cat_id_str)] = scaled
    
    total_scaled = sum(category_counts.values())
    print(f"Allocation loaded from {json_path}")
    print(f"  Original total: {total_allocated}, Scaled total: {total_scaled} (ratio={scale_ratio})")
    
    return category_counts


def get_unique_output_path(output_dir, filename):
    """
    获取唯一的输出路径，避免覆盖已有文件
    例如：1_3_30_10961_lora.png 已存在 ->
         1_3_30_10961_1_lora.png ->
         1_3_30_10961_2_lora.png 以此类推
    """
    output_path = os.path.join(output_dir, filename)
    if not os.path.exists(output_path):
        return output_path
    
    name, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_name = f"{name}_{counter}{ext}"
        new_path = os.path.join(output_dir, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def batch_generate_ship_images(args):
    print(f"Loading model from {args.model_name_or_path}...")
    pipe = OmniGenPipeline.from_pretrained(args.model_name_or_path, vae_path=args.vae_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    
    try:
        if hasattr(pipe, 'transformer'):
            pipe.transformer = pipe.transformer.half()
            print("Model converted to FP16 for faster inference")
    except Exception as e:
        print(f"FP16 conversion skipped: {e}")
    
    if hasattr(pipe, 'transformer') and hasattr(pipe.transformer, 'forward'):
        try:
            pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile not available: {e}")
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 读取分配比例
    category_counts = load_allocation(args.allocation_json, args.scale_ratio)
    
    # 按 category_id 排序
    sorted_categories = sorted(category_counts.keys())
    
    print(f"\n{'='*60}")
    print("Allocation-based generation plan:")
    for cid in sorted_categories:
        print(f"  Category {cid:2d} ({CATEGORY_NAMES.get(cid, 'Unknown'):40s}): {category_counts[cid]:3d} images")
    print(f"  Total: {sum(category_counts.values())} images")
    print(f"{'='*60}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_file = os.path.join(args.output_dir, "generation_info.txt")
    
    comparison_tasks = []
    
    for category_id in sorted_categories:
        ship_type = CATEGORY_NAMES.get(category_id, f"Category_{category_id}")
        reference_images = get_reference_images(args.data_dir, category_id)
        
        if len(reference_images) == 0:
            print(f"Warning: No reference images for category {category_id}, skipping...")
            continue
        
        comparison_ref = random.choice(reference_images)
        comparison_prompt = random.choice(DIVERSE_PROMPTS).format(ship_type=ship_type)
        
        comparison_tasks.append({
            'category_id': category_id,
            'ship_type': ship_type,
            'num_images': category_counts[category_id],
            'input_image_path': comparison_ref,
            'prompt': comparison_prompt
        })
    
    num_comparison = len(comparison_tasks)
    total_images = sum(category_counts.values())
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Allocation JSON: {args.allocation_json}\n")
        f.write(f"Scale ratio: {args.scale_ratio}\n")
        f.write(f"Total images to generate: {total_images}\n")
        for cid in sorted_categories:
            f.write(f"  Category {cid}: {category_counts[cid]}\n")
        f.write("\n")
    
    total_generated = 0
    img_counter = {cid: 0 for cid in sorted_categories}
    
    # =====================================
    # Phase 1: Base model (no LoRA)
    # =====================================
    if not args.skip_phase1:
        print(f"\n{'='*60}")
        print(f"Phase 1: Generating {num_comparison} images with BASE MODEL (no LoRA)")
        print(f"{'='*60}")
        
        for task_idx, task in enumerate(comparison_tasks):
            category_id = task['category_id']
            ship_type = task['ship_type']
            input_image_path = task['input_image_path']
            prompt = task['prompt']
            
            print(f"\n  [Base {task_idx + 1}/{num_comparison}] Category {category_id}: {ship_type}")
            print(f"    Reference: {Path(input_image_path).name}")
            
            input_img = Image.open(input_image_path).convert('RGB')
            input_height, input_width = input_img.size[1], input_img.size[0]
            input_height = (input_height // 16) * 16
            input_width = (input_width // 16) * 16
            
            result = pipe(
                prompt=prompt,
                input_images=[input_image_path],
                height=input_height,
                width=input_width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                use_img_guidance=True,
                img_guidance_scale=args.img_guidance_scale,
                max_input_image_size=args.max_image_size,
                seed=random.randint(0, 2**32 - 1) if args.seed is None else args.seed + total_generated,
                use_kv_cache=True,
            )
            
            # 保存到输出目录（与inference.py保持一致的平铺结构）
            output_filename = f"{Path(input_image_path).stem}_base.png"
            output_path = get_unique_output_path(args.output_dir, output_filename)
            
            if isinstance(result, list):
                result[0].save(output_path)
            else:
                result.save(output_path)
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{output_filename} | ref: {Path(input_image_path).name} | prompt: {prompt} | cat: {category_id}\n")
            
            total_generated += 1
            img_counter[category_id] += 1
            print(f"    Saved: {output_path}")
        
        print(f"\n  Phase 1 complete: {num_comparison} base model images generated")
    
    # =====================================
    # Merge LoRA
    # =====================================
    if args.lora_path is not None:
        lora_path_to_load = None
        if os.path.isdir(args.lora_path):
            latest_checkpoint = find_latest_checkpoint(args.lora_path)
            if latest_checkpoint:
                lora_path_to_load = latest_checkpoint
                print(f"\nFound latest LoRA checkpoint: {latest_checkpoint}")
            else:
                lora_path_to_load = args.lora_path
        elif os.path.exists(args.lora_path):
            lora_path_to_load = args.lora_path
        
        if lora_path_to_load and os.path.exists(lora_path_to_load):
            print(f"\n{'='*60}")
            print(f"Merging LoRA from {lora_path_to_load}...")
            print(f"{'='*60}")
            pipe.merge_lora(lora_path_to_load)
        else:
            print(f"\nWarning: LoRA path not found, using base model for all remaining images")
    
    # =====================================
    # Phase 2: LoRA comparison (1 per category, same reference as Phase 1)
    # =====================================
    print(f"\n{'='*60}")
    print(f"Phase 2: Generating {num_comparison} LoRA comparison images")
    print(f"{'='*60}")
    
    for task_idx, task in enumerate(comparison_tasks):
        category_id = task['category_id']
        ship_type = task['ship_type']
        input_image_path = task['input_image_path']
        prompt = task['prompt']
        
        print(f"\n  [LoRA-Comp {task_idx + 1}/{num_comparison}] Category {category_id}: {ship_type}")
        print(f"    Reference: {Path(input_image_path).name}")
        
        input_img = Image.open(input_image_path).convert('RGB')
        input_height, input_width = input_img.size[1], input_img.size[0]
        input_height = (input_height // 16) * 16
        input_width = (input_width // 16) * 16
        
        result = pipe(
            prompt=prompt,
            input_images=[input_image_path],
            height=input_height,
            width=input_width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            use_img_guidance=True,
            img_guidance_scale=args.img_guidance_scale,
            max_input_image_size=args.max_image_size,
            seed=random.randint(0, 2**32 - 1) if args.seed is None else args.seed + total_generated,
            use_kv_cache=True,
        )
        
        output_filename = f"{Path(input_image_path).stem}_lora.png"
        output_path = get_unique_output_path(args.output_dir, output_filename)
        
        if isinstance(result, list):
            result[0].save(output_path)
        else:
            result.save(output_path)
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{output_filename} | ref: {Path(input_image_path).name} | prompt: {prompt} | cat: {category_id}\n")
        
        total_generated += 1
        img_counter[category_id] += 1
        print(f"    Saved: {output_path}")
    
    print(f"\n  Phase 2 complete: {num_comparison} LoRA comparison images generated")
    
    # =====================================
    # Phase 3: Remaining LoRA images per category (random references)
    # =====================================
    total_remaining = 0
    remaining_tasks = []
    for task in comparison_tasks:
        cid = task['category_id']
        n_needed = category_counts[cid] - img_counter[cid]
        if n_needed > 0:
            total_remaining += n_needed
            remaining_tasks.append({
                'category_id': cid,
                'ship_type': task['ship_type'],
                'n_needed': n_needed
            })
    
    if total_remaining > 0:
        print(f"\n{'='*60}")
        print(f"Phase 3: Generating {total_remaining} additional LoRA images")
        print(f"{'='*60}")
        
        for task in remaining_tasks:
            category_id = task['category_id']
            ship_type = task['ship_type']
            n_needed = task['n_needed']
            
            reference_images = get_reference_images(args.data_dir, category_id)
            if len(reference_images) == 0:
                continue
            
            for img_idx in range(n_needed):
                input_image_path = random.choice(reference_images)
                prompt = random.choice(DIVERSE_PROMPTS).format(ship_type=ship_type)
                
                print(f"\n    [Cat {category_id} {img_idx + 1}/{n_needed}] {ship_type}")
                print(f"      Reference: {Path(input_image_path).name}")
                
                input_img = Image.open(input_image_path).convert('RGB')
                input_height, input_width = input_img.size[1], input_img.size[0]
                input_height = (input_height // 16) * 16
                input_width = (input_width // 16) * 16
                
                result = pipe(
                    prompt=prompt,
                    input_images=[input_image_path],
                    height=input_height,
                    width=input_width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    use_img_guidance=True,
                    img_guidance_scale=args.img_guidance_scale,
                    max_input_image_size=args.max_image_size,
                    seed=random.randint(0, 2**32 - 1) if args.seed is None else args.seed + total_generated,
                    use_kv_cache=True,
                )
                
                output_filename = f"{Path(input_image_path).stem}_{img_idx:02d}_lora.png"
                output_path = get_unique_output_path(args.output_dir, output_filename)
                
                if isinstance(result, list):
                    result[0].save(output_path)
                else:
                    result.save(output_path)
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{output_filename} | ref: {Path(input_image_path).name} | prompt: {prompt} | cat: {category_id}\n")
                
                total_generated += 1
                img_counter[category_id] += 1
                print(f"      Saved: {output_path}")
    
    # =====================================
    # Summary
    # =====================================
    print(f"\n{'='*60}")
    print(f"Batch generation complete!")
    print(f"Total images generated: {total_generated}")
    print(f"Per category:")
    for cid in sorted_categories:
        n_planned = category_counts[cid]
        n_done = img_counter.get(cid, 0)
        status = "OK" if n_done >= n_planned else f"MISSING {n_planned - n_done}"
        print(f"  Category {cid:2d}: {n_done}/{n_planned} {status}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Rate-based Batch Ship Image Generation")
    
    parser.add_argument("--model_name_or_path", type=str, 
                        default="/root/autodl-tmp/Fusion1/OmniGen/models/OmniGen-v1-weights")
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, 
                        default="/root/autodl-tmp/Fusion1/OmniGen/Ship/results/LoRA/reference_to_target/checkpoints")
    parser.add_argument("--data_dir", type=str, 
                        default="/root/autodl-tmp/Fusion1/OmniGen/Ship/data/images/FGSC/train",
                        help="Directory containing category subdirectories with reference images")
    parser.add_argument("--output_dir", type=str, 
                        default="/root/autodl-tmp/Fusion1/OmniGen/Ship/results/batch_generated")
    parser.add_argument("--allocation_json", type=str,
                        default="/root/autodl-tmp/Fusion1/OmniGen/Classify/ablation_synthetic_dataset1619/exp_01_weighted_cus/synthetic_data/selection_log.json",
                        help="Path to selection_log.json containing allocated_counts")
    parser.add_argument("--scale_ratio", type=float, default=0.2,
                        help="Scale ratio for allocation counts (0.2 means 51→10, 80→16)")
    parser.add_argument("--skip_phase1", action="store_true", default=True)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--img_guidance_scale", type=float, default=1.6)
    parser.add_argument("--max_image_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=242)

    args = parser.parse_args()
    batch_generate_ship_images(args)


if __name__ == "__main__":
    main()
