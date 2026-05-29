"""
舰船图像批量生成推理脚本
用于为每种类别基于随机参考图像和随机多样化生成指令，一次生成nx23张图像
"""

import os
import argparse
import torch
import random
from PIL import Image
from pathlib import Path
from glob import glob

from Ship import OmniGenPipeline

# 头部类别（已有足够数据，不需要补充）
# HEAD_CATEGORIES = {0, 2, 4, 6, 10, 13, 17}
HEAD_CATEGORIES = {0, 2, 4, 6, 10, 13, 17,1,3,5,7,8,9}

# 尾部类别（需要补充数据的类别）
TAIL_CATEGORIES = [cid for cid in range(23) if cid not in HEAD_CATEGORIES]

# 类别ID到名称的映射
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
# 多样化生成指令模板（通用型，适用于随机参考图像）
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
    """查找最新的 checkpoint 目录"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [d for d in os.listdir(checkpoint_dir) 
                   if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.replace('-', '').isdigit()]
    
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: int(x))
    return os.path.join(checkpoint_dir, checkpoints[-1])


def get_reference_images(data_dir, category_id):
    """获取指定类别的所有参考图像路径"""
    category_dir = os.path.join(data_dir, str(category_id))
    if not os.path.exists(category_dir):
        print(f"Warning: Category directory {category_dir} does not exist")
        return []
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(category_dir, ext)))
    
    return sorted(image_paths)


def batch_generate_ship_images(args):
    """批量执行舰船图像生成推理：为16个尾部类别生成数据，采用三阶段策略"""
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
            print("Model compiled with torch.compile for faster inference")
        except Exception as e:
            print(f"torch.compile not available: {e}")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    comparison_tasks = []
    
    for category_id in TAIL_CATEGORIES:
        ship_type = CATEGORY_NAMES[category_id]
        
        reference_images = get_reference_images(args.data_dir, category_id)
        
        if len(reference_images) == 0:
            print(f"Warning: No reference images found for category {category_id}, skipping...")
            continue
        
        comparison_ref = random.choice(reference_images)
        comparison_prompt = random.choice(DIVERSE_PROMPTS).format(ship_type=ship_type)
        
        comparison_tasks.append({
            'category_id': category_id,
            'ship_type': ship_type,
            'input_image_path': comparison_ref,
            'prompt': comparison_prompt
        })
    
    num_comparison = len(comparison_tasks)
    
    print(f"\n{'='*60}")
    print("Starting batch ship image generation...")
    print(f"Head categories (skipped): {sorted(HEAD_CATEGORIES)}")
    print(f"Tail categories (generating): {TAIL_CATEGORIES}")
    print(f"Images per category: {args.num_images_per_category}")
    print(f"Total tail categories: {num_comparison}")
    print(f"{'='*60}")
    if not args.skip_phase1:
        print(f"Phase 1: {num_comparison} base model images (no LoRA)")
    else:
        print(f"Phase 1: SKIPPED (use --skip_phase1=False to enable)")
    print(f"Phase 2: {num_comparison} LoRA images (same references)")
    print(f"Phase 3: {num_comparison * (args.num_images_per_category - 1)} LoRA images (random references)")
    if args.skip_phase1:
        total_images = num_comparison * args.num_images_per_category - num_comparison
    else:
        total_images = num_comparison * args.num_images_per_category
    print(f"Total: {total_images} images")
    print(f"{'='*60}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}\n")
    
    log_file = os.path.join(args.output_dir, "generation_info.txt")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        if args.skip_phase1:
            total_images = num_comparison * args.num_images_per_category - num_comparison
        else:
            total_images = num_comparison * args.num_images_per_category
        f.write(f"Total images: {total_images}\n")
        if not args.skip_phase1:
            f.write(f"  Phase 1 (Base model): {num_comparison}\n")
        else:
            f.write(f"  Phase 1 (Base model): SKIPPED\n")
        f.write(f"  Phase 2 (LoRA comparison): {num_comparison}\n")
        f.write(f"  Phase 3 (LoRA random): {num_comparison * (args.num_images_per_category - 1)}\n\n")
    
    total_generated = 0
    
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
            print(f"    Prompt: {prompt[:80]}...")
            
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
            
            output_filename = f"{Path(input_image_path).stem}_base.png"
            output_path = os.path.join(args.output_dir, output_filename)
            
            if isinstance(result, list):
                result[0].save(output_path)
            else:
                result.save(output_path)
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{output_filename} | ref: {Path(input_image_path).name} | prompt: {prompt}\n")
            
            total_generated += 1
            print(f"    Saved (base): {output_path}")
        
        print(f"\n  Phase 1 complete: {num_comparison} base model images generated")
    
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
    
    print(f"\n{'='*60}")
    print(f"Phase 2: Generating {num_comparison} images WITH LoRA (same references as Phase 1)")
    print(f"{'='*60}")
    
    for task_idx, task in enumerate(comparison_tasks):
        category_id = task['category_id']
        ship_type = task['ship_type']
        input_image_path = task['input_image_path']
        prompt = task['prompt']
        
        print(f"\n  [LoRA-Comp {task_idx + 1}/{num_comparison}] Category {category_id}: {ship_type}")
        print(f"    Reference: {Path(input_image_path).name}")
        print(f"    Prompt: {prompt[:80]}...")
        
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
        output_path = os.path.join(args.output_dir, output_filename)
        
        if isinstance(result, list):
            result[0].save(output_path)
        else:
            result.save(output_path)
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{output_filename} | ref: {Path(input_image_path).name} | prompt: {prompt}\n")
        
        total_generated += 1
        print(f"    Saved (lora): {output_path}")
    
    print(f"\n  Phase 2 complete: {num_comparison} LoRA comparison images generated")
    
    num_remaining = args.num_images_per_category - 1
    total_remaining = num_comparison * num_remaining
    
    print(f"\n{'='*60}")
    print(f"Phase 3: Generating {total_remaining} images WITH LoRA ({num_remaining} per category)")
    print(f"{'='*60}")
    
    for category_id in TAIL_CATEGORIES:
        ship_type = CATEGORY_NAMES[category_id]
        
        reference_images = get_reference_images(args.data_dir, category_id)
        
        if len(reference_images) == 0:
            continue
        
        print(f"\n  Processing Category {category_id}: {ship_type}")
        
        selected_images = [random.choice(reference_images) for _ in range(num_remaining)]
        selected_prompts = [random.choice(DIVERSE_PROMPTS).format(ship_type=ship_type) 
                           for _ in range(num_remaining)]
        
        for img_idx in range(num_remaining):
            input_image_path = selected_images[img_idx]
            prompt = selected_prompts[img_idx]
            
            print(f"\n    [{img_idx + 1}/{num_remaining}] Generating...")
            print(f"      Reference: {Path(input_image_path).name}")
            print(f"      Prompt: {prompt[:80]}...")
            
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
            output_path = os.path.join(args.output_dir, output_filename)
            
            if isinstance(result, list):
                result[0].save(output_path)
            else:
                result.save(output_path)
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{output_filename} | ref: {Path(input_image_path).name} | prompt: {prompt}\n")
            
            total_generated += 1
            print(f"      Saved: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"Batch generation complete!")
    print(f"Total images generated: {total_generated}")
    print(f"  - Phase 1 (Base model comparison): {num_comparison}")
    print(f"  - Phase 2 (LoRA comparison): {num_comparison}")
    print(f"  - Phase 3 (LoRA random): {total_remaining}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Batch Ship Image Generation Inference")
    
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
    
    parser.add_argument("--num_images_per_category", type=int, default=20,
                        help="Number of images to generate per category (n)")
    
    parser.add_argument("--skip_phase1", action="store_true", default=True,
                        help="Skip Phase 1 (base model generation without LoRA). Default: True")
    
    parser.add_argument("--num_inference_steps", type=int, default=25,
                        help="Number of denoising steps (lower=faster, 25-30 recommended)")
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--img_guidance_scale", type=float, default=1.6)
    parser.add_argument("--max_image_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=142, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    batch_generate_ship_images(args)


if __name__ == "__main__":
    main()
