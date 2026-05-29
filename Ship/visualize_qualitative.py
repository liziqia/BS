"""
定性结果可视化脚本
从每个类别的 9 张 Phase 3 LoRA 图像中随机选择一张，生成 4 张 4x3 的示意图
"""

import os
import re
import random
from PIL import Image, ImageDraw, ImageFont

RESULT_DIR = "/root/autodl-tmp/Fusion1/OmniGen/Ship/results/batch_generated"
TRAIN_DIR = "/root/autodl-tmp/Fusion1/OmniGen/Ship/data/images/FGSC/train"
OUTPUT_DIR = "/root/autodl-tmp/Fusion1/OmniGen/Ship/results/qualitative_results"

IMG_SIZE = 512

TAIL_CATEGORIES = [1, 3, 5, 7, 8, 9, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22]


def find_ref_image(ref_filename):
    for cat_dir in os.listdir(TRAIN_DIR):
        cat_path = os.path.join(TRAIN_DIR, cat_dir)
        if os.path.isdir(cat_path):
            ref_path = os.path.join(cat_path, ref_filename)
            if os.path.exists(ref_path):
                return ref_path
    return None


def parse_log_file():
    log_file = os.path.join(RESULT_DIR, "generation_info.txt")
    category_images = {cat: [] for cat in TAIL_CATEGORIES}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '_lora.png' not in line:
                continue
            
            match = re.match(r'(.+?)\s*\|\s*ref:\s*(.+?)\s*\|\s*prompt:\s*(.+)', line.strip())
            if match:
                generated = match.group(1).strip()
                ref = match.group(2).strip()
                prompt = match.group(3).strip()
                
                cat_id = int(ref.split('_')[0])
                if cat_id in TAIL_CATEGORIES:
                    is_phase3 = '_0' in generated or '_1' in generated or '_2' in generated or \
                               '_3' in generated or '_4' in generated or '_5' in generated or \
                               '_6' in generated or '_7' in generated or '_8' in generated or \
                               '_9' in generated
                    
                    if is_phase3:
                        category_images[cat_id].append({
                            'generated': generated,
                            'ref': ref,
                            'prompt': prompt
                        })
    
    return category_images


def select_one_per_category(category_images):
    selected = []
    
    for cat_id in TAIL_CATEGORIES:
        images = category_images[cat_id]
        if len(images) == 0:
            print(f"Warning: No Phase 3 images for category {cat_id}")
            continue
        
        chosen = random.choice(images)
        ref_path = find_ref_image(chosen['ref'])
        
        if ref_path:
            chosen['ref_path'] = ref_path
            selected.append(chosen)
            print(f"Category {cat_id}: selected {chosen['generated']} with ref {chosen['ref']}")
        else:
            print(f"Warning: Could not find ref image for category {cat_id}: {chosen['ref']}")
    
    return selected


def create_prompt_image(text, width, height, font_size=32):
    img = Image.new('RGBA', (width, height), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    prompt_clean = text.replace('<|image_1|>', '').strip()
    
    margin = 15
    line_height = font_size + 4
    max_width = width - 2 * margin
    
    words = prompt_clean.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    total_text_height = len(lines) * line_height
    y_offset = max(margin, (height - total_text_height) // 2)
    
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, y_offset + i * line_height), line, fill='black', font=font)
    
    return img


def create_figure(items, output_path):
    rows = len(items)
    total_width = 3 * IMG_SIZE
    total_height = rows * IMG_SIZE
    
    fig = Image.new('RGB', (total_width, total_height), color='white')
    
    for row_idx, item in enumerate(items):
        y = row_idx * IMG_SIZE
        
        ref_path = item['ref_path']
        lora_path = os.path.join(RESULT_DIR, item['generated'])
        prompt = item['prompt']
        cat_id = int(item['ref'].split('_')[0])
        
        print(f"  Row {row_idx}: Category {cat_id}")
        print(f"    Ref: {ref_path} (exists: {os.path.exists(ref_path) if ref_path else False})")
        print(f"    Generated: {lora_path} (exists: {os.path.exists(lora_path)})")
        
        if ref_path and os.path.exists(ref_path):
            ref_img = Image.open(ref_path).convert('RGB')
            ref_img = ref_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        else:
            print(f"    Warning: Ref image not found!")
            ref_img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='gray')
        
        if os.path.exists(lora_path):
            lora_img = Image.open(lora_path).convert('RGB')
            lora_img = lora_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        else:
            print(f"    Warning: Generated image not found!")
            lora_img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='gray')
        
        prompt_img = create_prompt_image(prompt, IMG_SIZE, IMG_SIZE)
        prompt_img_rgb = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white')
        prompt_img_rgb.paste(prompt_img, mask=prompt_img)
        
        x_offset = 0
        fig.paste(ref_img, (x_offset, y))
        x_offset += IMG_SIZE
        fig.paste(prompt_img_rgb, (x_offset, y))
        x_offset += IMG_SIZE
        fig.paste(lora_img, (x_offset, y))
    
    fig.save(output_path, quality=95)
    print(f"Saved: {output_path}")


def main():
    print("Parsing generation_info.txt...")
    category_images = parse_log_file()
    
    total_phase3 = sum(len(imgs) for imgs in category_images.values())
    print(f"Found {total_phase3} Phase 3 images across {len(TAIL_CATEGORIES)} categories")
    
    print("\nSelecting one random image per category...")
    selected = select_one_per_category(category_images)
    print(f"Selected {len(selected)} categories")
    
    if len(selected) == 0:
        print("Error: No images selected!")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for i in range(4):
        start_idx = i * 4
        end_idx = start_idx + 4
        chunk = selected[start_idx:end_idx]
        
        if len(chunk) == 0:
            continue
        
        print(f"\nCreating figure {i+1} with {len(chunk)} rows...")
        output_path = os.path.join(OUTPUT_DIR, f"qualitative_results_part{i+1}.png")
        create_figure(chunk, output_path)
    
    print(f"\nAll 4 figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
