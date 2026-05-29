#!/usr/bin/env python3
"""
JSON生成脚本：根据提示词和图像名称生成训练数据
包括：旋转、掩码、颜色抖动、i2i任务
"""

import os
import json
import random
import shutil
from PIL import Image, ImageEnhance
import numpy as np

# 类别映射
SHIP_CLASSES = {
    0: "Amphibious Assault Ship",
    1: "Nimitz-class Aircraft Carrier",
    2: "Arleigh Burke-class Destroyer",
    3: "Hyūga-class Helicopter Destroyer",
    4: "Yamato-class Battleship",
    5: "Blue Ridge-class Command Ship",
    6: "Type 075 Amphibious Assault Ship",
    7: "Wasp-class Amphibious Assault Ship",
    8: "America-class Amphibious Assault Ship",
    9: "San Antonio-class Amphibious Transport Dock",
    10: "Virginia-class Nuclear Submarine",
    11: "Mercy-class Hospital Ship",
    12: "Gepard-class Frigate",
    13: "Type 054A Frigate",
    14: "Container Ship",
    15: "Roll-on/Roll-off Ship",
    16: "Bridge Construction Vessel",
    17: "Semi-submersible Ship",
    18: "Oil Tanker",
    19: "Bulk Carrier",
    20: "Air-Cushioned Landing Craft",
    21: "Liquefied Natural Gas Carrier",
    22: "Ultra-large Container Ship"
}

# 路径配置
TRAIN_DIR = "data/images/FGSC/train"
REFERENCE_DIR = "data/images/FGSC/reference"
TARGET_DIR = "data/images/FGSC/target"
OUTPUT_JSON = "data/annotations/train.json"
FG_TRAIN_TXT = "data/images/FGSC/anno/FG_train.txt"

def ensure_dir(dir_path):
    """确保目录存在"""
    os.makedirs(dir_path, exist_ok=True)

def copy_image(src, dst):
    """复制图像"""
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)

def rotate_image(image_path, angle):
    """旋转图像（顺时针）"""
    img = Image.open(image_path)
    rotated = img.rotate(-angle, expand=True)
    return rotated

def apply_mask(image_path, grid_size, mask_ratio):
    """应用随机掩码"""
    img = Image.open(image_path)
    width, height = img.size
    grid_w = width // grid_size
    grid_h = height // grid_size
    
    mask = np.ones((grid_size, grid_size), dtype=np.float32)
    num_masked = int(grid_size * grid_size * mask_ratio)
    indices = np.random.choice(grid_size * grid_size, num_masked, replace=False)
    mask.flat[indices] = 0
    
    img_array = np.array(img)
    mask_expanded = np.repeat(np.repeat(mask, grid_h, axis=0), grid_w, axis=1)
    mask_expanded = np.tile(mask_expanded[:, :, np.newaxis], (1, 1, 3))
    
    masked_array = (img_array * mask_expanded).astype(np.uint8)
    return Image.fromarray(masked_array)

def apply_color_jitter(image_path, brightness=0.3, contrast=0.3, saturation=0.3):
    """应用颜色抖动"""
    img = Image.open(image_path)
    
    factor = random.uniform(1 - brightness, 1 + brightness)
    img = ImageEnhance.Brightness(img).enhance(factor)
    
    factor = random.uniform(1 - contrast, 1 + contrast)
    img = ImageEnhance.Contrast(img).enhance(factor)
    
    factor = random.uniform(1 - saturation, 1 + saturation)
    img = ImageEnhance.Color(img).enhance(factor)
    
    return img

def save_processed_image(image, output_path):
    """保存处理后的图像"""
    ensure_dir(os.path.dirname(output_path))
    image.save(output_path)

def create_sample_entry(instruction, input_images, output_image):
    """创建样本条目"""
    return {
        "instruction": instruction,
        "input_images": input_images,
        "output_image": output_image
    }

# ============================================================================
# 示例样本定义（5个）
# ============================================================================
def create_example_samples():
    """创建5个示例样本（根据prompts.md定义）"""
    samples = []
    
    # 示例1: 旋转90度（prompts.md开头定义）
    src = os.path.join(TRAIN_DIR, "1", "1_1_122_11386.jpg")
    ref = os.path.join(REFERENCE_DIR, "1", "1_1_122_11386_turned90.jpg")
    tgt = os.path.join(TARGET_DIR, "1", "1_1_122_11386.jpg")
    if os.path.exists(src):
        rotated = rotate_image(src, 90)
        save_processed_image(rotated, ref)
        copy_image(src, tgt)
    samples.append(create_sample_entry(
        "This is a remote sensing image of a Nimitz-class Aircraft Carrier. Generate a view of this ship rotated counterclockwise by 90 degrees. <|image_1|>",
        ["reference/1/1_1_122_11386_turned90.jpg"],
        "target/1/1_1_122_11386.jpg"
    ))
    
    # 示例2: i2i任务（prompts.md开头定义）
    src1 = os.path.join(TRAIN_DIR, "1", "1_3_7_12568.jpg")
    src2 = os.path.join(TRAIN_DIR, "1", "1_3_11_11039.jpg")
    ref = os.path.join(REFERENCE_DIR, "1", "1_3_7_12568.jpg")
    tgt = os.path.join(TARGET_DIR, "1", "1_3_11_11039.jpg")
    if os.path.exists(src1):
        copy_image(src1, ref)
    if os.path.exists(src2):
        copy_image(src2, tgt)
    samples.append(create_sample_entry(
        "This is a remote sensing image of a Nimitz-class Aircraft Carrier. Generate a different view of this ship with the bow facing the opposite direction and brighter lighting. <|image_1|>",
        ["reference/1/1_3_7_12568.jpg"],
        "target/1/1_3_11_11039.jpg"
    ))
    
    # 示例3: i2i任务（类别1第1条）
    src1 = os.path.join(TRAIN_DIR, "1", "1_3_141_10828.jpg")
    src2 = os.path.join(TRAIN_DIR, "1", "1_3_141_10829.jpg")
    ref = os.path.join(REFERENCE_DIR, "1", "1_3_141_10828.jpg")
    tgt = os.path.join(TARGET_DIR, "1", "1_3_141_10829.jpg")
    if os.path.exists(src1):
        copy_image(src1, ref)
    if os.path.exists(src2):
        copy_image(src2, tgt)
    samples.append(create_sample_entry(
        "This is a remote sensing image of a Nimitz-class Aircraft Carrier. Two ships are visible in the image, with one fully displayed in the center and another partially visible in the bottom-right corner. Generate a diverse image with a lower viewing angle so that the ship originally in the bottom-right corner appears in the center. <|image_1|>",
        ["reference/1/1_3_141_10828.jpg"],
        "target/1/1_3_141_10829.jpg"
    ))
    
    # 示例4: i2i任务（类别1第2条）
    src1 = os.path.join(TRAIN_DIR, "1", "1_3_162_10720.jpg")
    src2 = os.path.join(TRAIN_DIR, "1", "1_3_163_10721.jpg")
    ref = os.path.join(REFERENCE_DIR, "1", "1_3_162_10720.jpg")
    tgt = os.path.join(TARGET_DIR, "1", "1_3_163_10721.jpg")
    if os.path.exists(src1):
        copy_image(src1, ref)
    if os.path.exists(src2):
        copy_image(src2, tgt)
    samples.append(create_sample_entry(
        "This is a remote sensing image of a Nimitz-class Aircraft Carrier. Two ships are visible in the image, with one fully displayed in the center and another partially visible at the bottom. Generate a diverse image with a lower viewing angle so that the ship originally at the bottom appears in the center. <|image_1|>",
        ["reference/1/1_3_162_10720.jpg"],
        "target/1/1_3_163_10721.jpg"
    ))
    
    # 示例5: 掩码（类别1第3条）
    src = os.path.join(TRAIN_DIR, "1", "1_5_149_20123.jpg")
    ref = os.path.join(REFERENCE_DIR, "1", "1_5_149_20123_mask_5_16.jpg")
    tgt = os.path.join(TARGET_DIR, "1", "1_5_149_20123.jpg")
    if os.path.exists(src):
        masked = apply_mask(src, 4, 5/16)
        save_processed_image(masked, ref)
        copy_image(src, tgt)
    samples.append(create_sample_entry(
        "This is a remote sensing image of a Nimitz-class Aircraft Carrier with an unusual shape, having less distinct angles between the bow and hull. The input image has been partially masked. Generate the complete image of this ship. <|image_1|>",
        ["reference/1/1_5_149_20123_mask_5_16.jpg"],
        "target/1/1_5_149_20123.jpg"
    ))
    
    return samples

# ============================================================================
# i2i任务定义（69个）
# ============================================================================
def create_i2i_samples():
    """创建i2i任务样本"""
    samples = []
    
    i2i_tasks = [
        # 类别0
        {
            "instruction": "This is a remote sensing image of an Amphibious Assault Ship. Two rectangular ships are visible side by side with clean decks, and a dock is faintly visible in the upper-right corner. Generate a diverse image of this type of ship, keeping only one rectangular ship with a clean deck, docked near a pier. The dock should be a large open area with rectangular grid lines on both sides, located in the upper-right. <|image_1|>",
            "input": "0_2_54_13320",
            "output": "0_2_54_13476"
        },
        {
            "instruction": "This is a remote sensing image of an Amphibious Assault Ship with a purple deck. Based on this reference, generate a diverse ship image: remove the small white boat on the left side of the ship, remove the yellow-green rectangular ship or building in the bottom-right corner, and add a silver-gray dock on the right side of the ship with large yellow-brown buildings on it. Purple buildings should be visible in the bottom-right corner. The remaining area should be shimmering water. <|image_1|>",
            "input": "0_2_73_13253",
            "output": "0_2_74_13079"
        },
        {
            "instruction": "This is a remote sensing image of an Amphibious Assault Ship that appears to be under construction. Generate a diverse version of this image with a brighter deck and a dock on the left side. The ship should have three white spots at both the bow and stern, and three black shadow lines on each side, resembling a shadow line divided into three segments. The left shadow lines should be darker while the right ones lighter. The dock should have roads parallel to the shore and white stripes, appearing as if covered with a cyan-blue filter, with several small buildings and their shadows on it. <|image_1|>",
            "input": "0_2_91_13205",
            "output": "0_2_90_13676"
        },
        # 类别1
        {
            "instruction": "This is a remote sensing image of a Nimitz-class Aircraft Carrier. Generate a diverse aircraft carrier image, maintaining the ship's orientation but changing the image filter to make it look slightly old and rusty. The runway should be clearer. Add a white fighter jet with its nose pointing inward at the right side of the bow, near the corner between the bow and the hull. <|image_1|>",
            "input": "1_3_46_11291",
            "output": "1_3_44_10454"
        },
        {
            "instruction": "This is a remote sensing image of a Nimitz-class Aircraft Carrier. Based on the reference image, generate a clearer and more diverse ship image. Keep the scene exactly the same, but make the deck color and overall image color appear darker. A small portion of the dock near the aircraft carrier should be covered by the ship's shadow. <|image_1|>",
            "input": "1_3_65_12385",
            "output": "1_3_64_10490"
        },
        {
            "instruction": "This is a remote sensing image of a Nimitz-class Aircraft Carrier. Based on the reference image, generate a clearer and more diverse ship image. Keep the scene exactly the same, but make the water appear shimmering. Remove the two rows of small white buildings at the bow, and slightly reduce the number of helicopters on the right side of the ship. <|image_1|>",
            "input": "1_3_64_10490",
            "output": "1_3_66_12355"
        },
        # 类别2
        {
            "instruction": "This is a remote sensing image of an Arleigh Burke-class Destroyer. Based on the reference image, generate a diverse image of the same ship with reversed orientation. <|image_1|>",
            "input": "2_6_31_11179",
            "output": "2_6_31_11252"
        },
        {
            "instruction": "This is a remote sensing image of an Arleigh Burke-class Destroyer. Based on the reference image, generate a diverse image of the same ship with reversed bow direction, and the ship's right side should be close to a dock. The water surface should appear dark green. <|image_1|>",
            "input": "2_6_83_11152",
            "output": "2_6_83_10199"
        },
        {
            "instruction": "This is a remote sensing image of an Arleigh Burke-class Destroyer. Generate a clearer remote sensing image of this ship type. The characteristic features are white wireframe patterns at both the bow and stern: a square wireframe horizontal line at the bow, and two square wireframes at the stern, one containing a circular pattern. Unlike other assault ships or destroyers, the Arleigh Burke-class Destroyer has a diagonal line in the white wireframe at the stern. The ship's right side is a dock, and the remaining area is green water. <|image_1|>",
            "input": "2_6_81_12872",
            "output": "2_6_83_10199"
        },
        # 类别3
        {
            "instruction": "This is a remote sensing image of a Hyūga-class Helicopter Destroyer. Based on this image, generate a diverse ship image with simplified deck and dock buildings. The wireframe target pattern at the stern should include not only rectangular frames but also two circles and four diagonal lines crossing in pairs. <|image_1|>",
            "input": "3_5_49_12303",
            "output": "3_5_66_11563"
        },
        {
            "instruction": "This is a remote sensing image of a Hyūga-class Helicopter Destroyer. Two ships are visible in the image. Shift the viewing angle slightly downward to place the ship that is incompletely displayed at the bottom fully in the center of the image. <|image_1|>",
            "input": "3_5_36_11761",
            "output": "3_5_38_11764"
        },
        {
            "instruction": "This is a remote sensing image of a Hyūga-class Helicopter Destroyer. Generate a diverse image with reversed ship orientation. The ship's right side should be next to a narrow dock. <|image_1|>",
            "input": "3_6_124_12176",
            "output": "3_6_124_10546"
        },
        # 类别4
        {
            "instruction": "This is a remote sensing image of a Yamato-class Battleship. Based on this ship image, generate a diverse version. <|image_1|>",
            "input": "4_6_92_10946",
            "output": "4_6_112_11619"
        },
        {
            "instruction": "This is a remote sensing image of a Yamato-class Battleship. Based on this ship image, generate a diverse version with reversed ship orientation. <|image_1|>",
            "input": "4_7_6_10146",
            "output": "4_7_8_10149"
        },
        {
            "instruction": "This is a remote sensing image of a Yamato-class Battleship. Generate a diverse ship image. Remove the small white boat on the left side of the ship in the reference image, and dock another identical ship on the left side so that two parallel ships appear in the image. <|image_1|>",
            "input": "4_8_114_12031",
            "output": "4_8_114_12034"
        },
        # 类别5
        {
            "instruction": "This is a remote sensing image of a Blue Ridge-class Command Ship docked in a shipyard, close to a dock on the left side. Generate a diverse version where it is docked on the right side, close to the dock on the right. <|image_1|>",
            "input": "5_5_48_11732",
            "output": "5_5_49_10424"
        },
        {
            "instruction": "This is a remote sensing image of a Blue Ridge-class Command Ship. Generate a diverse version with reversed ship orientation. <|image_1|>",
            "input": "5_6_87_21370",
            "output": "5_6_88_21369"
        },
        {
            "instruction": "This is a remote sensing image of a Blue Ridge-class Command Ship. Generate a diverse version where the ship leaves the port and dock. One ship should appear in the center of the image with its bow pointing to the bottom-right, surrounded by water on all sides. An incomplete ship should be visible in the upper-right corner, oriented the same as the central ship. <|image_1|>",
            "input": "5_6_159_20370",
            "output": "5_7_42_20399"
        },
        # 类别6
        {
            "instruction": "This is a remote sensing image of a Type 075 Amphibious Assault Ship. Generate a diverse version of this ship image with a purple deck, making the dock narrower and the aspect ratio more consistent. <|image_1|>",
            "input": "6_7_152_12343",
            "output": "6_7_145_22692"
        },
        {
            "instruction": "This is a remote sensing image of a Type 075 Amphibious Assault Ship. Based on the ship features in the image, generate a diverse version with the bow pointing to the bottom-right, a wider dock, and the ship's left side close to the dock. Make the white wireframes at the bow, stern, and mid-ship more clear and prominent. <|image_1|>",
            "input": "6_8_34_11561",
            "output": "6_8_31_12633"
        },
        {
            "instruction": "This is a remote sensing image of a Type 075 Amphibious Assault Ship. Add another identical ship on the right side of the ship in the center of the image. <|image_1|>",
            "input": "6_9_34_11560",
            "output": "6_9_35_11564"
        },
        # 类别7
        {
            "instruction": "This is a remote sensing image of a Wasp-class Amphibious Assault Ship. Based on this ship image, generate a diverse ship image. The dock on the right side of the ship is orange-yellow, and the remaining area is green water. <|image_1|>",
            "input": "7_5_10_11334",
            "output": "7_5_24_11349"
        },
        {
            "instruction": "This is a remote sensing image of a Wasp-class Amphibious Assault Ship. Generate a diverse ship image, highlighting the white wireframe at the stern. <|image_1|>",
            "input": "7_5_78_12519",
            "output": "7_5_80_12413"
        },
        {
            "instruction": "This is a remote sensing image of a Wasp-class Amphibious Assault Ship. Generate a diverse ship image, highlighting the white wireframe at the stern. <|image_1|>",
            "input": "7_5_158_12935",
            "output": "7_5_156_10381"
        },
        # 类别8
        {
            "instruction": "This is a remote sensing image of an America-class Amphibious Assault Ship. Generate a diverse version with clearer ship deck, surrounded by water on all sides. <|image_1|>",
            "input": "8_5_86_12940",
            "output": "8_5_89_12325"
        },
        {
            "instruction": "This is a remote sensing image of an America-class Amphibious Assault Ship. Generate a diverse version based on the reference image. <|image_1|>",
            "input": "8_6_124_10747",
            "output": "8_6_134_11757"
        },
        {
            "instruction": "This is a remote sensing image of an America-class Amphibious Assault Ship. Generate a diverse version based on the reference image. <|image_1|>",
            "input": "8_3_105_12816",
            "output": "8_3_64_12772"
        },
        # 类别9
        {
            "instruction": "This is a remote sensing image of a San Antonio-class Amphibious Transport Dock. Generate a diverse image by removing the smaller ship in the upper-right corner. <|image_1|>",
            "input": "9_5_56_11733",
            "output": "9_5_56_11735"
        },
        {
            "instruction": "This is a remote sensing image of a San Antonio-class Amphibious Transport Dock. Based on this ship image, generate a diverse ship image with the same white wireframe and other ship features, but with the bow pointing downward and the dock on the right side of the hull. <|image_1|>",
            "input": "9_6_82_12414",
            "output": "9_6_82_12412"
        },
        {
            "instruction": "This is a remote sensing image of a San Antonio-class Amphibious Transport Dock. Based on this ship image, generate a diverse ship image with the same white wireframe and other ship features, but with the bow pointing upward and the dock on the left side of the hull. Three square white buildings are visible on the ship in the original image. Remove the two white buildings near the mid-ship section, keeping only the front one. <|image_1|>",
            "input": "9_6_87_11248",
            "output": "9_6_85_11634"
        },
        # 类别10
        {
            "instruction": "This is a remote sensing image of a Virginia-class Nuclear Submarine. A submarine is visible in the center of the image, with shore buildings in the bottom-left corner. Generate an image with a narrow dock and two such submarines on either side. One should be in the center of the image, and another partially visible in the bottom-left corner, oriented the same as the central ship. <|image_1|>",
            "input": "10_7_48_10169",
            "output": "10_7_48_10168"
        },
        {
            "instruction": "This is a remote sensing image of a Virginia-class Nuclear Submarine. A submarine is in the center of the image, with a dock in the upper-right corner. Based on this image, generate a diverse image. <|image_1|>",
            "input": "10_8_35_12731",
            "output": "10_8_35_12729"
        },
        {
            "instruction": "This is a remote sensing image of a Virginia-class Nuclear Submarine docked next to a narrow dock. Based on this image, generate a diverse image with a narrow dock and two such submarines docked on either side. <|image_1|>",
            "input": "10_11_125_22393",
            "output": "10_12_139_20534"
        },
        # 类别11
        {
            "instruction": "This is a remote sensing image of a Mercy-class Hospital Ship. Based on this image, generate a diverse version that makes the ship appear clearer, and increase the proportion of the narrow dock covered by the ship's shadow. <|image_1|>",
            "input": "11_7_83_12924",
            "output": "11_7_84_11839"
        },
        {
            "instruction": "This is a remote sensing image of a Mercy-class Hospital Ship. Based on this ship image, generate a diverse version, changing from a 'rear-right偏后' perspective to a 'rear-right偏右' perspective. Remove the other ship that is partially visible in the bottom-left corner. <|image_1|>",
            "input": "11_7_112_10576",
            "output": "11_7_156_11189"
        },
        {
            "instruction": "This is a remote sensing image of a Mercy-class Hospital Ship. Rotate the image 180 degrees for diverse image generation, removing unrelated ships. <|image_1|>",
            "input": "11_7_29_12347",
            "output": "11_7_28_12685"
        },
        # 类别12
        {
            "instruction": "This is a remote sensing image of a Gepard-class Frigate. Based on this image with multiple ships, generate a diverse image with more parallel ships of this type, filling the remaining water area. <|image_1|>",
            "input": "12_4_127_20378",
            "output": "12_4_127_20382"
        },
        {
            "instruction": "This is a remote sensing image of a Gepard-class Frigate. Generate a diverse image by creating another identical ship next to this one, parallel to it. <|image_1|>",
            "input": "12_4_128_22486",
            "output": "12_4_128_22488"
        },
        {
            "instruction": "This is a remote sensing image of a Gepard-class Frigate. Two ships are visible in the image. Generate a new image with the viewing angle shifted slightly to the bottom-left. <|image_1|>",
            "input": "12_3_32_21323",
            "output": "12_3_32_21322"
        },
        # 类别13
        {
            "instruction": "This is a remote sensing image of a Type 054A Frigate. Shift the viewing angle slightly to the left and generate a new image. <|image_1|>",
            "input": "13_3_54_20481",
            "output": "13_3_55_20480"
        },
        {
            "instruction": "This is a remote sensing image of a Type 054A Frigate. Based on the ship in the reference image, generate an image of this ship sailing independently at sea. <|image_1|>",
            "input": "13_4_140_21120",
            "output": "13_4_137_21644"
        },
        {
            "instruction": "This is a remote sensing image of a Type 054A Frigate. Generate a diverse image by shifting the viewing angle to the upper-left. <|image_1|>",
            "input": "13_5_140_21119",
            "output": "13_5_141_21118"
        },
        # 类别14
        {
            "instruction": "This is a remote sensing image of a Container Ship. Generate a diverse image based on this image. <|image_1|>",
            "input": "14_6_49_10616",
            "output": "14_6_45_11035"
        },
        {
            "instruction": "This is a remote sensing image of a Container Ship. Generate a diverse image based on this image. Remove the construction framework from the image. <|image_1|>",
            "input": "14_4_127_11330",
            "output": "14_4_127_11339"
        },
        {
            "instruction": "This is a remote sensing image of a Container Ship. Generate the reverse view of this image. <|image_1|>",
            "input": "14_5_106_11329",
            "output": "14_5_106_11347"
        },
        # 类别15
        {
            "instruction": "This is a remote sensing image of a Roll-on/Roll-off Ship. Generate the reverse view of this image. <|image_1|>",
            "input": "15_5_44_11054",
            "output": "15_5_46_11056"
        },
        {
            "instruction": "This is a remote sensing image of a Roll-on/Roll-off Ship. Generate a diverse image. <|image_1|>",
            "input": "15_4_35_11040",
            "output": "15_4_40_11073"
        },
        {
            "instruction": "This is a remote sensing image of a Roll-on/Roll-off Ship. Reduce the exposure level to make the image appear clearer. <|image_1|>",
            "input": "15_3_127_11043",
            "output": "15_3_131_11090"
        },
        # 类别16
        {
            "instruction": "This is a remote sensing image of a Bridge Construction Vessel. Simplify the image, keeping only the core ship and dock. <|image_1|>",
            "input": "16_2_3_12351",
            "output": "16_2_6_12277"
        },
        {
            "instruction": "This is a remote sensing image of a Bridge Construction Vessel. Generate an image of this ship sailing through the waves at sea. <|image_1|>",
            "input": "16_2_50_12323",
            "output": "16_2_53_12286"
        },
        {
            "instruction": "This is a remote sensing image of a Bridge Construction Vessel. Generate an image of this ship sailing through the waves at sea. <|image_1|>",
            "input": "16_3_177_12348",
            "output": "16_3_116_12293"
        },
        # 类别17
        {
            "instruction": "This is a remote sensing image of a Semi-submersible Ship. Shift the viewing angle slightly downward. <|image_1|>",
            "input": "17_4_11_10698",
            "output": "17_4_12_10918"
        },
        {
            "instruction": "This is a remote sensing image of a Semi-submersible Ship. Shift the viewing angle slightly to the upper-left. <|image_1|>",
            "input": "17_4_141_10559",
            "output": "17_4_140_10594"
        },
        {
            "instruction": "This is a remote sensing image of a Semi-submersible Ship. Shift the viewing angle slightly downward. <|image_1|>",
            "input": "17_5_160_10428",
            "output": "17_5_159_10372"
        },
        # 类别18
        {
            "instruction": "This is a remote sensing image of an Oil Tanker. Simplify the image, keeping only the core ship. <|image_1|>",
            "input": "18_5_0_100021",
            "output": "18_5_0_100014"
        },
        {
            "instruction": "This is a remote sensing image of an Oil Tanker. Remove the white buildings from the image. <|image_1|>",
            "input": "18_5_12_100040",
            "output": "18_5_15_100012"
        },
        {
            "instruction": "This is a remote sensing image of an Oil Tanker. Based on this image, generate a diverse version of the same type of ship but with a white deck. The stern should have a small red area, and the surrounding sea should appear blue. <|image_1|>",
            "input": "18_5_39_100028",
            "output": "18_5_35_11360"
        },
        # 类别19
        {
            "instruction": "This is a remote sensing image of a Bulk Carrier. Simplify the image. The generated image should be of the same type of ship but with a gray-white deck. <|image_1|>",
            "input": "19_3_93_11908",
            "output": "19_3_94_11867"
        },
        {
            "instruction": "This is a remote sensing image of a Bulk Carrier. Generate a diverse image based on this ship image. <|image_1|>",
            "input": "19_4_81_11873",
            "output": "19_4_76_11886"
        },
        {
            "instruction": "This is a remote sensing image of a Bulk Carrier. Generate a diverse image with a slightly lower viewing angle. <|image_1|>",
            "input": "19_4_163_11858",
            "output": "19_4_161_11884"
        },
        # 类别20
        {
            "instruction": "This is a remote sensing image of an Air-Cushioned Landing Craft. Simplify the image. <|image_1|>",
            "input": "20_3_89_11224",
            "output": "20_3_90_11268"
        },
        {
            "instruction": "This is a remote sensing image of an Air-Cushioned Landing Craft. Generate a diverse image with a slightly lower viewing angle. <|image_1|>",
            "input": "20_4_152_11222",
            "output": "20_4_154_11298"
        },
        {
            "instruction": "This is a remote sensing image of an Air-Cushioned Landing Craft. Shift the viewing angle slightly to the left. <|image_1|>",
            "input": "20_3_108_11234",
            "output": "20_3_107_11253"
        },
        # 类别21
        {
            "instruction": "This is a remote sensing image of a Liquefied Natural Gas Carrier. Generate a side view of this ship. <|image_1|>",
            "input": "21_5_105_100043",
            "output": "21_6_0_100005"
        },
        {
            "instruction": "This is a remote sensing image of a Liquefied Natural Gas Carrier. Generate an image of this ship sailing through the waves at sea. <|image_1|>",
            "input": "21_6_45_100082",
            "output": "21_6_50_100021"
        },
        {
            "instruction": "This is a remote sensing image of a Liquefied Natural Gas Carrier. Generate a front view image of this ship. <|image_1|>",
            "input": "21_6_170_100024",
            "output": "21_7_92_100071"
        },
        # 类别22
        {
            "instruction": "This is a remote sensing image of an Ultra-large Container Ship. Generate a diverse version of the reference image, making the ship appear brighter. <|image_1|>",
            "input": "22_6_120_100019",
            "output": "22_6_120_100049"
        },
        {
            "instruction": "This is a remote sensing image of an Ultra-large Container Ship. Generate a clearer image based on the reference image. <|image_1|>",
            "input": "22_5_125_100020",
            "output": "22_5_120_100046"
        },
        {
            "instruction": "This is a remote sensing image of an Ultra-large Container Ship. Make the ship in the image appear brighter. <|image_1|>",
            "input": "22_6_130_100030",
            "output": "22_6_130_100047"
        },
    ]
    
    for task in i2i_tasks:
        input_name = task["input"]
        output_name = task["output"]
        
        # 解析类别
        class_id = input_name.split("_")[0]
        
        # 源文件路径（图像存储在类别子目录中）
        src_input = os.path.join(TRAIN_DIR, class_id, f"{input_name}.jpg")
        src_output = os.path.join(TRAIN_DIR, class_id, f"{output_name}.jpg")
        
        # 目标文件路径
        dst_input = os.path.join(REFERENCE_DIR, class_id, f"{input_name}.jpg")
        dst_output = os.path.join(TARGET_DIR, class_id, f"{output_name}.jpg")
        
        # 复制图像
        if os.path.exists(src_input):
            copy_image(src_input, dst_input)
        else:
            print(f"警告: 找不到输入图像 {src_input}")
        
        if os.path.exists(src_output):
            copy_image(src_output, dst_output)
        else:
            print(f"警告: 找不到输出图像 {src_output}")
        
        # 创建 JSON 条目
        samples.append(create_sample_entry(
            task["instruction"],
            [f"reference/{class_id}/{input_name}.jpg"],
            f"target/{class_id}/{output_name}.jpg"
        ))
    
    return samples

# ============================================================================
# 批量生成简单任务
# ============================================================================
def create_batch_samples(train_images, samples_per_class=6):
    """批量生成简单任务样本"""
    samples = []
    
    class_images = {}
    for img_name in train_images:
        parts = img_name.split("_")
        if len(parts) >= 4:
            class_id = int(parts[0])
            if class_id not in class_images:
                class_images[class_id] = []
            class_images[class_id].append(img_name)
    
    for class_id in range(23):
        if class_id not in class_images:
            continue
        
        images = class_images[class_id]
        ship_name = SHIP_CLASSES[class_id]
        
        num_images = min(samples_per_class, len(images))
        step = len(images) // num_images
        selected_images = [images[i * step] for i in range(num_images)]
        
        for idx, img_name in enumerate(selected_images):
            task_type = idx % 3
            
            if task_type == 0:
                angle = random.choice([45, 90, 135, 180, 225, 270, 315])
                ref_name = f"{img_name}_turned{angle}"
                target_name = img_name
                
                src_path = os.path.join(TRAIN_DIR, f"{class_id}", f"{img_name}.jpg")
                ref_path = os.path.join(REFERENCE_DIR, f"{class_id}", f"{ref_name}.jpg")
                target_path = os.path.join(TARGET_DIR, f"{class_id}", f"{target_name}.jpg")
                
                if os.path.exists(src_path):
                    rotated = rotate_image(src_path, angle)
                    save_processed_image(rotated, ref_path)
                    copy_image(src_path, target_path)
                    
                    if angle == 90:
                        instruction = f"This is a remote sensing image of a {ship_name}. Generate a diverse view of this ship rotated counterclockwise by 90 degrees. <|image_1|>"
                    elif angle == 180:
                        instruction = f"This is a remote sensing image of a {ship_name}. Generate a diverse view of this ship rotated by 180 degrees. <|image_1|>"
                    else:
                        instruction = f"This is a remote sensing image of a {ship_name}. Generate a diverse view of this ship rotated counterclockwise by {angle} degrees. <|image_1|>"
                    
                    samples.append(create_sample_entry(
                        instruction,
                        [f"reference/{class_id}/{ref_name}.jpg"],
                        f"target/{class_id}/{target_name}.jpg"
                    ))
            
            elif task_type == 1:
                grid_size = random.choice([4, 8])
                mask_ratio = random.uniform(2/16, 8/16)
                mask_ratio_rounded = round(mask_ratio, 2)
                
                if grid_size == 4:
                    mask_name = f"{img_name}_mask_{int(mask_ratio*16)}_16"
                else:
                    mask_name = f"{img_name}_mask_{int(mask_ratio*64)}_64"
                
                target_name = img_name
                
                src_path = os.path.join(TRAIN_DIR, f"{class_id}", f"{img_name}.jpg")
                ref_path = os.path.join(REFERENCE_DIR, f"{class_id}", f"{mask_name}.jpg")
                target_path = os.path.join(TARGET_DIR, f"{class_id}", f"{target_name}.jpg")
                
                if os.path.exists(src_path):
                    masked = apply_mask(src_path, grid_size, mask_ratio_rounded)
                    save_processed_image(masked, ref_path)
                    copy_image(src_path, target_path)
                    
                    instruction = f"This is a remote sensing image of a {ship_name}. The input image has been partially masked with a {grid_size}x{grid_size} grid. Generate a diverse complete image based on this masked reference. <|image_1|>"
                    
                    samples.append(create_sample_entry(
                        instruction,
                        [f"reference/{class_id}/{mask_name}.jpg"],
                        f"target/{class_id}/{target_name}.jpg"
                    ))
            
            else:
                color_name = f"{img_name}_color"
                target_name = img_name
                
                src_path = os.path.join(TRAIN_DIR, f"{class_id}", f"{img_name}.jpg")
                ref_path = os.path.join(REFERENCE_DIR, f"{class_id}", f"{color_name}.jpg")
                target_path = os.path.join(TARGET_DIR, f"{class_id}", f"{target_name}.jpg")
                
                if os.path.exists(src_path):
                    jittered = apply_color_jitter(src_path)
                    save_processed_image(jittered, ref_path)
                    copy_image(src_path, target_path)
                    
                    instruction = f"This is a remote sensing image of a {ship_name}. Based on this image with slight color variation, generate a diverse image of this ship. <|image_1|>"
                    
                    samples.append(create_sample_entry(
                        instruction,
                        [f"reference/{class_id}/{color_name}.jpg"],
                        f"target/{class_id}/{target_name}.jpg"
                    ))
    
    return samples

# ============================================================================
# 主函数
# ============================================================================
def main():
    """主函数"""
    print("开始生成训练数据...")
    
    ensure_dir(REFERENCE_DIR)
    ensure_dir(TARGET_DIR)
    
    with open(FG_TRAIN_TXT, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        # 解析格式: "类别/图像名.jpg 标签"
        train_images = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 1:
                # 提取图像名称（去掉类别前缀和.jpg后缀）
                img_path = parts[0]
                img_name = img_path.split("/")[-1].replace(".jpg", "")
                train_images.append(img_name)
    
    print(f"找到 {len(train_images)} 张训练图像")
    
    all_samples = []
    
    print("创建示例样本...")
    all_samples.extend(create_example_samples())
    
    print("创建i2i任务样本...")
    all_samples.extend(create_i2i_samples())
    
    print("创建批量简单任务样本...")
    all_samples.extend(create_batch_samples(train_images, samples_per_class=6))
    
    print(f"保存JSON文件到 {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    print(f"共生成 {len(all_samples)} 个样本")
    print("完成！")

if __name__ == "__main__":
    main()
