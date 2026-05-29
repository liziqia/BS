import numpy as np
from PIL import Image
import os

SRC_IMAGE = "../data/original/1/1_3_30_10961.jpg"
OUTPUT_DIR = "../diffusion_figure"

SIGMA_LEVELS = [0.15, 0.30, 0.50, 0.70, 1.00, 1.50]

os.makedirs(OUTPUT_DIR, exist_ok=True)

original = Image.open(SRC_IMAGE).convert('RGB')
arr = np.array(original).astype(np.float32) / 255.0

original.save(os.path.join(OUTPUT_DIR, "0_original.jpg"))
print(f"已保存 原图")

for sigma in SIGMA_LEVELS:
    noise = np.random.randn(*arr.shape).astype(np.float32) * sigma
    noisy_arr = np.clip(arr + noise, 0.0, 1.0)
    noisy_img = Image.fromarray((noisy_arr * 255).astype(np.uint8))
    noisy_img.save(os.path.join(OUTPUT_DIR, f"noisy_sigma_{sigma:.2f}.jpg"))
    print(f"已保存 σ={sigma:.2f}")

pure_noise = np.random.randn(*arr.shape).astype(np.float32) * 0.5 + 0.5
pure_noise = np.clip(pure_noise, 0.0, 1.0)
pure_img = Image.fromarray((pure_noise * 255).astype(np.uint8))
pure_img.save(os.path.join(OUTPUT_DIR, "pure_noise.jpg"))
print(f"已保存 纯噪声图像")

print(f"\n全部图像已保存至 {os.path.abspath(OUTPUT_DIR)}")
