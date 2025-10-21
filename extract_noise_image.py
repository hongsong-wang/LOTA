"""
Author: Hongsong Wang
Contact: hongsongwang@seu.edu.cn
Date: 2025-10-21
Description: Process a color image per B, G, R channel using the lowest 3 bit-planes.
"""
import cv2
import numpy as np
import os
from tqdm import tqdm

def process_single_image(img):
    # Ensure input is a numpy array with uint8
    img = np.asarray(img, dtype=np.uint8)
    # Validate shape
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected color image with shape (H, W, 3), got {img.shape}")

    # Extract the lowest 3 bits for each channel (bit planes 0..2)
    low3 = img & 0b111  # shape (H, W, 3), values 0..7
    # For each channel, check if any of the low-3 bits is non-zero -> boolean mask
    mask = (low3 != 0).astype(np.uint8)  # 0 or 1, shape (H, W, 3)
    # Scale to 0 or 255 for a displayable binary image per channel
    result = mask * 255  # dtype uint8
    return result

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]

    for fname in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        img = cv2.imread(input_path)
        if img is None:
            print(f"Skip: {fname}")
            continue

        result = process_single_image(img)

        cv2.imwrite(output_path, result)

if __name__ == "__main__":
    input_dir = "D:/BaiduNetdiskDownload/Image/AI-generated"
    output_dir = "D:/BaiduNetdiskDownload/Image/output"
    process_folder(input_dir, output_dir)
