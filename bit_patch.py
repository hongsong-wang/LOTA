import numpy as np
import cv2
import cv2 as cv
from PIL import Image
import random
from torchvision.transforms import transforms

def compute(patch):
    patch = np.array(patch).astype(np.int64)
    diff_horizontal = np.sum(np.abs(patch[:, :-1, :] - patch[:, 1:, :]))
    diff_vertical = np.sum(np.abs(patch[:-1, :, :] - patch[1:, :, :]))
    diff_diagonal = np.sum(np.abs(patch[:-1, :-1, :] - patch[1:, 1:, :]))
    diff_diagonal += np.sum(np.abs(patch[1:, :-1, :] - patch[:-1, 1:, :]))
    res = diff_horizontal + diff_vertical + diff_diagonal
    return res.sum()

'''
    low_three_bits = img_np & 0x07

    # 将每个通道的0-7值映射到0-255
    mapped_r = (low_three_bits[:, :, 0] * 255 // 7).astype(np.uint8)
    mapped_g = (low_three_bits[:, :, 1] * 255 // 7).astype(np.uint8)
    mapped_b = (low_three_bits[:, :, 2] * 255 // 7).astype(np.uint8)

    # 组合成三通道图像
    mapped_img = np.stack([mapped_r, mapped_g, mapped_b], axis=-1)
'''


def bit_patch(img, img_height, bit_mode, patch_size, patch_mode):
    img_np = np.array(img)
    #combined_image = None
    if bit_mode == "scaling":
        mask_low = 0x07
        red_low3 = ((img_np[:, :, 0] & mask_low) * (255 // 7)).astype(np.uint8)
        green_low3 = ((img_np[:, :, 1] & mask_low) * (255 // 7)).astype(np.uint8)
        blue_low3 = ((img_np[:, :, 2] & mask_low) * (255 // 7)).astype(np.uint8)
        combined_image = cv.merge((red_low3, green_low3, blue_low3))
    elif bit_mode == "thresholding":
        mask_low3 = 0x07
        red_channel = img_np[:, :, 0]
        green_channel = img_np[:, :, 1]
        blue_channel = img_np[:, :, 2]
        red_low3 = ((red_channel & mask_low3) != 0).astype(np.uint8) * 255
        green_low3 = ((green_channel & mask_low3) != 0).astype(np.uint8) * 255
        blue_low3 = ((blue_channel & mask_low3) != 0).astype(np.uint8) * 255
        combined_image = cv.merge((red_low3, green_low3, blue_low3))
    else:
        raise ValueError(f"Unsupported bit_mode: {bit_mode}")
    h, w, _ = combined_image.shape
    combined_image = Image.fromarray(combined_image)
    min_len = min(h, w)
    rz = transforms.Resize((img_height, img_height))
    if min_len < patch_size:
        combined_image = rz(combined_image)
    num_patch = (img_height // patch_size) ** 2
    patch_list = []
    rp = transforms.RandomCrop(patch_size)
    for _ in range(num_patch):
        patch_list.append(rp(combined_image))

    if patch_mode == "max":
        patch_list.sort(key=lambda x: compute(x), reverse=True)
        selected_patch = patch_list[0]
    elif patch_mode == "min":
        patch_list.sort(key=lambda x: compute(x), reverse=False)
        selected_patch = patch_list[0]
    else:
        selected_patch = random.choice(patch_list)

    patch_np = np.array(selected_patch)
    return cv2.resize(patch_np, (img_height, img_height))