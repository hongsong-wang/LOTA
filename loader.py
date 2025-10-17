import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import cv2
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

MODEL_NAME_MAP = {
    0: 'BigGAN',
    1: 'Midjourney',
    2: 'Wukong',
    3: 'Stable_Diffusion_v1.4',
    4: 'Stable_Diffusion_v1.5',
    5: 'ADM',
    6: 'GLIDE',
    7: 'VQDM'
}

def create_preprocessing_pipeline(options):
    if options.isPatch:
        transform_func = transforms.Lambda(
            lambda img: bit_patch_process(
                img, options.img_height, options.bit_mode,
                options.patch_size, options.patch_mode
            )
        )
    else:
        transform_func = transforms.Resize((256, 256))

    return transforms.Compose([
        transform_func,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def apply_preprocessing(image, options):
    pipeline = create_preprocessing_pipeline(options)
    return pipeline(image)


class GenerativeImageTrainingSet(Dataset):
    def __init__(self, root_dir, dataset_name, options):
        super().__init__()
        self.options = options
        self.base_path = os.path.join(root_dir, dataset_name, "train")

        self.natural_images = self._load_images("nature")
        self.ai_images = self._load_images("ai")

        self.all_images = self.natural_images + self.ai_images
        self.labels = torch.cat([
            torch.ones(len(self.natural_images)),
            torch.zeros(len(self.ai_images))
        ])

    def _load_images(self, category):
        category_path = os.path.join(self.base_path, category)
        return [os.path.join(category_path, f) for f in os.listdir(category_path)]

    def _load_rgb(self, img_path):
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except Exception as e:
            print(f"Image Loading Error {img_path}: {str(e)}")
            return Image.new('RGB', (256, 256), (0, 0, 0))

    def __getitem__(self, index):
        try:
            img = self._load_rgb(self.all_images[index])
            label = self.labels[index]
        except:
            prev_index = max(0, index - 1)
            img = self._load_rgb(self.all_images[prev_index])
            label = self.labels[prev_index]

        processed_img = apply_preprocessing(img, self.options)
        return processed_img, label

    def __len__(self):
        return len(self.all_images)

class GenerativeImageValidationSet(Dataset):
    def __init__(self, root_dir, dataset_name, is_natural, options):
        super().__init__()
        self.options = options
        self.base_path = os.path.join(root_dir, dataset_name, "val")

        category = "nature" if is_natural else "ai"
        self.img_dir = os.path.join(self.base_path, category)
        self.image_paths = [os.path.join(self.img_dir, f)
                            for f in os.listdir(self.img_dir)]

        self.labels = torch.ones(len(self.image_paths)) if is_natural else torch.zeros(len(self.image_paths))

    def _load_rgb(self, img_path):
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except Exception as e:
            print(f"Val Image Loading Error {img_path}: {str(e)}")
            return Image.new('RGB', (256, 256), (0, 0, 0))

    def __getitem__(self, index):
        img = self._load_rgb(self.image_paths[index])
        label = self.labels[index]

        processed_img = apply_preprocessing(img, self.options)
        return processed_img, label

    def __len__(self):
        return len(self.image_paths)

def create_validation_loader(options, dataset_name, is_natural):
    val_dataset = GenerativeImageValidationSet(
        options.image_root, dataset_name, is_natural, options
    )

    def collate_batch(batch):
        inputs = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return inputs, labels

    return DataLoader(
        val_dataset,
        batch_size=options.val_batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_batch
    ), len(val_dataset)


def setup_validation_loaders(options):
    choices = options.choices
    loaders = []

    for idx, selected in enumerate(choices):
        if selected:
            loader_info = {}
            model_name = MODEL_NAME_MAP[idx]
            print(f"Val dataset: {model_name}")

            loader_info['name'] = model_name
            loader_info['val_ai_loader'], loader_info['ai_size'] = create_validation_loader(
                options, model_name, False
            )
            loader_info['val_nature_loader'], loader_info['nature_size'] = create_validation_loader(
                options, model_name, True
            )

            loaders.append(loader_info)

    return loaders

def create_training_loader(options):
    choices = options.choices
    root_dir = options.image_root

    datasets = []

    dataset_config = [
        (0, "BigGAN"),
        (1, "Midjourney"),
        (2, "Wukong"),
        (3, "Stable_Diffusion_v1.4"),
        (4, "Stable_Diffusion_v1.5"),
        (5, "ADM"),
        (6, "GLIDE"),
        (7, "VQDM")
    ]

    for idx, folder_name in dataset_config:
        if choices[idx]:
            dataset = GenerativeImageTrainingSet(
                root_dir, folder_name, options
            )
            datasets.append(dataset)
            print(f"Train dataset: {MODEL_NAME_MAP[idx]}")

    combined_dataset = torch.utils.data.ConcatDataset(datasets)

    return DataLoader(
        combined_dataset,
        batch_size=options.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

def get_loader(opt):
    return create_training_loader(opt)

def get_val_loader(opt):
    return setup_validation_loaders(opt)

def get_single_loader(opt, image_dir, is_real):
    return create_validation_loader(opt, image_dir, is_real)

from bit_patch import bit_patch as bit_patch_process