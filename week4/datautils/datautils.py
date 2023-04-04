
import os
import cv2
import glob
from PIL import Image

import zipfile
import torch

from typing import *
from torch.utils.data import Dataset
import kornia.augmentation as K
from torch.utils.data import DataLoader

from datautils.datasets import MITSplitDataset



def create_mit_dataloader(
    batch_size: int,
    dataset_path: str,
    device: torch.device,
    config: Any,
    inference: bool = False,
    dataset_kwargs: dict = {},
):
    train_dirs = glob.glob(os.path.join(dataset_path, "train/*"))
    train_dirs.sort()
    test_dirs = glob.glob(os.path.join(dataset_path, "test/*"))
    test_dirs.sort()

    transforms = [K.Resize(config.input_resize), K.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240,
                                                                                                 0.2250]), ]
    # transforms = []
    train_transforms = list(transforms)

    if "transforms" in config:
        train_transforms += [
            K.RandomBrightness(
                brightness=(config.transforms.brightness_min, config.transforms.brightness_max),
                p=0.05,
            ),
            K.RandomAffine(
                degrees=config.transforms.rotation,
                translate=config.transforms.translate,
                scale=config.transforms.scale,
                shear=config.transforms.shear,
                p=0.05,
            ),
            K.RandomHorizontalFlip(p=0.05),
            K.RandomVerticalFlip(p=0.05),
        ]

    train_dataset_kwargs = dataset_kwargs.copy()
    train_dataset_kwargs["transform"] = K.AugmentationSequential(
        *train_transforms,
        data_keys=["input"]
    )
    val_dataset_kwargs = dataset_kwargs.copy()
    val_dataset_kwargs["transform"] = K.AugmentationSequential(
        *transforms,
        data_keys=["input"]
    )

    train_dataset = MITSplitDataset(train_dirs, device, config, **train_dataset_kwargs)
    test_dataset = MITSplitDataset(test_dirs, device, config, **val_dataset_kwargs)

    if not inference:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_dataloader = test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        val_dataloader = test_dataloader = None
        train_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    return train_dataloader, val_dataloader, test_dataloader

def return_image_full_range(image):
    return torch.clamp(K.Normalize(mean=[-0.4850, -0.4560, -0.4060], std=[1/0.2290, 1/0.2240, 1/0.2250])(image), min = 0, max = 1) * 255