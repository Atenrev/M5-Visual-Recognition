import os
import glob
from PIL import Image
from typing import Any, Tuple, List, Optional

import kornia.augmentation as K
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor

from src.common.sample import Sample
from src.datasets.base_dataset import BaseDataset


class MITSplitDataset(BaseDataset):

    def __init__(self,
                 classes_paths: List[str],
                 device: torch.device,
                 config: Any,
                 transform: Optional[Any] = None,
                 ):
        super().__init__(device, config)
        self.transform = transform
        self.classes_paths = classes_paths
        self.image_paths = []
        self.targets = []
        self.labels = []

        for i, class_path in enumerate(classes_paths):
            for image_path in glob.glob(os.path.join(class_path, "*.jpg")):
                self.image_paths.append(image_path)
                self.targets.append(i)
                self.labels.append(os.path.basename(class_path))

    def __len__(self):
        return len(self.image_paths)

    def getitem(self, idx: int) -> Sample:
        image_path = self.image_paths[idx]
        sample_id = os.path.basename(image_path)

        image = Image.open(image_path).convert("RGB")
        image = pil_to_tensor(image).float() / 255

        if self.transform:
            image = self.transform(image).squeeze()

        return Sample(sample_id, {
            "image": image,
            "target": self.targets[idx],
            "label": self.labels[idx],
        }).to(self.device)


def create_dataloader(
    batch_size: int,
    dataset_path: str,
    device: torch.device,
    config: Any,
    inference: bool = False,
    dataset_kwargs: dict = {},
) -> Tuple[DataLoader[Any], Optional[DataLoader[Any]], Optional[DataLoader[Any]]]:
    train_dirs = glob.glob(os.path.join(dataset_path, "train/*"))
    train_dirs.sort()
    test_dirs = glob.glob(os.path.join(dataset_path, "test/*"))
    test_dirs.sort()

    transforms = [K.Resize(config.input_resize), K.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240,
                                                                                                 0.2250]), ]
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
