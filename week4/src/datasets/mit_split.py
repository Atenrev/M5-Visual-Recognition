import os
import glob
import numpy as np
import torch
import kornia.augmentation as K

from PIL import Image

from typing import List, Any, Optional
from torch.utils.data import Dataset, DataLoader


class MITSplitDataset(Dataset):

    def __init__(self,
                 classes_paths: List[str],
                 transform: Optional[Any] = None,
                 ):
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

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert("RGB")).transpose(2, 0, 1)
        image = torch.tensor(image).float() / 255

        if self.transform:
            image = self.transform(image).squeeze()

        return image, self.targets[idx]
    

def create_mit_dataloader(
    batch_size: int,
    dataset_path: str,
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

    train_dataset = MITSplitDataset(train_dirs, **train_dataset_kwargs)
    test_dataset = MITSplitDataset(test_dirs, **val_dataset_kwargs)

    if not inference:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        train_dataset = test_dataset
        test_dataloader = test_dataset = None
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    return train_dataloader, test_dataloader