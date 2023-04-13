import os

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import kornia.augmentation as K
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor

from typing import Any


def create_coco_dataloader(
    batch_size: int,
    dataset_path: str,
    config: Any,
    inference: bool = False,
    dataset_kwargs: dict = {},
):
    transforms = [
        K.Resize((config.input_resize, config.input_resize)),
        K.Normalize(
            mean=[0.4850, 0.4560, 0.4060],
            std=[0.2290, 0.2240, 0.2250]),
    ]
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

    train_dataset = CocoDetection(
        root=os.path.join(dataset_path, "train2014"),
        annFile=os.path.join(dataset_path, "instances_train2014.json"),
        transform=ToTensor(),
    )
    test_dataset = CocoDetection(
        root=os.path.join(dataset_path, "val2014"),
        annFile=os.path.join(dataset_path, "instances_val2014.json"),
        transform=ToTensor(),
    )

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    if not inference:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
    else:
        train_dataset = test_dataset
        test_dataloader = test_dataset = None
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

    return train_dataloader, test_dataloader
