import torch
import glob
import os
import torchvision.transforms as T

from PIL import Image
from typing import Any, Tuple, List, Optional
from torch.utils.data import DataLoader

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

        if self.transform:
            image = self.transform(image)

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

    if config.transforms and config.transforms.resize:
        transforms = T.Compose([
                    T.Resize(299),
                    T.CenterCrop(299),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dataset_kwargs["transform"] = transforms
    
    train_dataset = MITSplitDataset(train_dirs, device, config, **dataset_kwargs)
    test_dataset = MITSplitDataset(test_dirs, device, config, **dataset_kwargs)

    if not inference:
        # Split the dataset into train, validation, and test
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size])

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
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
        val_dataloader = test_dataloader = None
        train_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    return train_dataloader, val_dataloader, test_dataloader
