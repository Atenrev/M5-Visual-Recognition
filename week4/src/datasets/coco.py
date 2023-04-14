import os
# from collections import defaultdict
# from pathlib import Path
# import json
# from PIL import Image
#
# import PIL.Image

import torch
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset

from torchvision.datasets import CocoDetection

import torchvision.transforms.v2 as transforms

from torchvision import datasets

from typing import Any


# class CocoDataset(Dataset):
#     """PyTorch dataset for COCO annotations."""
#
#     def __init__(self, data_dir, transforms=None):
#         """Load COCO annotation data."""
#         self.data_dir = Path(data_dir)
#         self.transforms = transforms
#
#         # load the COCO annotations json
#         anno_file_path = self.data_dir/f'../instances_{os.path.basename(self.data_dir)}.json'
#         with open(str(anno_file_path)) as file_obj:
#             self.coco_data = json.load(file_obj)
#         # put all of the annos into a dict where keys are image IDs to speed up retrieval
#         self.image_id_to_annos = defaultdict(list)
#         for anno in self.coco_data['annotations']:
#             image_id = anno['image_id']
#             self.image_id_to_annos[image_id] += [anno]
#
#     def __len__(self):
#         return len(self.coco_data['images'])
#
#     def __getitem__(self, index):
#         """Return tuple of image and labels as torch tensors."""
#         image_data = self.coco_data['images'][index]
#         image_id = image_data['id']
#         if image_id == 71:
#             print('debug')
#         image_path = self.data_dir/image_data['file_name']
#         image = Image.open(image_path)
#
#         annos = self.image_id_to_annos[image_id]
#         anno_data = {
#             'boxes': [],
#             'labels': [],
#             'area': [],
#             'iscrowd': [],
#         }
#         for anno in annos:
#             coco_bbox = anno['bbox']
#             left = coco_bbox[0]
#             top = coco_bbox[1]
#             right = coco_bbox[0] + coco_bbox[2]
#             bottom = coco_bbox[1] + coco_bbox[3]
#             area = coco_bbox[2] * coco_bbox[3]
#             anno_data['boxes'].append([left, top, right, bottom])
#             anno_data['labels'].append(anno['category_id'])
#             anno_data['area'].append(area)
#             anno_data['iscrowd'].append(anno['iscrowd'])
#
#         target = {
#             'boxes': torch.as_tensor(anno_data['boxes'], dtype=torch.float32),
#             'labels': torch.as_tensor(anno_data['labels'], dtype=torch.int64),
#             'image_id': torch.tensor([image_id]),  # pylint: disable=not-callable (false alarm)
#             'area': torch.as_tensor(anno_data['area'], dtype=torch.float32),
#             'iscrowd': torch.as_tensor(anno_data['iscrowd'], dtype=torch.int64),
#         }
#
#         width, height = image.size
#
#         if self.transforms is not None:
#             image = self.transforms(image)
#
#             target['boxes'][:, 0::2] /= width
#             target['boxes'][:, 1::2] /= height
#
#         return image, target


def create_coco_dataloader(
    batch_size: int,
    dataset_path: str,
    config: Any,
    inference: bool = False,
    dataset_kwargs: dict = {},
):
    transform = transforms.Compose(
        [
         transforms.Resize((config.input_resize, config.input_resize)),
         transforms.ToImageTensor(),
         transforms.ConvertImageDtype(torch.float32),
         transforms.Normalize(
             mean=[0.4850, 0.4560, 0.4060],
             std=[0.2290, 0.2240, 0.2250]),
         ])

    train_dataset = CocoDetection(
        root=os.path.join(dataset_path, "train2014"),
        annFile=os.path.join(dataset_path, "instances_train2014.json"),
        transforms=transform,
    )
    train_dataset = datasets.wrap_dataset_for_transforms_v2(train_dataset)

    test_dataset = CocoDetection(
        root=os.path.join(dataset_path, "val2014"),
        annFile=os.path.join(dataset_path, "instances_val2014.json"),
        transforms=transform,
    )
    test_dataset = datasets.wrap_dataset_for_transforms_v2(test_dataset)

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
