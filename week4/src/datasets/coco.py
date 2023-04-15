import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision.datasets import CocoDetection
import torchvision.transforms.v2 as transforms
from torchvision import datasets

from typing import Any

from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

# from collections import defaultdict
# from pathlib import Path
# import json
# from PIL import Image
# import PIL.Image


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
            collate_fn=lambda batch: tuple(zip(*batch)),
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: tuple(zip(*batch)),
        )
    else:
        train_dataset = test_dataset
        test_dataloader = test_dataset = None
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: tuple(zip(*batch)),
        )

    return train_dataloader, test_dataloader


class TripletCOCO(Dataset):
    """
    Custom Dataset class for generating triplets from COCO dataset for image retrieval task with Faster R-CNN or Mask R-CNN.
    """

    def __init__(self, coco_dataset, transform=None):
        """
        Args:
            coco_dataset (COCODataset): COCO dataset object, e.g., torchvision.datasets.CocoDetection
            transform (callable, optional): Optional transform to be applied on the images. Default is None.
        """
        self.coco_dataset = coco_dataset
        self.transform = transform
        self.categories = self.get_categories()

    def get_categories(self):
        categories = []
        for i in range(len(self.coco_dataset)):
            ann_ids = self.coco_dataset.coco.getAnnIds(imgIds=self.coco_dataset.ids[i])

            # if not ann_ids:
            #     print(f"ann_ids is int: {ann_ids}, image id: {self.coco_dataset.ids[i]} number: {i}")
            #     break

            if len(ann_ids) > 0:
                ann_ids = ann_ids[0]
                # Need to access 0 because loadAnns always returns a list!
                cat = self.coco_dataset.coco.loadAnns(ann_ids)[0]['category_id']
            else:
                cat = -1
            categories.append(cat)
        return categories

    def __getitem__(self, index):
        """
        Generates a triplet of images (anchor, positive, negative) for the given index.

        Args:
            index (int): Index of the anchor image.

        Returns:
            tuple: A tuple of anchor, positive, and negative images along with empty lists for target and metadata.
        """
        # Get the anchor image
        anchor_img, _ = self.coco_dataset[index]

        # Get the category of the anchor image
        anchor_ann_id = self.coco_dataset.coco.getAnnIds(imgIds=self.coco_dataset.ids[index])
        anchor_category = self.coco_dataset.coco.loadAnns(anchor_ann_id)[0]['category_id']

        # Find positive sample (image containing the same category as anchor)
        positive_indices = np.where((np.asarray(self.coco_dataset.ids) != index) &
                                    (np.asarray(self.categories) == anchor_category))[0]
        positive_index = np.random.choice(positive_indices)
        positive_img, _ = self.coco_dataset[positive_index]

        # Find negative sample (image containing a different category than anchor)
        negative_indices = np.where(np.asarray(self.categories) != anchor_category)[0]
        negative_index = np.random.choice(negative_indices)
        negative_img, _ = self.coco_dataset[negative_index]

        # Apply transformations if provided
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        # Return the triplet along with empty lists for target
        return (anchor_img, positive_img, negative_img), []

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.coco_dataset)


class TripletCOCOMiner(BaseMiner):
    """
    Custom Triplet Miner class for mining triplets from COCO dataset for image retrieval task with Faster R-CNN or Mask R-CNN.
    """

    def __init__(self, margin=0.1, **kwargs):
        """
        Args:
            margin (float, optional): Margin value for triplet mining. Default is 0.1.
            **kwargs: Additional keyword arguments to be passed to the parent class (BaseMiner).
        """
        super(TripletCOCOMiner, self).__init__(**kwargs)
        self.margin = margin

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        """
        Mines hard triplets from the given embeddings and labels.

        Args:
            embeddings (torch.Tensor): Embeddings of anchor images.
            labels (torch.Tensor): Labels of anchor images.
            ref_emb (torch.Tensor): Embeddings of reference images.
            ref_labels (torch.Tensor): Labels of reference images.

        Returns:
            tuple: A tuple of anchor, positive, and negative indices for hard triplets.
        """
        mat = self.distance(embeddings, ref_emb)
        a, p, n = lmu.get_all_triplets_indices(labels, ref_labels)
        pos_pairs = mat[a, p]
        neg_pairs = mat[a, n]
        triplet_margin = pos_pairs - neg_pairs if self.distance.is_inverted else neg_pairs - pos_pairs
        triplet_mask = triplet_margin <= self.margin
        return a[triplet_mask], p[triplet_mask], n[triplet_mask]

    def get_labels(self, idx):
        """
        Returns the labels corresponding to the given indices.

        Args:
            idx (numpy.ndarray): Array of indices.

        Returns:
            numpy.ndarray: Array of labels corresponding to the given indices.
        """
        return np.array([self.data_source.coco_dataset.coco.loadAnns(self.data_source.coco_dataset.coco.getAnnIds(imgIds=self.data_source.coco_dataset.ids[i]))[0]['category_id'] for i in idx])



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
