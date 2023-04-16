import os
import numpy as np
import json
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision.datasets import CocoDetection
# import torchvision.transforms.v2 as transforms
import torchvision.transforms as transforms
from torchvision import datasets

from typing import Any

from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from PIL import Image

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
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((config.input_resize, config.input_resize)),
    #         transforms.ToImageTensor(),
    #         transforms.ConvertImageDtype(torch.float32),
    #         transforms.Normalize(
    #             mean=[0.4850, 0.4560, 0.4060],
    #             std=[0.2290, 0.2240, 0.2250]),
    #     ])

    train_dataset = CocoDetection(
        root=os.path.join(dataset_path, "train2014"),
        annFile=os.path.join(dataset_path, "instances_train2014.json"),
        # transforms=transform,
    )
    # train_dataset = datasets.wrap_dataset_for_transforms_v2(train_dataset)

    test_dataset = CocoDetection(
        root=os.path.join(dataset_path, "val2014"),
        annFile=os.path.join(dataset_path, "instances_val2014.json"),
        # transforms=transform,
    )
    # test_dataset = datasets.wrap_dataset_for_transforms_v2(test_dataset)

    if not inference:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            # collate_fn=lambda batch: tuple(zip(*batch)),
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            # collate_fn=lambda batch: tuple(zip(*batch)),
        )
    else:
        train_dataset = test_dataset
        test_dataloader = test_dataset = None
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            # collate_fn=lambda batch: tuple(zip(*batch)),
        )

    return train_dataloader, test_dataloader


class TripletCOCO(Dataset):
    """
    Custom dataset class for creating triplets from COCO dataset for image retrieval task with Faster R-CNN or Mask R-CNN.
    """

    def __init__(self, coco_dataset, json_file, subset):
        """
        Args:
            coco_dataset (torchvision.datasets.CocoDetection): COCO dataset object.
            json_file (str): Path to the JSON file containing positive and negative examples for creating triplets.
        """
        self.coco_dataset = coco_dataset
        self.json_file = json_file
        with open(json_file, 'r') as f:
            self.retrieval_annotations = json.load(f)
        self.subset = subset

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        img, target = self.coco_dataset[idx]
        # If target has no annotations, randomly select a new image
        while 'category_id' not in target.keys():
            idx = random.choice(range(len(self.coco_dataset)))
            img, target = self.coco_dataset[idx]

        # Take the category of the first annotation
        category_id = target['category_id'][0]
        img_id = target['image_id']
        positive_ids = self.retrieval_annotations[self.subset][str(category_id)]

        # Select positive example
        positive_id = img_id
        while positive_id == img_id:
            positive_id = random.choice(positive_ids)
        positive_index = self.coco_dataset.ids.index(positive_id)
        positive_img, _ = self.coco_dataset[positive_index]

        # Select negative example
        # Crate a list of all categories except the current category
        negative_categories = list(self.retrieval_annotations[self.subset].keys())
        negative_categories.remove(str(category_id))
        negative_cat = random.choice(negative_categories)
        # Select a random image from the selected category
        negative_id = random.choice(self.retrieval_annotations[self.subset][negative_cat])
        negative_index = self.coco_dataset.ids.index(negative_id)
        negative_img, _ = self.coco_dataset[negative_index]

        return img, positive_img, negative_img, []

    def get_labels(self, idx):
        """
        Returns the category labels corresponding to the given indices.
        Args:
            idx (numpy.ndarray): Array of indices.
        Returns:
            numpy.ndarray: Array of category labels corresponding to the given indices.
        """
        return np.array([self.coco_dataset.coco.loadAnns(self.coco_dataset.coco.getAnnIds(imgIds=self.coco_dataset.ids[i]))[0]['category_id'] for i in idx])


class TripletHistogramsCOCO(Dataset):
    """
    Custom dataset class for creating triplets from COCO dataset for image retrieval task with Faster R-CNN or Mask R-CNN.
    """

    def __init__(self, coco_dataset, k=1):
        """
        Args:
            coco_dataset (torchvision.datasets.CocoDetection): COCO dataset object.
            k (int): Number of objects to consider for a positive example.
        """
        self.coco_dataset = coco_dataset
        self.similarity_matrix = self.similarity_matrix()
        self.k = k

    def __len__(self):
        return len(self.coco_dataset)

    @staticmethod
    def histograms_intersection(hist1, hist2):
        return np.sum(np.minimum(hist1, hist2))

    def similarity_matrix(self):
        print('Calculating similarity matrix...')
        histograms = self.get_all_histograms()
        return np.array([[self.histograms_intersection(histograms[i], histograms[j])
                          for j in range(len(self.coco_dataset))] for i in range(len(self.coco_dataset))])

    def get_histogram(self, img_id, num_cats=91):
        return np.bincount([self.coco_dataset.coco.loadAnns(ann_id)[0]['category_id'] for ann_id in
                            self.coco_dataset.coco.getAnnIds(imgIds=img_id)], minlength=num_cats)

    def get_all_histograms(self):
        print('Calculating histograms...')
        return [self.get_histogram(img_id) for img_id in self.coco_dataset.ids]

    def __getitem__(self, idx):
        anchor_img, _ = self.coco_dataset[idx]

        chose = random.randint(1, self.k)
        images_ids = sorted(range(len(self.coco_dataset)), key=lambda i: self.similarity_matrix[idx][i])
        positive_img, _ = self.coco_dataset[images_ids[chose]]
        negative_img, _ = self.coco_dataset[images_ids[-1]]

        return anchor_img, positive_img, negative_img, []


class RetrievalCOCO(Dataset):
    """
    Custom dataset class for Image Retrieval task on COCO dataset with Faster R-CNN or Mask R-CNN object detector.
    """
    def __init__(self, coco_dataset, json_file, subset, config):
        """
        Args:
            coco_dataset (torchvision.datasets.CocoDetection): COCO dataset object.
            json_file (str): Path to the JSON file containing positive and negative examples for creating triplets.
            subset (str): Subset of the COCO dataset to use. One of 'database', 'val', 'test'.
        """
        self.coco_dataset = coco_dataset
        self.json_file = json_file
        with open(json_file, 'r') as f:
            self.retrieval_annotations = json.load(f)
        self.subset = subset
        self.images_dict = self.get_images_dict()
        self.config = config

    def get_images_dict(self):
        new_dic = {}
        for k, v in self.retrieval_annotations[self.subset].items():
            for x in v:
                new_dic.setdefault(x, []).append(k)
        return new_dic

    def __len__(self):
        return len(self.images_dict.keys())

    def transforms(self, image):
        transform = transforms.Compose(
            [
                transforms.Resize((self.config.input_resize, self.config.input_resize)),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060],
                    std=[0.2290, 0.2240, 0.2250]),
            ])
        return transform(image)

    def __getitem__(self, idx):
        img_id = list(self.images_dict.keys())[idx]
        filename = self.coco_dataset.coco.loadImgs(img_id)[0]['file_name']

        image_path = os.path.join(self.coco_dataset.root, filename)
        image = Image.open(image_path)

        cats = [int(obj) for obj in self.images_dict[img_id]]

        return image, cats

