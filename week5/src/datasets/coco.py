import os
import numpy as np
import json
import random

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CocoDetection


class ImageToTextCOCO(Dataset):
    """
    Custom dataset class for creating triplets from COCO dataset for image retrieval task with Faster R-CNN or Mask R-CNN.
    """

    def __init__(self, coco_dataset, caption_anns: str, subset: str = "train"):
        """
        Args:
            coco_dataset (torchvision.datasets.CocoDetection): COCO dataset object.
            json_file (str): Path to the JSON file containing positive and negative examples for creating triplets.
        """
        self.coco_dataset = coco_dataset

        with open(caption_anns, 'r') as f:
            # Format of the json file: 
            # [{’image_id’: 318556, ’id’: 48, ’caption’: ’A very clean and well…’}, ...]
            self.caption_anns = json.load(f)["annotations"]

        self.subset = subset

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx, return_triplet: bool = False):
        img, target = self.coco_dataset[idx]

        # If target has no annotations, randomly select a new image
        while len(target) == 0:
            idx = random.choice(range(len(self.coco_dataset)))
            img, target = self.coco_dataset[idx]
        
        target = target[0]
        # Get image caption (positive caption)
        positive_caption = [caption['caption'] for caption in self.caption_anns if caption['image_id'] == target['image_id']][0]

        if not return_triplet:
            return img, positive_caption

        # Get negative caption
        negative_caption = random.choice([caption['caption'] for caption in self.caption_anns if caption['image_id'] != target['image_id']])

        return img, positive_caption, negative_caption

    def get_labels(self, idx):
        """
        Returns the category labels corresponding to the given indices.
        Args:
            idx (numpy.ndarray): Array of indices.
        Returns:
            numpy.ndarray: Array of category labels corresponding to the given indices.
        """
        return np.array([self.coco_dataset.coco.loadAnns(self.coco_dataset.coco.getAnnIds(imgIds=self.coco_dataset.ids[i]))[0]['category_id'] for i in idx])
    

def create_dataloader(
        dataset_path: str,
        batch_size: int,
        inference: bool = False,
):
    """
    Creates a dataloader for the COCO dataset.
    Args:
        batch_size (int): Batch size.
        dataset_path (str): Path to the COCO dataset.
        config (Any): Config object.
        inference (bool): Whether to create a dataloader for inference.
        dataset_kwargs (dict): Keyword arguments for the dataset.
    Returns:
        train_dataloader (torch.utils.data.DataLoader): Dataloader for training.
        test_dataloader (torch.utils.data.DataLoader): Dataloader for testing.
    """
    train_coco_dataset = CocoDetection(
        root=os.path.join(dataset_path, "train2014"),
        annFile=os.path.join(dataset_path, "instances_train2014.json"),
    )

    val_coco_dataset = CocoDetection(
        root=os.path.join(dataset_path, "val2014"),
        annFile=os.path.join(dataset_path, "instances_val2014.json"),
    )
    
    # Create dataset
    train_dataset = ImageToTextCOCO(
        coco_dataset=train_coco_dataset,
        caption_anns=os.path.join(dataset_path, "captions_train2014.json"),
        subset="train",
    )
    val_dataset = val_coco_dataset
    

    if not inference:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            # collate_fn=lambda batch: tuple(zip(*batch)),
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            # collate_fn=lambda batch: tuple(zip(*batch)),
        )
    else:
        train_dataset = val_dataset
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            # collate_fn=lambda batch: tuple(zip(*batch)),
        )
        return train_dataloader

    # Temporarily set test_dataloader to val_dataloader
    test_dataloader = val_dataloader

    return train_dataloader, val_dataloader, test_dataloader