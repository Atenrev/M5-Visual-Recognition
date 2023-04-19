import os
import cv2
import json
import torch
import random
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader, Dataset


class ImageToTextCOCO(Dataset):
    """
    Custom dataset class for creating triplets from COCO dataset for image retrieval task with Faster R-CNN or Mask R-CNN.
    """

    def __init__(self, root_path: str, caption_anns: str, transforms=None, subset: str = 'train2014'):
        """
        Args:
            root_path (str): Path to the COCO dataset.
            caption_anns (str): Path to the json file containing the captions.
            transforms (torchvision.transforms): Transforms to apply to the images.
            subset (str): Subset of the dataset to use. Options: train2014, val2014.
        """
        print(f"Loading COCO {subset} dataset...")

        with open(caption_anns, 'r') as f:
            # Format of the json file: 
            # [{’image_id’: 318556, ’id’: 48, ’caption’: ’A very clean and well…’}, ...]
            self.caption_anns = json.load(f)["annotations"]

        image_with_caption = [caption['image_id'] for caption in self.caption_anns]
        self.image_paths: List[str] = []
        self.image_ids: List[int] = []
        
        for image_id in tqdm(os.listdir(os.path.join(root_path, subset))):
            # If image_id is not in caption_anns, then it is not a valid image
            image_id_int = int(image_id.split('.')[0].split('_')[-1])

            if image_id_int not in image_with_caption:
                continue

            self.image_paths.append(os.path.join(root_path, subset, image_id))
            self.image_ids.append(image_id_int)

        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)
    
    def load_image(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img / 255.0
        return img

    def __getitem__(self, idx, return_triplet: bool = True):
        img = self.load_image(idx)
        image_id = self.image_ids[idx]

        if self.transforms is not None:
            img = self.transforms(img)
        
        # Get image caption (positive caption)
        positive_caption = [caption['caption'] for caption in self.caption_anns if caption['image_id'] == image_id][0]

        if not return_triplet:
            return img, positive_caption

        # Get negative caption
        negative_caption = random.choice([caption['caption'] for caption in self.caption_anns if caption['image_id'] != image_id])

        return img, positive_caption, negative_caption
    

def create_dataloader(
        dataset_path: str,
        batch_size: int,
        inference: bool = False,
        input_size: int = 224,
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
    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            # transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]),
            # transforms.ToTensor(),
        ])
    # train_coco_dataset = CocoDetection(
    #     root=os.path.join(dataset_path, "train2014"),
    #     annFile=os.path.join(dataset_path, "instances_train2014.json"),
    #     transforms=lambda x, t: (transform(x), t),
    # )

    # val_coco_dataset = CocoDetection(
    #     root=os.path.join(dataset_path, "val2014"),
    #     annFile=os.path.join(dataset_path, "instances_val2014.json"),
    #     transforms=lambda x, t: (transform(x), t),
    # )
    
    # Create dataset
    train_dataset = ImageToTextCOCO(
        root_path=dataset_path,
        caption_anns=os.path.join(dataset_path, "captions_train2014.json"),
        subset="train2014",
        transforms=transform,
    )
    val_dataset = ImageToTextCOCO(
        root_path=dataset_path,
        caption_anns=os.path.join(dataset_path, "captions_val2014.json"),
        subset="val2014",
        transforms=transform,
    )    

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