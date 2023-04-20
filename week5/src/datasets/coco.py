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


class BaseCOCO(Dataset):
    def __init__(self, root_path: str, caption_anns: str, transforms=None, subset: str = 'train2014', test_mode: bool = False):
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
            caption_anns = json.load(f)["annotations"]

        image_with_caption = [caption['image_id'] for caption in caption_anns]

        # Check if image_paths and image_ids are in cache. If so, load them.
        if os.path.exists(os.path.join("cache", f"{subset}_image_paths.npy")):
            print("Loading image_paths and image_ids from cache...")
            self.image_paths = np.load(os.path.join("cache", f"{subset}_image_paths.npy"), allow_pickle=True).tolist()
            self.image_ids = np.load(os.path.join("cache", f"{subset}_image_ids.npy"), allow_pickle=True).tolist()
            self.captions = np.load(os.path.join("cache", f"{subset}_captions.npy"), allow_pickle=True).tolist()
        else:
            print("Creating image_paths and image_ids...")
            self.image_paths: List[str] = []
            self.image_ids: List[int] = []
            self.captions: List[str] = []

            for image_id in tqdm(os.listdir(os.path.join(root_path, subset))):
                # If image_id is not in caption_anns, then it is not a valid image
                image_id_int = int(image_id.split('.')[0].split('_')[-1])

                # Each image can have up to 5 captions. Get all captions for the image.
                caption_idxs = [i for i, x in enumerate(image_with_caption) if x==image_id_int]
                assert 1 <= len(caption_idxs) <= 5, f"Image {image_id} should have between 1 and 5 captions, but has {len(caption_idxs)} captions."

                for caption_idx in caption_idxs:
                    self.image_paths.append(os.path.join(root_path, subset, image_id))
                    self.image_ids.append(image_id_int)
                    self.captions.append(caption_anns[caption_idx]['caption'])

            # Save image_paths, image_ids and captions to cache
            os.makedirs("cache", exist_ok=True)
            np.save(os.path.join("cache", f"{subset}_image_paths.npy"), self.image_paths)
            np.save(os.path.join("cache", f"{subset}_image_ids.npy"), self.image_ids)
            np.save(os.path.join("cache", f"{subset}_captions.npy"), self.captions)

        print(f"Loaded {len(self.image_paths)} images from COCO {subset} dataset.")

        if test_mode:
            print("Using only 100 images for testing...")
            self.image_paths = self.image_paths[:100]
            self.image_ids = self.image_ids[:100]
            self.captions = self.captions[:100]

        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img / 255.0
        return img


class ImageToTextCOCO(BaseCOCO):
    """
    Custom dataset class for creating triplets from COCO dataset for image retrieval task.
    """

    def __getitem__(self, idx, return_triplet: bool = True):
        img = self.load_image(idx)

        if self.transforms is not None:
            img = self.transforms(img)

        # Get image caption (positive caption)
        positive_caption = self.captions[idx]

        if not return_triplet:
            return img, positive_caption

        # Get negative caption
        negative_caption_idx = random.randint(0, len(self.captions) - 1)
        while negative_caption_idx == idx:
            negative_caption_idx = random.randint(0, len(self.captions) - 1)
        negative_caption = self.captions[negative_caption_idx]

        return img, positive_caption, negative_caption


class TextToImageCOCO(BaseCOCO):
    """
    Custom dataset class for creating triplets from COCO dataset for image retrieval task.
    """

    def __getitem__(self, idx, return_triplet: bool = True):
        anchor_caption = self.captions[idx]

        # Get positive image
        positive_img = self.load_image(idx)

        if self.transforms is not None:
            positive_img = self.transforms(positive_img)

        if not return_triplet:
            return anchor_caption, positive_img

        # Get negative image
        negative_img_idx = random.randint(0, len(self.image_paths) - 1)
        while negative_img_idx == idx:
            negative_img_idx = random.randint(0, len(self.image_paths) - 1)
        negative_img = self.load_image(negative_img_idx)

        if self.transforms is not None:
            negative_img = self.transforms(negative_img)

        return anchor_caption, positive_img, negative_img
    

def create_dataloader(
        dataset_path: str,
        batch_size: int,
        inference: bool = False,
        input_size: int = 224,
        mode: str = "image_to_text",
        test_mode: bool = False,
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
            transforms.Resize((input_size, input_size)),
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
    if mode == "image_to_text" or mode == "symmetric":
        train_dataset = ImageToTextCOCO(
            root_path=dataset_path,
            caption_anns=os.path.join(dataset_path, "captions_train2014.json"),
            subset="train2014",
            transforms=transform,
            test_mode=test_mode,
        )
        val_dataset = ImageToTextCOCO(
            root_path=dataset_path,
            caption_anns=os.path.join(dataset_path, "captions_val2014.json"),
            subset="val2014",
            transforms=transform,
            test_mode=test_mode,
        )    
    elif mode == "text_to_image":
        train_dataset = TextToImageCOCO(
            root_path=dataset_path,
            caption_anns=os.path.join(dataset_path, "captions_train2014.json"),
            subset="train2014",
            transforms=transform,
            test_mode=test_mode,
        )
        val_dataset = TextToImageCOCO(
            root_path=dataset_path,
            caption_anns=os.path.join(dataset_path, "captions_val2014.json"),
            subset="val2014",
            transforms=transform,
            test_mode=test_mode,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if not inference:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
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