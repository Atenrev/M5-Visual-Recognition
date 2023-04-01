import os
import cv2
import glob
from PIL import Image

import zipfile
import torch

from typing import *
from torch.utils.data import Dataset



class BaseDataset(Dataset[Any]):
    """
    Base Dataset.

    Adapted from week 1 bc omfg sergi de veres hem de tenir una conversa sobre praxis

    """

    def __init__(self,
                 device: torch.device,
                 config: Any,
                 ) -> None:
        """
        Constructor of the Dataset.
        """
        self.device = device
        self.config = config

    def getitem(self, idx: int):
        """
        Returns the item at the given index as a Sample object.
        """
        raise NotImplementedError

    def __getitem__(self, idx: int):
        """
        Returns the item at the given index.

        .. warning::
            This method must not be overridden.
        """
        return self.getitem(idx)



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

    def getitem(self, idx: int):
        image_path = self.image_paths[idx]
        sample_id = os.path.basename(image_path)

        image = Image.open(image_path).convert("RGB")
        image = torch.tensor(image).float() / 255

        if self.transform:
            image = self.transform(image).squeeze()

        return image, self.labels[idx]

class ZippedDataloader:

    # self-stolen from https://github.com/EauDeData/cvc-dataset-projector/blob/main/src/datautils/dataloaders.py
    # just for quicly testing the pipeline
    # extracts a zip and yields a class with defined iterator

    def __init__(self, path_to_zip, temporal_folder = 'local/') -> None:
        os.makedirs(temporal_folder, exist_ok=True)

        with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
            zip_ref.extractall(temporal_folder)
        
        self.files = [os.path.join(temporal_folder, x) for x in os.listdir(temporal_folder)]
        self.inner_state = 0

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        return cv2.imread(self.files[index], cv2.IMREAD_COLOR).transpose(2, 0, 1), index
    
    def __next__(self):
        
        if self.inner_state > (len(self) - 1):
            self.inner_state += 1
            return self[self.inner_state - 1]
        
        raise StopIteration
