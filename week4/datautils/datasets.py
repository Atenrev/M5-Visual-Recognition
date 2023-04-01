import os
import cv2

import zipfile

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
    
class MITDataset:
    def __init__(self) -> None:
        pass