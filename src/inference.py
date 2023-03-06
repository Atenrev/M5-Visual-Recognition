import os
import torch
import logging

from tqdm import tqdm
from typing import Any
from torch.utils.data import DataLoader


class InferenceEngine:
    """
    Inference Engine
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        """
        Constructor of the InferenceEngine.

        Args:
            model: The model to use.
            device: The device to use.
        """
        self.model = model
        self.device = device
        self.output_dir = "inference_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, dataloader: DataLoader[Any]) -> None:
        """
        Run the inference engine through the dataloader and
        save the results to the output directory.

        Args:
            dataloader: The dataloader to use.
        """
        logging.info(f"Running inference on dataset.")
        self.model.eval()

        pass
