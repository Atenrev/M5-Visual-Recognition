import torch
import yaml
import kornia.augmentation as K
import numpy as np

from typing import Any
from munch import DefaultMunch


def get_configuration(yaml_path: str) -> Any:
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
        return DefaultMunch.fromDict(yaml_dict)


def return_image_full_range(image):
    return (torch.clamp(K.Normalize(mean=[-0.4850, -0.4560, -0.4060], std=[1/0.2290, 1/0.2240, 1/0.2250])(image), min = 0, max = 1) * 255).squeeze().cpu().numpy().astype(np.uint8).transpose(1, 2,  0)