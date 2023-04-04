import torch
import kornia.augmentation as K
import numpy as np


def return_image_full_range(image):
    return (torch.clamp(K.Normalize(mean=[-0.4850, -0.4560, -0.4060], std=[1/0.2290, 1/0.2240, 1/0.2250])(image), min = 0, max = 1) * 255).squeeze().cpu().numpy().astype(np.uint8).transpose(1, 2,  0)