from pathlib import Path
from detectron2.config import get_cfg
from detectron2_1.adv import DAGAttacker
from detectron2.structures import pairwise_iou, Boxes
from detectron2 import model_zoo
import torch
import numpy as np
from PIL import Image

def run_adv(cfg, args):
    pass

def run_naive(cfg, args):
    pass

def predict(image, model):
    pass

def run(cfg, args):
    pass