import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

MODELS = {
    "mask": 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    "faster": 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
}

MODEL = MODELS['mask']
device = 'cuda'

def task_d(*args, attacked_image = './data/weird/el_bone.jpg'):

    npimage = cv2.imread(attacked_image)
    cv2.resize(npimage, (224, int(224 * npimage.shape[0]/npimage.shape[1]) ))
    image = torch.from_numpy(cv2.imread(attacked_image).transpose(2, 0, 1)).float().to(device)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)
    model = predictor.model.backbone

    print(image.shape)
    print(model(image.unsqueeze(0)).shape)
# i give up i went crazy

if __name__ == '__main__': 
    task_d()