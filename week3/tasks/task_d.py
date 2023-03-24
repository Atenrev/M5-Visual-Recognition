from pathlib import Path
from detectron2.config import get_cfg
# from detectron2.adv import DAGAttacker
from detectron2.structures import pairwise_iou, Boxes
from detectron2 import model_zoo
import torch
import numpy as np
from PIL import Image
import cv2

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


MODELS = {
    "mask": 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    "faster": 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
}

image_path = None # Yes We are doing It That Way

def run_adv(cfg, args,):

    # credits: https://github.com/yizhe-ang/detectron2-1/blob/master/notebooks/adv.ipynb

    # attacker = DAGAttacker(cfg) ########## COMMENTED UNTIL IT WORKS

    # coco_instances_results, perturbed = attacker.run_DAG(vis=False)
    pass


def run_naive(cfg, args):
    pass

def predict(image, model, cfg):
    # Inference
    outputs = model(image)
    v = Visualizer(
        model[:, :, ::-1],
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imwrite(
        'tmp.png',
        out.get_image()[:, :, ::-1],
    )
    return out

def run(cfg, args):
    run_adv(cfg, args)