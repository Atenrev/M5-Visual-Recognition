from pathlib import Path
from detectron2.config import get_cfg
from detectron2_1.adv import DAGAttacker
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

image_path =  '/home/adri/Desktop/master/M5/M5-Visual-Recognition/week3/data/mscoco/task_c/000000004036.jpg' # Yes We are doing It That Way

def run_adv():

    # credits: https://github.com/yizhe-ang/detectron2-1/blob/master/notebooks/adv.ipynb

    cfg = get_cfg()
    MODEL = MODELS['mask']
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)
    #attacker = DAGAttacker(cfg)
    print('hello')


    #for batch in attacker.data_loader:
     #   x = batch[0]
     #   print(x)
     #   break




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


if __name__ == '__main__': 
    run_adv()