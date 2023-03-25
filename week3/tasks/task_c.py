import json
from pathlib import Path
import numpy as np
import pandas as pd
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

def task_b_2(args):
    dataset_path = Path(args.dataset_path)
    out_path = Path(args.out_path) 
    out_path.mkdir(parents=True, exist_ok=True)

    MODEL = MODELS[args.model]
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)

    for path_to_img in dataset_path.glob("*.jpg"):
        print(f"{path_to_img}")
        im = cv2.imread(str(path_to_img))

        # run inference
        outs = predictor(im)
        viz = Visualizer(
            im[:, :, ::-1],
            MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
        )
        out = viz.draw_instance_predictions(outs["instances"].to("cpu"))

        cv2.imwrite(
            str(out_path / path_to_img.parts[-1]),
            out.get_image()[:, :, ::-1],
            [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        )


def run(cfg, args):

    task_b_2(cfg, args)

    args.dataset_path = "./data/mscoco/task_c"
    args.out_path = "./output/task_c"
    args.model = "mask"
    task_b_2(args)