import os
import cv2

from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


def run_model_on_images(cfg, input_dir, output_dir):
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)

    img_paths = [os.path.join(input_dir, f) for f in os.listdir(
        input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    model_name = cfg.MODEL.WEIGHTS.split("/")[-1].split(".")[0]
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        outputs = predictor(img)

        v = Visualizer(
            img[:, :, ::-1],
            MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=1.2,
            instance_mode=ColorMode.SEGMENTATION,
        )

        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, v.get_image()[:, :, ::-1])


def run(cfg, args):
    run_model_on_images(
        cfg,
        "/home/mcv/datasets/out_of_context/",
        "output/task_a"
    )
