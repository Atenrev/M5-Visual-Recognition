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


def task_b_1(cfg, args):
    data_dir = Path("./data/mscoco/annotations")
    out_dir = Path("./output/task_b")

    # load data
    with open(data_dir / "captions_train2017.json", 'r') as f_json:
        captions = json.load(f_json)

    with open(data_dir / "instances_train2017.json", 'r') as f_json:
        instances = json.load(f_json)

    # get categories and class names
    categories = instances["categories"]
    categories.sort(key=lambda x: x["id"])
    n_class = categories[-1]["id"]

    classes = ["" for _ in range(n_class + 1)]
    for x in categories:
        classes[x["id"] - 1] = x["name"]

    # get image to class mapping
    img_class = {}

    for x in instances["annotations"]:
        if x["image_id"] not in img_class.keys():
            img_class[x["image_id"]] = []
        img_class[x["image_id"]].append(x["category_id"])

    # get class to class co-occurrence matrix
    cnts = np.zeros((len(classes), len(classes)), dtype=int)

    for k, v in img_class.items():
        v = list(set(v))
        v.sort()
        for i in range(len(v)):
            for j in range(i, len(v)):
                cnts[v[i] - 1, v[j] - 1] += 1

    n_imgs = len(instances["images"])

    # get class probabilities
    diag_cnts = np.diag(np.ones(len(classes), dtype=bool))
    cnts_class = cnts[diag_cnts]
    cnts[diag_cnts] = 0

    cnts += cnts.T
    cnts[diag_cnts] += cnts_class

    probas_class = cnts_class / n_imgs
    joint_probas = cnts.astype(float) / n_imgs
    cond_probas = joint_probas / probas_class[:, None]

    probas_class = pd.DataFrame(probas_class, index=classes)
    cnts = pd.DataFrame(cnts, index=classes, columns=classes)
    joint_probas = pd.DataFrame(joint_probas, index=classes, columns=classes)
    cond_probas = pd.DataFrame(cond_probas, index=classes, columns=classes)

    # save results to out_dir
    cnts.to_csv(out_dir / "cnts.csv")
    joint_probas.to_csv(out_dir / "joint_probas.csv")
    cond_probas.to_csv(out_dir / "cond_probas.csv")
    probas_class.to_csv(out_dir / "probas_class.csv")

    cnts.to_markdown(out_dir / "cnts.md")
    joint_probas.to_markdown(out_dir / "joint_probas.md")
    cond_probas.to_markdown(out_dir / "cond_probas.md")
    probas_class.to_markdown(out_dir / "probas_class.md")


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

    task_b_1(cfg, args)

    args.dataset_path = "./data/mscoco/task_b"
    args.out_path = "./output/task_b"
    args.model = "mask"
    task_b_2(args)
