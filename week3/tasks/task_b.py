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

    with open(data_dir / "captions_train2017.json", 'r') as f_json:
        captions = json.load(f_json)

    with open(data_dir / "instances_train2017.json", 'r') as f_json:
        instances = json.load(f_json)

    print(list(captions.keys()))
    print(list(instances.keys()))

    print(captions["annotations"][:3])
    print(instances["annotations"][:3])

    # get categories and class names
    cat_objs = instances["categories"]
    cat_objs.sort(key=lambda x: x["id"])
    num_classes = cat_objs[-1]["id"]

    class_names = ["" for _ in range(num_classes + 1)]
    for x in cat_objs:
        class_names[x["id"] - 1] = x["name"]

    # get image to class mapping
    img_to_class = {}

    for x in instances["annotations"]:
        if x["image_id"] not in img_to_class.keys():
            img_to_class[x["image_id"]] = []
        img_to_class[x["image_id"]].append(x["category_id"])

    # get class to class co-occurrence matrix
    counts = np.zeros((len(class_names), len(class_names)), dtype=int)

    for k, v in img_to_class.items():
        v = list(set(v))
        v.sort()
        for ii in range(len(v)):
            for jj in range(ii, len(v)):
                counts[v[ii] - 1, v[jj] - 1] += 1

    n_images = len(instances["images"])

    # get class probabilities
    diagonal_ind = np.diag(np.ones(len(class_names), dtype=bool))
    class_counts = counts[diagonal_ind]
    counts[diagonal_ind] = 0

    counts += counts.T
    counts[diagonal_ind] += class_counts

    class_probs = class_counts / n_images
    jprobs = counts.astype(float) / n_images
    cprobs = jprobs / class_probs[:, None]

    class_probs = pd.DataFrame(class_probs, index=class_names)
    counts = pd.DataFrame(counts, index=class_names, columns=class_names)
    jprobs = pd.DataFrame(jprobs, index=class_names, columns=class_names)
    cprobs = pd.DataFrame(cprobs, index=class_names, columns=class_names)

    print(counts.head())
    print(jprobs.head())
    print(cprobs.head())
    print(class_probs.head())

    # save out_dir
    counts.to_csv(out_dir / "counts.csv")
    jprobs.to_csv(out_dir / "jprobs.csv")
    cprobs.to_csv(out_dir / "cprobs.csv")
    class_probs.to_csv(out_dir / "class_probs.csv")

    counts.to_markdown(out_dir / "counts.md")
    jprobs.to_markdown(out_dir / "jprobs.md")
    cprobs.to_markdown(out_dir / "cprobs.md")
    class_probs.to_markdown(out_dir / "class_probs.md")


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

    for img_path in dataset_path.glob("*.png"):
        print(f"{img_path}")
        im = cv2.imread(str(img_path))

        # Inference
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imwrite(
            str(out_path / img_path.parts[-1]),
            out.get_image()[:, :, ::-1],
            [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        )


def run(cfg, args):

    # task_b_1(cfg, args)
    args.dataset_path = "./data/mscoco/task_b"
    args.out_path = "./output/task_b"
    args.model = "mask"
    task_b_2(args)
