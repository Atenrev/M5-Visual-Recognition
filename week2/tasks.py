import os
import cv2
import random

from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from trainers import MyTrainer


def train(cfg, resume_or_load: bool):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=resume_or_load)
    trainer.train()
    

def evaluate(cfg):
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=True)
    metrics = MyTrainer.test(cfg, trainer.model)
    print(metrics)


def draw_seg(cfg, test_dataset: str, randomize: bool = True, num_images: int = 10, mapped: bool = False):
    predictor = DefaultPredictor(cfg)

    out_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TEST[0] + "preds_out")
    os.makedirs(out_dir, exist_ok=True)
    
    cpa_metadata = MetadataCatalog.get(test_dataset)
    dataset_dicts = DatasetCatalog.get(test_dataset)

    if randomize:
        random.shuffle(dataset_dicts)

    for d in tqdm(dataset_dicts[:num_images]):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1],
                    scale=0.8, 
                    instance_mode=ColorMode.SEGMENTATION  
        )
        # vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        instances = outputs["instances"].to("cpu")

        classes_to_keep = ["car"]
        classes_to_keep += ["person"] if mapped else ["pedestrian"]

        # Get the indices of the classes to keep
        class_indices = [cpa_metadata.thing_classes.index(cls) for cls in classes_to_keep]

        # Filter instances based on the predicted class labels
        indices_to_keep = [i for i in range(len(instances)) if instances.pred_classes[i] in class_indices]
        instances = instances[indices_to_keep]

        vis = v.draw_instance_predictions(instances)
        cv2.imwrite(f"{out_dir}/{d['image_id']}.jpg", vis.get_image()[:, :, ::-1])


def draw_dataset(cfg, dataset_name: str, randomize: bool = True, num_images: int = 10):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    # Print the number of classes annotations in the dataset
    n_cars = 0
    n_pedestrians = 0

    classes = MetadataCatalog.get(dataset_name).get("thing_classes", None)
    print("Classes in dataset:", classes)
    
    for d in dataset_dicts:
        for ann in d["annotations"]:
            if ann["category_id"] == 0:
                n_cars += 1
            elif ann["category_id"] == 1:
                n_pedestrians += 1
            else:
                continue

    print(f"Number of cars: {n_cars}")
    print(f"Number of pedestrians: {n_pedestrians}")

    out_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TEST[0] + "_gt_out")
    os.makedirs(out_dir, exist_ok=True)
   
    if randomize:
        random.shuffle(dataset_dicts)

    for d in dataset_dicts[:num_images]:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(f"{out_dir}/gt_{d['image_id']}.jpg", vis.get_image()[:, :, ::-1])