import os
import argparse
import cv2
import random
import json

from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, Metadata
from detectron2.structures import BoxMode

from trainers import MyTrainer


def _parse_args() -> argparse.Namespace:
    usage_message = """
                    Script for training a panels detector.
                    """

    parser = argparse.ArgumentParser(usage=usage_message)

    parser.add_argument("--mode", "-m", type=str, default="draw_dataset",
                        help="Mode (train, eval, draw_seg)")
    # Model settings
    parser.add_argument("--model", "-mo", type=str, default="mask_rcnn",
                        help="Model (mask_rcnn, faster_rcnn)")
    parser.add_argument("--checkpoint", "-ch", type=str, default=None, #"checkpoints/model_3.pth",
                        help="Model weights path")
    parser.add_argument("--head_num_classes", "-hnc", type=int, default=None,
                        help="Number of classes for the head. If not set, uses the default model head.")
    # Dataset settings
    parser.add_argument("--map_kitti_to_coco", action="store_true", default=False,
                        help="Map KITTI classes to COCO classes")
    parser.add_argument("--dataset_dir", "-tr", type=str, default="/home/mcv/datasets/KITTI-MOTS/training",
                        help="Train dataset name")
    parser.add_argument("--labels_dir", "-ld", type=str, default="./",
                        help="Train dataset name")
    # Trainer settings
    parser.add_argument("--dry", action="store_true", default=False)
    parser.add_argument("--resume_or_load", action="store_true", default=False)
    parser.add_argument("--batch_size", "-bs", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", "-ep", type=int, default=20,
                        help="Training epochs")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0002,
                        help="Training epochs")
    parser.add_argument("--num_gpus", "-ng", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--output_dir", "-od", type=str, default="output/",
                        help="Output directory")

    return parser.parse_args()


def load_kitti_and_map_to_coco(dataset_name: str, dataset_dir: str, labels_path: str):
    # Load the annotations from the JSON file
    with open(labels_path, "r") as f:
        annotations = json.load(f)
    
    class_dict = {1: 2, 2: 0}
    detectron_anns = []

    # Update the image paths and add the annotations
    for image in annotations["images"]:
        image["image_id"] = image["id"]
        image["file_name"] = os.path.join(dataset_dir, image["file_name"])
        image["annotations"] = []

        for ann in annotations["annotations"]:
            if ann["image_id"] == image["id"]:
                ann["category_id"] = class_dict[ann["category_id"]]
                ann["bbox_mode"] = BoxMode.XYWH_ABS,
                image["annotations"].append(ann)

        detectron_anns.append(image)
        

    # Create the DatasetCatalog entry
    DatasetCatalog.register(
        dataset_name,
        lambda: detectron_anns,
    )

    coco_names = [""] * 81
    coco_names[0] = "person"
    coco_names[2] = "car"

    # Define the metadata
    metadata = {
        "thing_classes": coco_names,
        # "thing_dataset_id_to_contiguous_id": {1: 2, 2: 0},
    }

    # Create the MetadataCatalog entry
    MetadataCatalog.get(dataset_name).set(**metadata)


def get_base_cfg(args):
    cfg = get_cfg()

    if args.model == "mask_rcnn":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    elif args.model == "faster_rcnn":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    else:
        raise ValueError("Unknown model")
    
    cfg.DATASETS.TRAIN = ("kitti_train",)
    cfg.DATALOADER.NUM_WORKERS = 1

    if args.checkpoint is None:
        if args.model == "mask_rcnn":
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        elif args.model == "faster_rcnn":
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    else:
        cfg.MODEL.WEIGHTS = args.checkpoint

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    # cfg.INPUT.MIN_SIZE_TRAIN = (800, )
    # cfg.INPUT.MIN_SIZE_TEST = (1200, )
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.NUM_GPUS = args.num_gpus
    
    if args.dry:
        cfg.SOLVER.MAX_ITER = 1
        iterations_for_one_epoch = 1
    else:
        single_iteration = cfg.SOLVER.NUM_GPUS * cfg.SOLVER.IMS_PER_BATCH
        # Get number of samples in train dataset
        td = DatasetCatalog.get("kitti_train")
        n_samples = len(td)
        print(n_samples)
        iterations_for_one_epoch = n_samples // single_iteration
        cfg.SOLVER.MAX_ITER = iterations_for_one_epoch * args.epochs
    
    cfg.TEST.EVAL_PERIOD = iterations_for_one_epoch 
    cfg.SOLVER.CHECKPOINT_PERIOD = iterations_for_one_epoch
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    #     512 if args.model == "mask_rcnn" else 512
    # ) 

    if args.head_num_classes is not None:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.head_num_classes

    return cfg


def train(cfg, resume_or_load: bool):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.DATASETS.TEST = ("kitti_val", )
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=resume_or_load)
    trainer.train()
    

def evaluate(cfg):
    cfg.DATASETS.TEST = ("kitti_test", )
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=True)
    metrics = MyTrainer.test(cfg, trainer.model)
    print(metrics)


def draw_seg(cfg, test_dataset: str, randomize: bool = True, num_images: int = 10):
    from detectron2.utils.visualizer import ColorMode
    predictor = DefaultPredictor(cfg)

    out_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TEST[0] + "_out")
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
        classes_to_keep += ["person"] if args.map_kitti_to_coco else ["pedestrian"]

        # Get the indices of the classes to keep
        class_indices = [cpa_metadata.thing_classes.index(cls) for cls in classes_to_keep]

        # Filter instances based on the predicted class labels
        indices_to_keep = [i for i in range(len(instances)) if instances.pred_classes[i] in class_indices]
        instances = instances[indices_to_keep]

        vis = v.draw_instance_predictions(instances)
        print(d)
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

    out_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TEST[0] + "_out")
    os.makedirs(out_dir, exist_ok=True)
   
    if randomize:
        random.shuffle(dataset_dicts)

    for d in dataset_dicts[:num_images]:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(f"{out_dir}/{d['image_id']}.jpg", vis.get_image()[:, :, ::-1])


def main(args: argparse.Namespace):
    if args.map_kitti_to_coco:
        load_kitti_and_map_to_coco("kitti_train", args.dataset_dir, 
                                   os.path.join(args.labels_dir, "labels_train_split.json"))
        load_kitti_and_map_to_coco("kitti_val", args.dataset_dir, 
                                   os.path.join(args.labels_dir, "labels_val_split.json"))
        load_kitti_and_map_to_coco("kitti_test", args.dataset_dir, 
                                   os.path.join(args.labels_dir, "labels_testing.json"))
    else:
        register_coco_instances("kitti_train",
                                {},
                                os.path.join(args.labels_dir, "labels_train_split.json"),
                                args.dataset_dir,
                                )
        
        register_coco_instances("kitti_val",
                                {},
                                os.path.join(args.labels_dir, "labels_val_split.json"),
                                args.dataset_dir,
                                )

        register_coco_instances("kitti_test",
                                {},
                                os.path.join(args.labels_dir, "labels_testing.json"),
                                args.dataset_dir,
                                )

    cfg = get_base_cfg(args)

    if args.mode == "train":
        train(cfg, args.resume_or_load)
    elif args.mode == "eval":
        evaluate(cfg)
    elif args.mode == "draw_seg":
        draw_seg(cfg, "kitti_test")
    elif args.mode == "draw_dataset":
        draw_dataset(cfg, "kitti_train")
    else:
        raise Exception("Unknown mode.")


if __name__ == "__main__":
    args = _parse_args()
    main(args)