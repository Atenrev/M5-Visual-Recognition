import os
import argparse
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

from utils import load_kitti_and_map_to_coco
from tasks import train, evaluate, draw_seg, draw_dataset, draw_sequence


def _parse_args() -> argparse.Namespace:
    usage_message = """
                    Week2 - Pretrained detectors.
                    """

    parser = argparse.ArgumentParser(usage=usage_message)

    parser.add_argument("--mode", "-m", type=str, default="draw_dataset",
                        help="Mode (train, eval, draw_seg, draw_sequence)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Seed")
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
    parser.add_argument("--dataset_dir", "-tr", type=str, default="/home/mcv/datasets/KITTI-MOTS",
                        help="Train dataset name")
    parser.add_argument("--labels_dir", "-ld", type=str, default="./",
                        help="Train dataset name")
    # Trainer settings
    parser.add_argument("--dry", action="store_true", default=False)
    parser.add_argument("--resume_or_load", action="store_true", default=False)
    parser.add_argument("--batch_size", "-bs", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", "-ep", type=int, default=10,
                        help="Training epochs")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0002,
                        help="Training epochs")
    parser.add_argument("--num_gpus", "-ng", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--output_dir", "-od", type=str, default="output/",
                        help="Output directory")
    # Other
    parser.add_argument("--sequence", "-seq", type=str, default="0000",
                        help="Sequence to draw in draw_sequence mode")

    return parser.parse_args()


def get_base_cfg(args):
    cfg = get_cfg()

    if args.model == "mask_rcnn":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    elif args.model == "faster_rcnn":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    else:
        raise ValueError("Unknown model")
    
    cfg.DATASETS.TRAIN = ("kitti_train",)
    if args.mode == "train":
        cfg.DATASETS.TEST = ("kitti_val",)
    elif args.mode == "eval":
        cfg.DATASETS.TEST = ("kitti_test",)
    elif args.mode == "draw_seg" or args.mode == "draw_sequence" or args.mode == "draw_dataset":
        cfg.DATASETS.TEST = ("kitti_test_real",)
    else:
        cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0

    if args.checkpoint is None:
        if args.model == "mask_rcnn":
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
        elif args.model == "faster_rcnn":
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
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
        iterations_for_one_epoch = n_samples // single_iteration
        cfg.SOLVER.MAX_ITER = iterations_for_one_epoch * args.epochs
    
    cfg.TEST.EVAL_PERIOD = iterations_for_one_epoch 
    cfg.SOLVER.CHECKPOINT_PERIOD = iterations_for_one_epoch
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    #     512 if args.model == "mask_rcnn" else 512
    # ) 

    if args.head_num_classes is not None:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.head_num_classes

    # cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = args.output_dir

    return cfg


def main(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    training_path = os.path.join(args.dataset_dir, "training")
    testing_path = os.path.join(args.dataset_dir, "testing")

    if args.map_kitti_to_coco:
        load_kitti_and_map_to_coco("kitti_train", training_path, 
                                   os.path.join(args.labels_dir, "labels_train_split.json"))
        load_kitti_and_map_to_coco("kitti_val", training_path, 
                                   os.path.join(args.labels_dir, "labels_val_split.json"))
        load_kitti_and_map_to_coco("kitti_test", training_path, 
                                   os.path.join(args.labels_dir, "labels_testing.json"))
        load_kitti_and_map_to_coco("kitti_test_real", testing_path,
                                      os.path.join(args.labels_dir, "labels_challenge.json"))
    else:
        # To fintune on kitti we just need to register the dataset as coco
        register_coco_instances("kitti_train",
                                {},
                                os.path.join(args.labels_dir, "labels_train_split.json"),
                                training_path,
                                )
        
        register_coco_instances("kitti_val",
                                {},
                                os.path.join(args.labels_dir, "labels_val_split.json"),
                                training_path,
                                )

        register_coco_instances("kitti_test",
                                {},
                                os.path.join(args.labels_dir, "labels_testing.json"),
                                training_path,
                                )
        register_coco_instances("kitti_test_real",
                                {},
                                os.path.join(args.labels_dir, "labels_challenge.json"),
                                testing_path,
                                )

    cfg = get_base_cfg(args)

    if args.mode == "train":
        train(cfg, args.resume_or_load)
    elif args.mode == "eval":
        evaluate(cfg)
    elif args.mode == "draw_seg":
        draw_seg(cfg, "kitti_test_real", args.model, mapped=args.map_kitti_to_coco)
    elif args.mode == "draw_dataset":
        draw_dataset(cfg, "kitti_test_real")
    elif args.mode == "draw_sequence":
        draw_sequence(cfg, "kitti_test_real", args.model, args.sequence, mapped=args.map_kitti_to_coco)
    else:
        raise Exception("Unknown mode.")


if __name__ == "__main__":
    args = _parse_args()
    main(args)
