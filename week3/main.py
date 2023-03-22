import os
import argparse
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog

from datasets.register_coco import register_coco_dataset
from datasets.register_out_of_context import register_out_of_context_dataset
from tasks import task_a, task_b, task_c, task_d, task_e


def _parse_args() -> argparse.Namespace:
    usage_message = """
                    Script for training a panels detector.
                    """

    parser = argparse.ArgumentParser(usage=usage_message)

    parser.add_argument("--mode", "-m", type=str, default="draw_dataset",
                        help="Mode (task_a, task_b, task_c, task_d, task_e)")
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
    parser.add_argument("--load_dataset", "-tr", type=str, default="coco",
                        help="Load dataset")
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
        raise ValueError("Unknown model.")
    
    cfg.DATALOADER.NUM_WORKERS = 0

    if args.checkpoint is None:
        if args.model == "mask_rcnn":
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
        elif args.model == "faster_rcnn":
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    else:
        cfg.MODEL.WEIGHTS = args.checkpoint

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.NUM_GPUS = args.num_gpus
    
    if args.dry:
        cfg.SOLVER.MAX_ITER = 1
        iterations_for_one_epoch = 1
    else:
        single_iteration = cfg.SOLVER.NUM_GPUS * cfg.SOLVER.IMS_PER_BATCH
        # Get number of samples in train dataset
        td = DatasetCatalog.get(f"{args.load_dataset}_train")
        n_samples = len(td)
        iterations_for_one_epoch = n_samples // single_iteration
        cfg.SOLVER.MAX_ITER = iterations_for_one_epoch * args.epochs
    
    cfg.TEST.EVAL_PERIOD = iterations_for_one_epoch 
    cfg.SOLVER.CHECKPOINT_PERIOD = iterations_for_one_epoch

    if args.head_num_classes is not None:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.head_num_classes

    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = args.output_dir

    return cfg


def main(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = get_base_cfg(args)

    if args.load_dataset == "coco":
        register_coco_dataset(cfg)
    elif args.load_dataset == "out_of_context":
        register_out_of_context_dataset(cfg)
    else:
        raise ValueError("Dataset not implemented.")

    if args.mode == "task_a":
        task_a.run(cfg, args)
    elif args.mode == "task_b":
        task_b.run(cfg, args)
    elif args.mode == "task_c":
        task_c.run(cfg, args)
    elif args.mode == "task_d":
        task_d.run(cfg, args)
    elif args.mode == "task_e":
        task_e.run(cfg, args)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
