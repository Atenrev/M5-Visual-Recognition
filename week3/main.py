import argparse
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg

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
    parser.add_argument("--checkpoint", "-ch", type=str, default=None,
                        help="Model weights path")
    # Dataset settings
    parser.add_argument("--load_dataset", "-tr", type=str, default="coco",
                        help="Load dataset")
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
