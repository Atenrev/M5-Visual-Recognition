import os
import argparse
import cv2

from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer


def _parse_args() -> argparse.Namespace:
    usage_message = """
                    Script for training a panels detector.
                    """

    parser = argparse.ArgumentParser(usage=usage_message)

    parser.add_argument("--mode", "-m", type=str, default="eval",
                        help="Mode (train, eval, draw_seg)")
    parser.add_argument("--dry", action="store_true", default=False)
    parser.add_argument("--resume_or_load", action="store_true", default=False)
    parser.add_argument("--batch_size", "-bs", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--num_gpus", "-ng", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--epochs", "-ep", type=int, default=20,
                        help="Training epochs")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0002,
                        help="Training epochs")
    parser.add_argument("--checkpoint", "-ch", type=str, default="checkpoints/model_3.pth",
                        help="Model weights path")
    parser.add_argument("--train_dataset", "-tr", type=str, default="acmd_v3",
                        help="Train dataset name")
    parser.add_argument("--test_dataset", "-te", type=str, default="eBDtheque_database_v3",
                        help="Test dataset name")
    parser.add_argument("--datasets_dir", "-dsdir", type=str, default="./datasets/",
                        help="Directory with all the datasets")
    parser.add_argument("--output_dir", "-od", type=str, default="output/",
                        help="Output directory")

    return parser.parse_args()


def get_base_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (args.train_dataset,)
    cfg.DATASETS.TEST = (args.test_dataset,)
    cfg.DATALOADER.NUM_WORKERS = 1

    if args.checkpoint is None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    else:
        cfg.MODEL.WEIGHTS = args.checkpoint

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    # cfg.INPUT.MIN_SIZE_TRAIN = (800, )
    # cfg.INPUT.MAX_SIZE_TRAIN = 1200
    cfg.INPUT.MIN_SIZE_TEST = (1200, )
    # cfg.INPUT.MAX_SIZE_TEST = 1200
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.NUM_GPUS = args.num_gpus
    
    if args.dry:
        cfg.SOLVER.MAX_ITER = 1
        iterations_for_one_epoch = 1
    else:
        single_iteration = cfg.SOLVER.NUM_GPUS * cfg.SOLVER.IMS_PER_BATCH
        iterations_for_one_epoch = 8043 // single_iteration # 8035
        cfg.SOLVER.MAX_ITER = iterations_for_one_epoch * args.epochs
    
    cfg.TEST.EVAL_PERIOD = iterations_for_one_epoch 
    cfg.SOLVER.CHECKPOINT_PERIOD = iterations_for_one_epoch
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    ) 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    return cfg


def train(cfg, resume_or_load: bool):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume_or_load)
    trainer.train()
    

def evaluate(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    metrics = DefaultTrainer.test(cfg, trainer.model)    


def draw_seg(cfg, test_dataset: str):
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    predictor = DefaultPredictor(cfg)

    out_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TEST[0] + "_out")
    os.makedirs(out_dir, exist_ok=True)
    
    cpa_metadata = MetadataCatalog.get(test_dataset)
    dataset_dicts = DatasetCatalog.get(test_dataset)

    for d in tqdm(dataset_dicts):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1],
                    scale=0.8, 
                    # instance_mode=ColorMode.IMAGE_BW  
        )
        vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(f"{out_dir}/{d['image_id']}.jpg", vis.get_image()[:, :, ::-1])


def main(args: argparse.Namespace):
    if args.train_dataset:
        register_coco_instances(args.train_dataset,
                                {},
                                f"{args.datasets_dir}/{args.train_dataset}/labels.json",
                                f"{args.datasets_dir}/{args.train_dataset}/data",
                                ) 

    register_coco_instances(args.test_dataset,
                            {},
                            f"{args.datasets_dir}/{args.test_dataset}/labels.json",
                            f"{args.datasets_dir}/{args.test_dataset}/data",
                            )
    
    cfg = get_base_cfg(args)

    if args.mode == "train":
        train(cfg, args.resume_or_load)
    elif args.mode == "eval":
        assert args.checkpoint is not None
        evaluate(cfg)
    elif args.mode == "draw_seg":
        assert args.checkpoint is not None
        draw_seg(cfg, args.test_dataset)
    else:
        raise Exception("Unknown mode.")


if __name__ == "__main__":
    args = _parse_args()
    main(args)