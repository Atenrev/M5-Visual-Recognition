import os
import argparse
import numpy as np
import cv2
import torch
from pytorch_metric_learning import testers, samplers, losses, distances, trainers, miners
import json
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

from src.models.resnet import ResNetWithEmbedder

from src.utils import get_configuration
from src.datasets.coco import create_coco_dataloader



def _parse_args() -> argparse.Namespace:
    usage_message = """
                    Script for training a panels detector.
                    """

    parser = argparse.ArgumentParser(usage=usage_message)

    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Seed")
    parser.add_argument("--output_dir", "-o", type=str, default="output",
                        help="Output directory")
    # Dataset settings
    parser.add_argument('--dataset_path', type=str, default='../datasets/COCO',
                        help='Path to the dataset.')
    parser.add_argument('--dataset_config_path', type=str, default='./configs/coco.yaml',
                        help='Path to the dataset config file.')
    # parser.add_argument("--train_instances", "-ta", type=str, default="../datasets/COCO/instances_train2014.json",
    #                     help="Path to COCO train instances file in JSON format")
    # parser.add_argument("--val_instances", "-va", type=str, default="../datasets/COCO/instances_val2014.json",
    #                     help="Path to COCO val instances file in JSON format")
    # parser.add_argument("--train_images", "-ti", type=str, default="../datasets/COCO/train2014",
    #                     help="Path to COCO train images")
    # parser.add_argument("--val_images", "-vi", type=str, default="../datasets/COCO/val2014",
    #                     help="Path to COCO val images")

    # Metric Learning Model settings
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of the embedding vector.')

    # Loss configuration
    parser.add_argument('--triplet_margin', type=float, default=0.05,
                        help='Margin for triplet loss.')

    # Training configuration
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use. Options: adam, sgd.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--lr_trunk', type=float, default=1e-5,
                        help='Learning rate for the trunk.')
    parser.add_argument('--lr_embedder', type=float, default=1e-4,
                        help='Learning rate for the embedder.')

    # Miner configuration
    parser.add_argument('--miner_pos_margin', type=float, default=0.2,
                        help='Positive margin for the miner.')
    parser.add_argument('--miner_type_of_triplets', type=str, default='all',
                        help='Type of triplets for the miner. Options: all, easy, semihard, hard.')

    # Object Detection Model settings
    parser.add_argument("--model", "-mo", type=str, default="faster_rcnn",
                        help="Model (mask_rcnn, faster_rcnn)")
    parser.add_argument("--checkpoint", "-ch", type=str, default=None,
                        help="Model weights path")

    return parser.parse_args()


def run_model_on_images(cfg, input_dir):
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)

    img_paths = [os.path.join(input_dir, f) for f in os.listdir(
        input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for img_path in tqdm(img_paths):
        img = cv2.imread(img_paths[5])
        # img = cv2.imread(img_path)
        outputs = predictor(img)

        pred_classes = outputs['instances'].pred_classes.cpu().tolist()
        # MetadataCatalog.get("my_dataset").thing_classes =

        class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).things_classes

        v = Visualizer(
            img[:, :, ::-1],
            MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            scale=1.2,
            instance_mode=ColorMode.SEGMENTATION,
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        drawings = v.get_image()[:, :, ::-1]

    # Read instances files in JSON format
    instances_train = json.load(open("/Users/Alex/MacBook Pro/MSc in CV/M5 - Visual Recognition/M5-Visual-Recognition/datasets/COCO/instances_train2014.json"))





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

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.OUTPUT_DIR = args.output_dir

    return cfg


def main(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNetWithEmbedder(resnet='18', embed_size=args.embedding_size)

    model.to(device)

    # Dataset loading
    dataset_config = get_configuration(args.dataset_config_path)

    train_dataloader, val_dataloader = create_coco_dataloader(
        args.batch_size, args.dataset_path, dataset_config
    )
    train_ds = train_dataloader.dataset
    # val_ds = val_dataloader.dataset

    # Loss configuration
    distance = distances.CosineSimilarity()
    criterion = losses.TripletMarginLoss(
        margin=args.triplet_margin,
        distance=distance,
    )

    miner = miners.TripletMarginMiner(
        margin=args.miner_pos_margin,
        type_of_triplets=args.miner_type_of_triplets,
    )

    # Optimizer configuration
    trunk_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_trunk)
    embedder_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_embedder)

    # Training
    trainer = trainers.MetricLossOnly(
        models={"trunk": model.trunk, "embedder": model.embedder},
        optimizers={"trunk_optimizer": trunk_optimizer,
                    "embedder_optimizer": embedder_optimizer},
        loss_funcs={"metric_loss": criterion},
        mining_funcs={"tuple_miner": miner} if miner else None,
        data_device=device,
        dataset=train_ds,
        batch_size=args.batch_size,
        # sampler=class_sampler,
        # end_of_iteration_hook=hooks.end_of_iteration_hook,
        # end_of_epoch_hook=end_of_epoch_hook,
    )
    trainer.train(num_epochs=args.epochs)


    # cfg = get_base_cfg(args)

    # register_coco_dataset(cfg)

    # train_data = load_coco_json(
    #     json_file=args.train_instances,
    #     image_root=args.train_images,
    #     dataset_name=args.dataset,
    # )



    # register_coco_instances(
    #     name=args.dataset + "_train",
    #     metadata={},
    #     json_file=args.train_instances,
    #     image_root=args.train_images,
    # )
    # register_coco_instances(
    #     name=args.dataset + "_val",
    #     metadata={},
    #     json_file=args.val_instances,
    #     image_root=args.val_images,
    # )
    # from detectron2.data import DatasetCatalog
    # metadata = MetadataCatalog.get(args.dataset + "_train")
    # dataset_dicts = DatasetCatalog.get(args.dataset + "_train")
    #
    # img = cv2.imread(dataset_dicts[5]["file_name"])
    #
    # v = Visualizer(
    #     img[:, :, ::-1],
    #     metadata,
    #     scale=1.2,
    #     instance_mode=ColorMode.SEGMENTATION,
    # )
    # v = v.draw_dataset_dict(dataset_dicts[5])
    # drawings = v.get_image()[:, :, ::-1]
    #
    #
    # run_model_on_images(
    #     cfg,
    #     "/Users/Alex/MacBook Pro/MSc in CV/M5 - Visual Recognition/M5-Visual-Recognition/datasets/COCO/train2014",
    # )


if __name__ == "__main__":
    args = _parse_args()
    main(args)
