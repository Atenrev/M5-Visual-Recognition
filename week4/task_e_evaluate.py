import os
import argparse

import torch
from tqdm import tqdm
import logging
import numpy as np

from src.metrics import *
from src.utils import return_image_full_range
from src.methods.annoyers import Annoyer
from src.models.resnet import ResNetWithEmbedder
from src.utils import get_configuration
from src.datasets.coco import create_coco_dataloader
from src.datasets.coco import RetrievalCOCO

from torch.utils.data import DataLoader

from detectron2 import model_zoo
from detectron2.config import get_cfg
# from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model

# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MCV-M5-Project, week 4, run retrieval. Team 1'
    )
    # General settings
    parser.add_argument('--output_path', type=str, default='./outputs_task_e',
                        help='Path to the output directory.')
    # Dataset settings
    parser.add_argument('--dataset_path', type=str, default='../datasets/COCO',
                        help='Path to the dataset.')
    parser.add_argument('--dataset_config_path', type=str, default='./configs/coco.yaml',
                        help='Path to the dataset config file.')
    parser.add_argument('--model_path', type=str,
                        default='./outputs_task_e/models/ResNet_COCO_emb_256_bs_16_ep_1/model_final.pth',
                        help='Path to the Metric Learning model.')
    parser.add_argument('--retrieval_file', type=str, default='mcv_image_retrieval_annotations.json',
                        help='Path to the retrieval file.')
    # Metric Learning Model settings
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of the embedding vector.')
    # Retrieval settings
    parser.add_argument('--n_neighbors', type=int, default=50,
                        help='Number of nearest neighbors to retrieve.')
    parser.add_argument('--test_subset', type=str, default='val',
                        help="Subset of the dataset to use for testing, either 'val' or 'test'.")
    parser.add_argument('--min_objects', type=int, default=1,
                        help="Minimum number of objects in the image to be considered for good retrieval.")
    # Object Detection Model settings
    parser.add_argument("--model", "-mo", type=str, default="faster_rcnn",
                        help="Model (mask_rcnn, faster_rcnn)")
    parser.add_argument("--checkpoint", "-ch", type=str, default=None,
                        help="Model weights path")

    args = parser.parse_args()
    return args


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
    # cfg.OUTPUT_DIR = args.output_dir
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    return cfg


def histograms_intersection(hist1, hist2):
    return np.sum(np.minimum(hist1, hist2))


def run_experiment(database_dataloader, test_dataloader, model, embed_size, n_neighbors, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Annoyer
    annoy = Annoyer(model, database_dataloader, emb_size=embed_size,
                    device=device, distance='angular')
    # TODO: Add experiment_name!
    try:
        annoy.load()
    except:
        logging.info("Fitting annoy index...")
        annoy.state_variables['built'] = False
        annoy.fit()

    # Object Detection Model
    cfg = get_base_cfg(args)
    # predictor = DefaultPredictor(cfg)
    predictor = build_model(cfg)
    predictor.eval()
    # predictor.summary()
    num_cats = 91

    # Metrics
    mavep = []
    mavep25 = []
    top_1_acc = []
    top_5_acc = []
    top_10_acc = []

    embeds = []
    for idx in tqdm(range(len(test_dataloader.dataset))):
        print(f"Test dataloader idx: {idx}")
        query, label_query = test_dataloader.dataset[idx]

        print(f"Query shape: {query.shape}")
        # query = torch.unsqueeze(query, 0)
        print(f"Query shape: {query.shape}")
        with torch.no_grad():
            outputs = predictor([{"image": query}])
        # outputs = predictor(query)
        print(f"Outputs: {outputs}")

        pred_classes = outputs['instances'].pred_classes.cpu().tolist()
        label_query_hist = np.bincount(pred_classes, minlength=num_cats)

        V = model(query.unsqueeze(0).to(device)).squeeze()
        embeds.append(V)

        query = (return_image_full_range(query))
        nns, distances = annoy.retrieve_by_vector(
            V, n=n_neighbors, include_distances=True)
        labels = list()

        for nn in nns:
            _, label = database_dataloader.dataset[nn]
            label_hist = np.bincount(label, minlength=num_cats)
            intersection = histograms_intersection(label_query_hist, label_hist)
            labels.append(int(intersection >= args.min_objects))

        mavep.append(calculate_mean_average_precision(labels, distances))
        mavep25.append(calculate_mean_average_precision(
            labels[:26], distances[:26]))
        top_1_acc.append(calculate_top_k_accuracy(labels, k=1))
        top_5_acc.append(calculate_top_k_accuracy(labels, k=5))
        top_10_acc.append(calculate_top_k_accuracy(labels, k=10))

    print("Metrics: ",
          f"\n\tmAveP@50: {np.mean(mavep)}",
          f"\n\tmAveP@25: {np.mean(mavep25)}",
          f"\n\ttop_1 - precision: {np.mean(top_1_acc)}",
          f"\n\ttop_5 - precision: {np.mean(top_5_acc)}",
          f"\n\ttop_10 - precision: {np.mean(top_10_acc)}"
          )


def main(args: argparse.Namespace):
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    # Dataset loading
    dataset_config = get_configuration(args.dataset_config_path)

    train_dataloader, val_dataloader = create_coco_dataloader(
        1, args.dataset_path, dataset_config
    )
    train_ds = train_dataloader.dataset
    val_ds = val_dataloader.dataset
    logging.info(f"Train dataset size: {len(train_ds)}")
    logging.info(f"Val dataset size: {len(val_ds)}")

    # Retrieval dataset
    database_dataset = RetrievalCOCO(
        coco_dataset=train_ds,
        json_file=os.path.join(args.dataset_path, args.retrieval_file),
        subset="database",
        config=dataset_config,
    )
    database_loader = DataLoader(database_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Test dataset, either val or test (conceptually the same)
    test_dataset = RetrievalCOCO(
        coco_dataset=val_ds,
        json_file=os.path.join(args.dataset_path, args.retrieval_file),
        subset=args.test_subset,
        config=dataset_config,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load torch model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetWithEmbedder(resnet='18', embed_size=args.embedding_size)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    run_experiment(
        database_dataloader=database_loader,
        test_dataloader=test_loader,
        model=model,
        embed_size=args.embedding_size,
        n_neighbors=args.n_neighbors,
        args=args,
    )


if __name__ == "__main__":
    args = __parse_args()
    main(args)
