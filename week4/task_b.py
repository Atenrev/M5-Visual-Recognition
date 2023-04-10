import os
import umap
import torch
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pytorch_metric_learning.utils.logging_presets as logging_presets
import matplotlib

from cycler import cycler
from pytorch_metric_learning import (
    testers, samplers, losses, distances, trainers, miners)
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from src.utils import get_configuration
from src.datasets.mit_split import create_mit_dataloader
from src.models.resnet import ResNetWithEmbedder
from src.models.vgg import VGG19


OUTPUT_PATH = './outputs'
EXPERIMENT_NAME = None


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MCV-M5-Project, week 4, tasks b and c. Team 1'
    )

    # General configuration
    parser.add_argument('--output_path', type=str, default='./outputs',
                        help='Path to the output directory.')
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='mit_split',
                        help='Dataset to use. Options: mit_split, coco.')
    parser.add_argument('--dataset_config_path', type=str, default='./week4/configs/mit_split.yaml',
                        help='Path to the dataset config file.')
    parser.add_argument('--dataset_path', type=str, default='./datasets/MIT_split',
                        help='Path to the dataset.')
    # Model configuration
    parser.add_argument('--model', type=str, default='resnet_18',
                        help='Model to use. Options: resnet_X, vgg.')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of the embedding vector.')
    # Loss configuration
    parser.add_argument('--loss', type=str, default='contrastive',
                        help='Loss to use. Options: contrastive, triplet.')
    parser.add_argument('--pos_margin', type=float, default=0.0,
                        help='Positive margin for contrastive loss. Also used for triplet loss.')
    parser.add_argument('--neg_margin', type=float, default=1.0,
                        help='Negative margin for contrastive loss.')
    parser.add_argument('--distance', type=str, default='euclidean',
                        help='Distance to use. Options: euclidean, cosine.')
    # Miner configuration
    parser.add_argument('--miner', type=str, default="PairMargin",
                        help='Miner to use. Options: BatchEasyHard, BatchHard, MultiSimilarity, PairMargin, TripletMargin.')
    parser.add_argument('--miner_pos_strategy', type=str, default='easy',
                        help='Positive strategy for the miner. Options: all, easy, semihard, hard.')
    parser.add_argument('--miner_neg_strategy', type=str, default='hard',
                        help='Negative strategy for the miner. Options: all, easy, semihard, hard.')
    parser.add_argument('--miner_epsilon', type=float, default=0.1,
                        help='Epsilon for the miner.')
    parser.add_argument('--miner_pos_margin', type=float, default=0.2,
                        help='Positive margin for the miner.')
    parser.add_argument('--miner_neg_margin', type=float, default=0.8,
                        help='Negative margin for the miner.')
    parser.add_argument('--miner_type_of_triplets', type=str, default='all',
                        help='Type of triplets for the miner. Options: all, easy, semihard, hard.')
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

    args = parser.parse_args()
    return args


def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, epoch, *args):
    global OUTPUT_PATH, EXPERIMENT_NAME
    output_dir = os.path.join(OUTPUT_PATH, "umap_plots")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(
        "UMAP plot for the {} split and label set {}".format(
            split_name, keyname)
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig = plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i)
                      for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0],
                 umap_embeddings[idx, 1], ".", markersize=1)
    plt.show()
    fig.savefig(os.path.join(output_dir, f"umap_{split_name}_{epoch}_{EXPERIMENT_NAME}.png"))


def main(args: argparse.Namespace):
    global OUTPUT_PATH, EXPERIMENT_NAME
    os.makedirs(args.output_path, exist_ok=True)
    OUTPUT_PATH = args.output_path
    experiment_name = f"{args.model}_{args.dataset}_loss_{args.loss}_distance_{args.distance}_posmargin_{args.pos_margin}_negmargin_{args.neg_margin}"
    EXPERIMENT_NAME = experiment_name
    model_folder = os.path.join(args.output_path, "models")
    os.makedirs(model_folder, exist_ok=True)
    logs_folder = os.path.join(args.output_path, "logs", experiment_name)
    os.makedirs(logs_folder, exist_ok=True)
    tensorboard_folder = os.path.join(
        args.output_path, "tensorboard", experiment_name)
    os.makedirs(tensorboard_folder, exist_ok=True)
    device = 'cuda'

    # Model loading
    if args.model.split("_")[0] == 'resnet':
        model = ResNetWithEmbedder(resnet=args.model.split("_")[
                                   1], embed_size=args.embedding_size)
    elif args.model == 'vgg':
        model = VGG19()
    else:
        raise ValueError(f'Unknown model: {args.model}')

    model.to(device)

    # Dataset loading
    dataset_config = get_configuration(args.dataset_config_path)

    if args.dataset == 'mit_split':
        train_dataloader, val_dataloader = create_mit_dataloader(
            args.batch_size, args.dataset_path, dataset_config)
        train_ds = train_dataloader.dataset
        val_ds = val_dataloader.dataset
    elif args.dataset == 'coco':
        raise NotImplementedError
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    class_sampler = samplers.MPerClassSampler(
        labels=train_ds.targets,
        m=args.batch_size // 8,
        batch_size=args.batch_size,
        length_before_new_iter=len(train_ds),
    )

    # Loss configuration
    if args.distance == 'euclidean':
        distance = distances.LpDistance(p=2)
    elif args.distance == 'cosine':
        distance = distances.CosineSimilarity()
    else:
        raise ValueError(f'Unknown distance: {args.distance}')

    if args.loss == 'contrastive':
        criterion = losses.ContrastiveLoss(
            pos_margin=args.pos_margin,
            neg_margin=args.neg_margin,
            distance=distance
        )
    elif args.loss == 'triplet':
        criterion = losses.TripletMarginLoss(
            margin=args.pos_margin,
            distance=distance,
        )
    else:
        raise ValueError(f'Unknown loss: {args.loss}')

    # Miner configuration
    if args.miner == "BatchEasyHard":
        miner = miners.BatchEasyHardMiner(
            pos_strategy=args.miner_pos_strategy,
            neg_strategy=args.miner_neg_strategy,
        )
    elif args.miner == "BatchHard":
        miner = miners.BatchHardMiner()
    elif args.miner == "MultiSimilarity":
        miner = miners.MultiSimilarityMiner(
            epsilon=args.miner_epsilon,
        )
    elif args.miner == "PairMargin":
        miner = miners.PairMarginMiner(
            pos_margin=args.miner_pos_margin,
            neg_margin=args.miner_neg_margin,
        )
    elif args.miner == "TripletMargin":
        miner = miners.TripletMarginMiner(
            margin=args.miner_pos_margin,
            type_of_triplets=args.miner_type_of_triplets,
        )

    # Optimizer configuration
    if args.optimizer == 'adam':
        trunk_optimizer = torch.optim.Adam(
            model.trunk.parameters(), lr=args.lr_trunk)
        embedder_optimizer = torch.optim.Adam(
            model.embedder.parameters(), lr=args.lr_embedder)
    elif args.optimizer == 'sgd':
        trunk_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_trunk)
        embedder_optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr_embedder)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    # Hooks
    record_keeper, _, _ = logging_presets.get_record_keeper(
        logs_folder, tensorboard_folder)
    hooks = logging_presets.get_hook_container(record_keeper)

    # Create the tester
    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        visualizer=umap.UMAP(),
        visualizer_hook=visualizer_hook,
        dataloader_num_workers=1,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
    )

    end_of_epoch_hook = hooks.end_of_epoch_hook(
        tester,
        {"val": val_ds},
        model_folder,
        test_interval=1,
        patience=1,
    )

    # Training
    trainer = trainers.MetricLossOnly(
        models={"trunk": model.trunk, "embedder": model.embedder},
        optimizers={"trunk_optimizer": trunk_optimizer,
                    "embedder_optimizer": embedder_optimizer},
        loss_funcs={"metric_loss": criterion},
        mining_funcs={"tuple_miner": miner},
        data_device=device,
        dataset=train_ds,
        batch_size=args.batch_size,
        sampler=class_sampler,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )
    trainer.train(num_epochs=args.epochs)

    # Save model
    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), os.path.join(
        model_folder, f'{experiment_name}.pth'))


if __name__ == "__main__":
    matplotlib.use('Agg')
    args = __parse_args()
    main(args)
