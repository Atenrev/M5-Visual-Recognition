import torch
import random
import argparse
import numpy as np
import matplotlib

from datetime import datetime

from src.trainer import train
from src.trackers.wandb_tracker import WandbTracker
from src.models.triplet_nets import ImageToTextTripletModel, TextToImageTripletModel, SymmetricSiameseModel
from src.models.resnet import ResNetWithEmbedder
from src.models.bert_text_encoder import BertTextEncoder
from src.models.clip_text_encoder import CLIPTextEncoder
from src.datasets.coco import create_dataloader as create_coco_dataloader


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MCV-M5-Project, week 5, task a. Team 1'
    )

    # General configuration
    parser.add_argument('--output_path', type=str, default='./outputs',
                        help='Path to the output directory.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for the experiment.')
    parser.add_argument('--mode', type=str, default='symmetric',
                        help='Mode to use. Options: image_to_text, text_to_image, symmetric.')
    # Dataset configuration
    parser.add_argument('--dataset_path', type=str, default='./datasets/COCO',
                        help='Path to the dataset.')
    parser.add_argument('--dataset_percentage', type=float, default=1.0,
                        help='Percentage of the dataset to use.')
    parser.add_argument('--random_subset', type=bool, default=False,
                        help='Whether to use a random subset of the dataset.')
    # Model configuration
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to the checkpoint to load.')
    parser.add_argument('--image_encoder', type=str, default='resnet_18',
                        help='Model to use. Options: resnet_X, vgg.')
    parser.add_argument('--text_encoder', type=str, default='clip',
                        help='Model to use. Options: clip, bert.')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of the embedding vector.')
    # Loss configuration
    # parser.add_argument('--loss', type=str, default='symmetric',
    #                     help='Loss function to use. Options: triplet, symmetric.')
    parser.add_argument('--triplet_margin', type=float, default=0.05,
                        help='Margin for triplet loss.')
    parser.add_argument('--triplet_norm', type=int, default=2,
                        help='Norm for triplet loss.')
    # Training configuration
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use. Options: adam.') # No more options, sorry
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate for the trunk.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay.')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Build config dict from args
    config = vars(args)
    config["metrics"] = ["accuracy"]
    experiment_name = f"{args.mode}_{args.image_encoder}_{args.text_encoder}_embed{args.embedding_size}_lr{args.lr}_wd{args.weight_decay}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tracker = WandbTracker(
        log_path="outputs_w5_task_a",
        experiment_name=experiment_name,
        config=config,
        project_name="m5_week1_task_a"
    )

    # Load data
    train_dataloader, val_dataloader, _ = create_coco_dataloader(
        args.dataset_path,
        args.batch_size,
        inference=False,
        mode=args.mode,
        percentage=args.dataset_percentage,
        random_subset=args.random_subset,
    )
    # Create dummy data for testing.
    # train_dataloader, val_dataloader, _ = create_dummy_dataloader(args)

    # Create model
    # Remember to make sure both models project to the same embedding space
    image_encoder = ResNetWithEmbedder(embed_size=args.embedding_size)

    if args.text_encoder == 'clip':
        text_encoder = CLIPTextEncoder(embed_size=args.embedding_size)
    elif args.text_encoder == 'bert':
        text_encoder = BertTextEncoder(embed_size=args.embedding_size)
    else:
        raise ValueError(f"Unknown text encoder {args.text_encoder}")

    if args.mode == 'symmetric':
        model = SymmetricSiameseModel(
            image_encoder,
            text_encoder,
            args,
        )
    elif args.mode == 'image_to_text':
        model = ImageToTextTripletModel(
            image_encoder,
            text_encoder,
            args
        )
    elif args.mode == 'text_to_image':
        model = TextToImageTripletModel(
            image_encoder,
            text_encoder,
            args
        )
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Load checkpoint
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    model.to(device)

    current_epoch = 0 if checkpoint is None else checkpoint['epoch']
    train(train_dataloader, val_dataloader, model, optimizer,
          device, args.epochs, tracker=tracker, current_epoch=current_epoch)


if __name__ == "__main__":
    matplotlib.use('Agg')
    args = __parse_args()
    main(args)
