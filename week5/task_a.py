import tqdm
import torch
import random
import argparse
import numpy as np
import matplotlib

from datetime import datetime

from src.losses import SymmetricCrossEntropyLoss
from src.trainer import train
from src.trackers.wandb_tracker import WandbTracker
from src.models.triplet_nets import ImageToTextTripletModel, ImageToTextWithTempModel
from src.models.resnet import ResNetWithEmbedder
from src.models.bert_text_encoder import BertTextEncoder
from src.models.clip_text_encoder import CLIPTextEncoder
from src.datasets.coco import create_dataloader as create_coco_dataloader


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MCV-M5-Project, week 4, tasks b and c. Team 1'
    )

    # General configuration
    parser.add_argument('--output_path', type=str, default='./outputs',
                        help='Path to the output directory.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for the experiment.')
    parser.add_argument('--mode', type=str, default='image_to_text',
                        help='Mode to use. Options: image_to_text, text_to_image.')
    # Dataset configuration
    parser.add_argument('--dataset_path', type=str, default='./datasets/COCO',
                        help='Path to the dataset.')
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
    parser.add_argument('--loss', type=str, default='symmetric',
                        help='Loss function to use. Options: triplet, symmetric.')
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
    experiment_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
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
    )
    # Create dummy data for testing.
    # def create_dummy_data():
    #     import string
    #     anchors = torch.randn((100, 3, 224, 224))
    #     # generate random strings
    #     positives = ["".join(random.choices(string.ascii_letters, k=80))
    #                  for _ in range(100)]
    #     negatives = ["".join(random.choices(string.ascii_letters, k=80))
    #                  for _ in range(100)]
    #     data = list(zip(anchors, positives, negatives))
    #     return data

    # train_dataloader = torch.utils.data.DataLoader(
    #     create_dummy_data(),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=4,
    # )
    # val_dataloader = torch.utils.data.DataLoader(
    #     create_dummy_data(),
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=4,
    # )

    # Create loss
    print(f"Using loss {args.loss}")
    if args.loss == 'triplet':
        loss_fn = torch.nn.TripletMarginLoss(
            margin=args.triplet_margin,
            p=args.triplet_norm
        )
    elif args.loss == 'symmetric':
        loss_fn = SymmetricCrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss {args.loss}")

    # Create model
    # Remember to make sure both models project to the same embedding space
    image_encoder = ResNetWithEmbedder(embed_size=args.embedding_size)

    if args.text_encoder == 'clip':
        text_encoder = CLIPTextEncoder(embed_size=args.embedding_size)
    elif args.text_encoder == 'bert':
        text_encoder = BertTextEncoder(embed_size=args.embedding_size)
    else:
        raise ValueError(f"Unknown text encoder {args.text_encoder}")

    if args.loss == 'symmetric':
        model = ImageToTextWithTempModel(
            image_encoder,
            text_encoder,
        )
    else:
        model = ImageToTextTripletModel(
            image_encoder,
            text_encoder,
        )

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Load checkpoint
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    model.to(device)

    current_epoch = 0 if args.checkpoint is None else checkpoint['epoch']
    train(train_dataloader, val_dataloader, model, loss_fn, 
          optimizer, device, args.epochs, tracker=tracker, current_epoch=current_epoch)


if __name__ == "__main__":
    matplotlib.use('Agg')
    args = __parse_args()
    main(args)
