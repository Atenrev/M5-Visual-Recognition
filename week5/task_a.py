import tqdm
import torch
import random
import argparse
import numpy as np
import matplotlib

from datetime import datetime

from src.losses import SymmetricCrossEntropyLoss
from src.metrics import LossMetric
from src.trackers.wandb_tracker import WandbTracker
from src.trackers.tracker import Stage, ExperimentTracker
from src.datasets.coco import create_dataloader as create_coco_dataloader
from src.models.triplet_nets import TripletModel, ImageToTextTripletModel, ImageToTextWithTempModel
from src.models.resnet import ResNetWithEmbedder
from src.models.clip_text_encoder import CLIPTextEncoder


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MCV-M5-Project, week 4, tasks b and c. Team 1'
    )

    # General configuration
    parser.add_argument('--output_path', type=str, default='./outputs',
                        help='Path to the output directory.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for the experiment.')
    # Dataset configuration
    parser.add_argument('--dataset_path', type=str, default='./datasets/COCO',
                        help='Path to the dataset.')
    # Model configuration
    parser.add_argument('--image_encoder', type=str, default='resnet_18',
                        help='Model to use. Options: resnet_X, vgg.')
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
                        help='Optimizer to use. Options: adam, sgd.')
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


RUN_COUNT = 0


def run_epoch(dataloader, model, loss_fn, optimizer, device, train=True, tracker=None) -> dict:
    import time
    global RUN_COUNT

    if train:
        model.train()
    else:
        model.eval()

    metrics = {'loss': LossMetric()}
    start = time.time()

    # Print loss with tqdm
    for batch in (pbar := tqdm.tqdm(dataloader, desc='Epoch', leave=False)):
        end = time.time()
        print("dataloading took", end - start)
        anchors, positives, negatives = batch
        anchors = anchors.to(device)
        positives = model.tokenize(positives).to(device)
        negatives = model.tokenize(negatives).to(device)

        # Forward
        if isinstance(model, TripletModel):
            embeddings = model(anchors, positives.input_ids, positives.attention_mask,
                            negatives.input_ids, negatives.attention_mask)
            loss = loss_fn(*embeddings)
        else:
            start = time.time()
            logits = model(anchors, positives.input_ids, positives.attention_mask)
            end = time.time()
            print("model took", end - start)
            loss = loss_fn(logits)

        # Backward
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Metrics
        metrics['loss'].update(loss.item())

        # Log metrics
        if tracker is not None:
            for i, metric in enumerate(metrics.values()):
                tracker.add_batch_metric(
                    metric.name, metric.values[-1], RUN_COUNT, commit=i == len(metrics) - 1)

        RUN_COUNT += 1
        pbar.set_postfix(
            {metric_name: metric_value.values[-1] for metric_name, metric_value in metrics.items()})
        start = time.time()

    return {metric_name: metric_value.average for metric_name, metric_value in metrics.items()}


def train(train_dataloader, val_dataloader, model, loss_fn, optimizer, device, tracker: ExperimentTracker = None, current_epoch: int = 0):
    best_val_value = np.inf

    for epoch in range(current_epoch, args.epochs):
        # Train
        tracker.set_stage(Stage.TRAIN)
        metrics_train = run_epoch(
            train_dataloader, model, loss_fn, optimizer, device, train=True, tracker=tracker)

        for metric_name, metric_value in metrics_train.items():
            tracker.add_epoch_metric(
                metric_name, metric_value, epoch)

        # Validate
        tracker.set_stage(Stage.VAL)
        with torch.no_grad():
            metrics_val = run_epoch(
                val_dataloader, model, loss_fn, optimizer, device, train=False, tracker=tracker)

        for metric_name, metric_value in metrics_val.items():
            tracker.add_epoch_metric(
                metric_name, metric_value, epoch)

        # Save checkpoint
        if metrics_val['loss'] < best_val_value:
            best_val_value = metrics_val['loss']
            tracker.save_checkpoint(
                epoch,
                model,
                optimizer
            )

        summary = f"Epoch {epoch}/{args.epochs} - Train loss: {metrics_train['loss']:.4f} - Val loss: {metrics_val['loss']:.4f}"
        print("\n", summary, "\n")

        tracker.flush()

    tracker.finish()


def main(args: argparse.Namespace):
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Build config dict from args
    config = vars(args)
    config["metrics"] = []
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
    )
    # Create dummy data for testing
    # def create_dummy_data():
    #     import string
    #     anchors = torch.randn((100, 3, 224, 224))
    #     # generate random strings
    #     positives = ["".join(random.choices(string.ascii_letters, k=80)) for _ in range(100)]
    #     negatives = ["".join(random.choices(string.ascii_letters, k=80)) for _ in range(100)]
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
    text_encoder = CLIPTextEncoder(embed_size=args.embedding_size)

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

    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train(train_dataloader, val_dataloader, model,
          loss_fn, optimizer, device, tracker=tracker)


if __name__ == "__main__":
    matplotlib.use('Agg')
    args = __parse_args()
    main(args)
