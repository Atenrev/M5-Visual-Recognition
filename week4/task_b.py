import os
import torch
import argparse
import pytorch_metric_learning as pml

from src.utils import get_configuration
from src.datasets.mit_split import create_mit_dataloader
from src.models.resnet import ResNet
from src.models.vgg import VGG19


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MCV-M5-Project, week 4, tasks b and c. Team 1'
    )

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='mit_split',
                        help='Dataset to use. Options: mit_split, coco.')
    parser.add_argument('--dataset_config_path', type=str, default='./configs/mit_split.yaml',
                        help='Path to the dataset config file.')
    parser.add_argument('--dataset_path', type=str, default='./datasets/MIT_split',
                        help='Path to the dataset.')
    # Model configuration
    parser.add_argument('--model', type=str, default='resnet',
                        help='Model to use. Options: resnet, vgg.')
    # Loss configuration
    parser.add_argument('--loss', type=str, default='contrastive',
                        help='Loss to use. Options: contrastive, triplet.')
    parser.add_argument('--pos_margin', type=float, default=0.0,
                        help='Positive margin for contrastive loss. Also used for triplet loss.')
    parser.add_argument('--neg_margin', type=float, default=1.0,
                        help='Negative margin for contrastive loss.')
    parser.add_argument('--distance', type=str, default='euclidean',
                        help='Distance to use. Options: euclidean, cosine.')
    # Training configuration
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use. Options: adam, sgd.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    device = 'cuda'

    # Model loading
    if args.model == 'resnet':
        model = ResNet(resnet='101', norm=None)
    elif args.model == 'vgg':
        model = VGG19()
    else:
        raise ValueError(f'Unknown model: {args.model}')

    model.to(device)

    # Dataset loading
    dataset_config = get_configuration(args.dataset_config_path)

    if args.dataset == 'mit_split':
        train_set, val_set, test_set = create_mit_dataloader(
            args.batch_size, args.dataset_path, dataset_config)
    elif args.dataset == 'coco':
        raise NotImplementedError
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    # Loss configuration
    if args.distance == 'euclidean':
        distance = pml.distances.LpDistance(p=2)
    elif args.distance == 'cosine':
        distance = pml.distances.CosineSimilarity()
    else:
        raise ValueError(f'Unknown distance: {args.distance}')

    if args.loss == 'contrastive':
        criterion = pml.losses.ContrastiveLoss(
            pos_margin=args.pos_margin,
            neg_margin=args.neg_margin,
            distance=distance
        )
    elif args.loss == 'triplet':
        criterion = pml.losses.TripletMarginLoss(
            margin=args.pos_margin,
            distance=distance,
        )
    else:
        raise ValueError(f'Unknown loss: {args.loss}')

    # Optimizer configuration
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')
    
    # Training
    trainer = pml.trainers.MetricLossOnly(
        models={"trunk": model},
        optimizers={"trunk_optimizer": optimizer},
        loss_funcs={"metric_loss": criterion},
        batch_size=args.batch_size,
        dataset=train_set,
        data_device=device,
    )
    trainer.train(num_epochs=args.epochs)

    # Save model
    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), f'./models/{args.model}_{args.dataset}_loss_{args.loss}_distance_{args.distance}_posmargin_{args.pos_margin}_negmargin_{args.neg_margin}.pt')


if __name__ == "__main__":
    args = __parse_args()
    main(args)
