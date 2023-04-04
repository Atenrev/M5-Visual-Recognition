import argparse


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MCV-M5-Project, week 4, tasks b and c. Team 1'
    )

    parser.add_argument('--dataset', type=str, default='MIT_split',
                        help='Dataset to use. Options: MIT_split, COCO')
    parser.add_argument('--dataset_path', type=str, default='./datasets/MIT_split',
                        help='Path to the dataset')
    parser.add_argument('--model', type=str, default='resnet',
                        help='Model to use. Options: resnet, vgg')
    parser.add_argument('--loss', type=str, default='contrastive',
                        help='Loss to use. Options: contrastive, triplet')
    parser.add_argument('--pos_margin', type=float, default=0.0,
                        help='Positive margin for contrastive loss')
    parser.add_argument('--neg_margin', type=float, default=1.0,
                        help='Negative margin for contrastive loss')
    parser.add_argument('--distance', type=str, default='euclidean',
                        help='Distance to use. Options: euclidean, cosine')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use. Options: adam, sgd')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    


if __name__ == "__main__":
    args = __parse_args()
    main(args)
