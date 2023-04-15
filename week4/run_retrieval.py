import os
import argparse
import numpy as np
from tqdm import tqdm

from src.metrics import *
from src.utils import return_image_full_range
from src.datasets.mit_split import create_mit_dataloader
from src.methods.annoyers import Annoyer, SKNNWrapper
from src.models.resnet import ResNetWithEmbedder
from src.models.vgg import VGG19
from src.utils import get_configuration


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MCV-M5-Project, week 4, run retrieval. Team 1'
    )

    parser.add_argument('--dataset_config_path', type=str, default='./configs/mit_split.yaml',
                        help='Path to the dataset config file.')
    parser.add_argument('--dataset_path', type=str, default='./datasets/MIT_split',
                        help='Path to the dataset.')
    parser.add_argument('--model', type=str, default='resnet_18',
                        help='Model to use. Options: resnet_X, vgg.')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of the embedding vector.')
    parser.add_argument('--model_weights', type=str, default=None,
                        help='Path to the model weights.')
    parser.add_argument('--n_neighbors', type=int, default=50,
                        help='Number of nearest neighbors to retrieve.')

    args = parser.parse_args()
    return args


def run_experiment(model, embed_size, n_neighbors=50):
    device = 'cuda'

    dataset_config = get_configuration(args.dataset_config_path)
    train_ds, test_ds = create_mit_dataloader(1, '../datasets/MIT_split/', dataset_config, inference=False)

    # Annoyer
    annoy = Annoyer(model, train_ds, emb_size=embed_size, device=device, distance='angular')
    try: 
        annoy.load()
    except:
        annoy.state_variables['built'] = False
        annoy.fit()

    # Metrics
    mavep = []
    mavep25 = []
    top_1_acc = []
    top_5_acc = []
    top_10_acc = []

    embeds = []
    for idx in tqdm(range(len(test_ds.dataset))):
        query, label_query = test_ds.dataset[idx]

        V = model(query.unsqueeze(0).to(device)).squeeze()
        embeds.append(V)

        query = (return_image_full_range(query))
        nns, distances = annoy.retrieve_by_vector(V, n=n_neighbors, include_distances=True)
        labels = list()

        for nn in nns:
            _, label = train_ds.dataset[nn]
            labels.append(int(label == label_query))

        mavep.append(calculate_mean_average_precision(labels, distances))
        mavep25.append(calculate_mean_average_precision(labels[:26], distances[:26]))
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
    if args.model.split("_")[0] == 'resnet':
        model = ResNetWithEmbedder(resnet=args.model.split("_")[
                                   1], embed_size=args.embedding_size)
    elif args.model == 'vgg':
        model = VGG19()
    else:
        raise ValueError(f'Unknown model: {args.model}')
    
    run_experiment(model, args.embedding_size, args.n_neighbors)


if __name__ == "__main__":
    args = __parse_args()
