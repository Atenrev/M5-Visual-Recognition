import os
import argparse
import numpy as np
import matplotlib
import torch
from tqdm import tqdm
from sklearn.preprocessing import label_binarize

from src.metrics import *
from src.utils import return_image_full_range
from src.datasets.mit_split import create_mit_dataloader
from src.methods.annoyers import Annoyer, SKNNWrapper
from src.models.resnet import ResNetWithEmbedder
from src.models.vgg import VGG19
from src.utils import get_configuration
from src.metrics import plot_prec_rec_curve_multiclass


LABELS = ['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding']


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MCV-M5-Project, week 4, run retrieval. Team 1'
    )

    parser.add_argument('--class_name', type=str, default='forest',
                        help='Class name to retrieve.')
    parser.add_argument('--dataset_config_path', type=str, default='./week4/configs/mit_split.yaml',
                        help='Path to the dataset config file.')
    parser.add_argument('--dataset_path', type=str, default='./datasets/MIT_split',
                        help='Path to the dataset.')
    parser.add_argument('--model', type=str, default='resnet_18',
                        help='Model to use. Options: resnet_X, vgg.')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of the embedding vector.')
    parser.add_argument('--model_weights_path', type=str, default="models/contrastive_pairmargin", # triplet_tripletmargin
                        help='Path to the model weights.')
    parser.add_argument('--n_neighbors', type=int, default=50,
                        help='Number of nearest neighbors to retrieve.')

    args = parser.parse_args()
    return args


def do_retrieval(class_name, train_dataloader, test_dataloader, model, embed_size, n_neighbors=50, experiment_name='resnet_base', device='cuda'):
    model = model.to(device)
    
    # Annoyer
    annoy = Annoyer(model, train_dataloader, emb_size=embed_size,
                    device=device, distance='angular', experiment_name=experiment_name)
    try:
        annoy.load()
    except:
        annoy.state_variables['built'] = False
        annoy.fit()

    embeds = []
    idx = np.random.randint(0, len(test_dataloader.dataset))
    query, label_query = test_dataloader.dataset[idx]

    while LABELS[label_query] != class_name:
        idx = np.random.randint(0, len(test_dataloader.dataset))
        query, label_query = test_dataloader.dataset[idx]

    V = model(query.unsqueeze(0).to(device)).squeeze()
    embeds.append(V)

    query = (return_image_full_range(query))
    nns, distances = annoy.retrieve_by_vector(
        V, n=n_neighbors, include_distances=True)
    distances = np.array(distances)
    labels = list()
    labels_pred = list()
    images = list()
    true_positives = list()

    for nn in nns:
        im, label = train_dataloader.dataset[nn]
        labels.append(int(label == label_query))
        labels_pred.append(label)
        images.append(return_image_full_range(im))
        true_positives.append(int(label == label_query))

    out_path = f"./week4/results/{experiment_name}/retrieval_{class_name}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plot_retrieved_images(query, images, labels, out=out_path)


def main(args: argparse.Namespace):
    # Dataloader
    dataset_config = get_configuration(args.dataset_config_path)
    train_ds, test_ds = create_mit_dataloader(
        1, args.dataset_path, dataset_config, inference=False)

    # Model
    if args.model.split("_")[0] == 'resnet':
        model = ResNetWithEmbedder(resnet=args.model.split("_")[1],
                                   embed_size=args.embedding_size)
    elif args.model == 'vgg':
        model = VGG19()
    else:
        raise ValueError(f'Unknown model: {args.model}')

    model.load_state_dict(torch.load(os.path.join(args.model_weights_path, "model_final.pth")))
    model.eval()
    experiment_name = args.model_weights_path.split("/")[-1]

    with torch.no_grad():
        do_retrieval(
            args.class_name, train_ds, test_ds, model, 
            args.embedding_size, args.n_neighbors, experiment_name=experiment_name)


if __name__ == "__main__":
    matplotlib.use('Agg')
    args = __parse_args()
    main(args)
