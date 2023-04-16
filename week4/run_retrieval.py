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

    parser.add_argument('--dataset_config_path', type=str, default='./week4/configs/mit_split.yaml',
                        help='Path to the dataset config file.')
    parser.add_argument('--dataset_path', type=str, default='./datasets/MIT_split',
                        help='Path to the dataset.')
    parser.add_argument('--model', type=str, default='resnet_18',
                        help='Model to use. Options: resnet_X, vgg.')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of the embedding vector.')
    parser.add_argument('--model_weights_path', type=str, default="models/",
                        help='Path to the model weights.')
    parser.add_argument('--n_neighbors', type=int, default=50,
                        help='Number of nearest neighbors to retrieve.')

    args = parser.parse_args()
    return args


def run_experiment(train_dataloader, test_dataloader, model, embed_size, n_neighbors=50, experiment_name='resnet_base', device='cuda'):
    model = model.to(device)
    
    # Annoyer
    annoy = Annoyer(model, train_dataloader, emb_size=embed_size,
                    device=device, distance='angular', experiment_name=experiment_name)
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
    Y_gt = []
    Y_probs = []
    for idx in tqdm(range(len(test_dataloader.dataset))):
        idx = np.random.randint(0, len(test_dataloader.dataset))
        query, label_query = test_dataloader.dataset[idx]

        V = model(query.unsqueeze(0).to(device)).squeeze()
        embeds.append(V)

        query = (return_image_full_range(query))
        nns, distances = annoy.retrieve_by_vector(
            V, n=n_neighbors, include_distances=True)
        labels = list()
        labels_pred = list()

        for nn in nns:
            _, label = train_dataloader.dataset[nn]
            labels.append(int(label == label_query))
            labels_pred.append(label)

        Y_gt.append(label_binarize([label_query], classes=[*range(8)])[0])
        
        distances = np.array(distances)
        labels_pred = np.array(labels_pred)

        # Compute probabilities per class
        weights = 1 / (distances + 1e-6)
        weighted_counts = np.bincount(labels_pred, weights=weights, minlength=8)
        probabilities = weighted_counts 
        Y_probs.append(probabilities)

        mavep.append(calculate_mean_average_precision(labels, distances))
        mavep25.append(calculate_mean_average_precision(
            labels[:26], distances[:26]))
        top_1_acc.append(calculate_top_k_accuracy(labels, k=1))
        top_5_acc.append(calculate_top_k_accuracy(labels, k=5))
        top_10_acc.append(calculate_top_k_accuracy(labels, k=10))

    Y_gt = np.array(Y_gt)
    Y_probs = np.array(Y_probs)
    output_path = f'./week4/plots/{experiment_name}_prec_rec_curve.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_prec_rec_curve_multiclass(Y_gt, Y_probs, LABELS, output_path=output_path)

    print("Metrics: ",
          f"\n\tmAveP@50: {np.mean(mavep)}",
          f"\n\tmAveP@25: {np.mean(mavep25)}",
          f"\n\ttop_1 - precision: {np.mean(top_1_acc)}",
          f"\n\ttop_5 - precision: {np.mean(top_5_acc)}",
          f"\n\ttop_10 - precision: {np.mean(top_10_acc)}"
          )
    return np.mean(mavep), np.mean(mavep25), np.mean(top_1_acc), np.mean(top_5_acc), np.mean(top_10_acc)


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

    for model_dir in os.listdir(args.model_weights_path)[2:3]:
        print(f"Running experiment: {model_dir}")
        model.load_state_dict(torch.load(os.path.join(args.model_weights_path, model_dir, "model_final.pth")))
        model.eval()

        with torch.no_grad():
             mavep, mavep25, top_1_acc, top_5_acc, top_10_acc = run_experiment(
                 train_ds, test_ds, model, 
                 args.embedding_size, args.n_neighbors, experiment_name=model_dir)
            
        with open(os.path.join(args.model_weights_path, model_dir, "results.txt"), "w") as f:
            f.write(f"mAveP@50: {mavep}\n")
            f.write(f"mAveP@25: {mavep25}\n")
            f.write(f"top_1 - precision: {top_1_acc}\n")
            f.write(f"top_5 - precision: {top_5_acc}\n")
            f.write(f"top_10 - precision: {top_10_acc}\n")


if __name__ == "__main__":
    matplotlib.use('Agg')
    args = __parse_args()
    main(args)
