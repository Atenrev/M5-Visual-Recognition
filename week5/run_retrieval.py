import os
import argparse
import numpy as np
import matplotlib
import torch
import random

from tqdm import tqdm
from sklearn.preprocessing import label_binarize

from src.metrics import (
    calculate_mean_average_precision, 
    calculate_top_k_accuracy, 
)
from src.methods.annoyers import Annoyer
from src.models.resnet import ResNetWithEmbedder
from src.metrics import plot_prec_rec_curve_multiclass
from src.models.triplet_nets import ImageToTextTripletModel, SymmetricSiameseModel, TextToImageTripletModel
from src.models.resnet import ResNetWithEmbedder
from src.models.bert_text_encoder import BertTextEncoder
from src.models.clip_text_encoder import CLIPTextEncoder
from src.datasets.coco import create_dataloader as create_coco_dataloader
from src.datasets.dummy import create_dataloader as create_dummy_dataloader


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MCV-M5-Project, week 5, run retrieval. Team 1'
    )

    parser.add_argument('--dataset_path', type=str, default='./datasets/COCO',
                        help='Path to the dataset.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for the experiment.')
    # Annoyer
    parser.add_argument('--n_neighbors', type=int, default=50,
                        help='Number of nearest neighbors to retrieve.')
    # Model
    parser.add_argument('--checkpoint', type=str, default="./checkpoints/epoch_1.pt",
                        help='Path to the checkpoint to load.')
    parser.add_argument('--mode', type=str, default='symmetric',
                        help='Mode to use. Options: image_to_text, text_to_image, symmetric.')
    parser.add_argument('--image_encoder', type=str, default='resnet_18',
                        help='Image Encoder to use. Options: resnet_X, vgg.')
    parser.add_argument('--text_encoder', type=str, default='clip',
                        help='Text Encoder to use. Options: clip, bert.')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of the embedding vector.')

    args = parser.parse_args()
    return args


def run_experiment(train_dataloader, test_dataloader, model, embed_size, n_neighbors=50, experiment_name='resnet_base', device='cuda'):
    # TODO: Adapt this function to this dataset and modality
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
    experiment_name = f'{args.image_encoder}_{args.text_encoder}_{args.mode}_embed_{args.embedding_size}_nneighbors_{args.n_neighbors}'
   # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load data
    train_dataloader, val_dataloader, _ = create_coco_dataloader(
        args.dataset_path,
        args.batch_size,
        inference=False,
        test_mode=True, # TODO: Change to False!!!
    )
    # Create dummy data for testing
    # val_dataloader = create_dummy_dataloader(args)

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

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    model.to(device)

    with torch.no_grad():
        mavep, mavep25, top_1_acc, top_5_acc, top_10_acc = run_experiment(
            train_dataloader, val_dataloader, model, 
            args.embedding_size, args.n_neighbors, experiment_name=experiment_name)
        
    with open(os.path.join(args.checkpoint, "../", "results.txt"), "w") as f:
        f.write(f"mAveP@50: {mavep}\n")
        f.write(f"mAveP@25: {mavep25}\n")
        f.write(f"top_1 - precision: {top_1_acc}\n")
        f.write(f"top_5 - precision: {top_5_acc}\n")
        f.write(f"top_10 - precision: {top_10_acc}\n")


if __name__ == "__main__":
    matplotlib.use('Agg')
    args = __parse_args()
    main(args)
