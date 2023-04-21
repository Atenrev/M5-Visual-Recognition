import os
import argparse
import numpy as np
import matplotlib
import torch
import random
from tqdm import tqdm
from datetime import datetime

from src.metrics import (
    calculate_mean_average_precision,
    calculate_top_k_accuracy,
)
from src.methods.annoyers import Annoyer
from src.models.resnet import ResNetWithEmbedder
from src.models.triplet_nets import ImageToTextTripletModel, SymmetricSiameseModel, TextToImageTripletModel
from src.models.resnet import ResNetWithEmbedder
from src.models.bert_text_encoder import BertTextEncoder
from src.models.clip_text_encoder import CLIPTextEncoder
from src.datasets.coco import create_dataloader as create_coco_dataloader


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MCV-M5-Project, week 5, run retrieval. Team 1'
    )

    parser.add_argument('--mode', type=str, required=True,
                        help='Mode to use. Options: image_to_text, text_to_image, symmetric.')
    parser.add_argument('--dataset_path', type=str, default='./datasets/COCO',
                        help='Path to the dataset.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for the experiment.')
    # Annoyer
    parser.add_argument('--n_neighbors', type=int, default=50,
                        help='Number of nearest neighbors to retrieve.')
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the checkpoint to load.')
    parser.add_argument('--mode', type=str, required=True,
                        help='Mode to use. Options: image_to_text, text_to_image, symmetric.')
    parser.add_argument('--image_encoder', type=str, required=True,
                        help='Image Encoder to use. Options: resnet_X, vgg.')
    parser.add_argument('--text_encoder', type=str, required=True,
                        help='Text Encoder to use. Options: clip, bert.')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='Size of the embedding vector.')

    args = parser.parse_args()
    return args


def run_experiment(
    dataloader, model, embed_size, mode, n_neighbors=50, experiment_name='resnet_base', device='cuda'
):
    # Model
    model = model.to(device)
    if mode == 'image_to_text' or mode == 'symmetric':
        embedder_query = model.image_encoder
        embedder_database = model.text_encoder
    elif mode == 'text_to_image':
        embedder_query = model.text_encoder
        embedder_database = model.image_encoder
    else:  # TODO: implement symmetric
        raise NotImplementedError(f"Mode {mode} not supported.")

    # Annoyer
    annoy = Annoyer(embedder_database, dataloader, emb_size=embed_size,
                    device=device, distance='angular', experiment_name=experiment_name)
    try:
        annoy.load()
    except:
        annoy.state_variables['built'] = False
        annoy.fit()

    # Metrics
    mavep, mavep25 = [], []
    top_1_acc, top_5_acc, top_10_acc = []

    for idx in tqdm(range(len(dataloader.dataset))):
        anchor, _, _ = dataloader.dataset[idx]

        print("anchor.shape ", anchor.shape)
        if type(anchor[0]) == str:  # Text2Image
            anchor = embedder_query.tokenize(anchor).to(device)
            V = embedder_query(anchor.input_ids, anchor.attention_mask).squeeze()
        else:  # Image2Text
            anchor = anchor.to(device)
            V = embedder_query(anchor).squeeze()

        nns, distances = annoy.retrieve_by_vector(
            V, n=n_neighbors, include_distances=True,
        )

        labels = []
        if type(anchor[0]) == str:
            # Text2Image
            for nn in nns:
                labels.append(nn == idx)  # Check if same idx (a caption is associated to a single image)
        else:
            # Image2Text
            for nn in nns:
                labels.append(
                    int(dataloader.dataset.image_paths[idx] == dataloader.dataset.image_paths[nn])
                )  # Check if same image path (an image can have multiple captions)

        mavep.append(calculate_mean_average_precision(labels, distances))
        mavep25.append(calculate_mean_average_precision(labels[:26], distances[:26]))
        top_1_acc.append(calculate_top_k_accuracy(labels, k = 1))
        top_5_acc.append(calculate_top_k_accuracy(labels, k = 5))
        top_10_acc.append(calculate_top_k_accuracy(labels, k = 10))

    print(
        "Metrics: ",
        f"\n\tmAveP@50: {np.mean(mavep) * 100} %",
        f"\n\tmAveP@25: {np.mean(mavep25) * 100} %",
        f"\n\ttop_1 - precision: {np.mean(top_1_acc) * 100} %",
        f"\n\ttop_5 - precision: {np.mean(top_5_acc) * 100} %",
        f"\n\ttop_10 - precision: {np.mean(top_10_acc) * 100} %",
    )
    print(f"Finished experiment {experiment_name}.")
    print("--------------------------------------------------")

    return np.mean(mavep), np.mean(mavep25), np.mean(top_1_acc), np.mean(top_5_acc), np.mean(top_10_acc)


def main(args: argparse.Namespace):
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load data
    _, val_dataloader, _ = create_coco_dataloader(
        args.dataset_path,
        1,
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

    experiment_name = f"{args.mode}_{args.image_encoder}_{args.text_encoder}_embed{args.embedding_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with torch.no_grad():
        mavep, mavep25, top_1_acc, top_5_acc, top_10_acc = run_experiment(
            val_dataloader, model,
            args.embedding_size, args.mode, args.n_neighbors,
            experiment_name=experiment_name
        )
        with open(os.path.join(args.checkpoint, "../", "results.txt"), "w") as f:
            f.write(f"mAveP@50: {mavep}\n")
            f.write(f"mAveP@25: {mavep25}\n")
            f.write(f"top_1 - precision: {top_1_acc}\n")
            f.write(f"top_5 - precision: {top_5_acc}\n")
            f.write(f"top_10 - precision: {top_10_acc}\n")

        if args.mode == 'symmetric':  # Run both Image2Text and Text2Image
            args.mode = 'text_to_image'
            experiment_name = f"{args.mode}_{args.image_encoder}_{args.text_encoder}_embed{args.embedding_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mavep, mavep25, top_1_acc, top_5_acc, top_10_acc = run_experiment(
                val_dataloader, model,
                args.embedding_size, args.mode, args.n_neighbors,
                experiment_name=experiment_name
            )

            with open(os.path.join(args.checkpoint, "../", "results_text_to_image.txt"), "w") as f:
                f.write(f"mAveP@50: {mavep}\n")
                f.write(f"mAveP@25: {mavep25}\n")
                f.write(f"top_1 - precision: {top_1_acc}\n")
                f.write(f"top_5 - precision: {top_5_acc}\n")
                f.write(f"top_10 - precision: {top_10_acc}\n")


if __name__ == "__main__":
    matplotlib.use('Agg')
    args = __parse_args()
    main(args)