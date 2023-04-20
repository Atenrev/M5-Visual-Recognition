import tqdm
import torch
import random
import argparse
import numpy as np


from src.embedding_viz import plot_both_embeddings
from src.models.triplet_nets import ImageToTextTripletModel, SymmetricSiameseModel, TextToImageTripletModel
from src.models.resnet import ResNetWithEmbedder
from src.models.bert_text_encoder import BertTextEncoder
from src.models.clip_text_encoder import CLIPTextEncoder
from src.datasets.coco import create_dataloader as create_coco_dataloader
from src.datasets.dummy import create_dataloader as create_dummy_dataloader


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MCV-M5-Project, week 4, tasks b and c. Team 1'
    )

    # General configuration
    parser.add_argument('--output_path', type=str, default='./outputs',
                        help='Path to the output directory.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for the experiment.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    # Dataset configuration
    parser.add_argument('--dataset_path', type=str, default='./datasets/COCO',
                        help='Path to the dataset.')
    # Model configuration
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


def run_epoch(dataloader, model, device):
    model.eval()

    # images = []
    image_embeddings_list = []
    text_embeddings_list = []

    # Print loss with tqdm
    for i, batch in tqdm.tqdm(enumerate(dataloader), desc='Epoch', leave=False):
        if i == 100:
            break
        
        anchors, positives, _ = batch
        anchors = anchors.to(device)
        positives = model.tokenize(positives).to(device)

        image_embeddings = model.image_encoder(anchors)
        text_embeddings = model.text_encoder(positives.input_ids, positives.attention_mask)
            
        image_embeddings_list.append(image_embeddings.detach().cpu().numpy())
        text_embeddings_list.append(text_embeddings.detach().cpu().numpy())
        # images.append(anchors.detach().cpu().numpy())

    return np.vstack(image_embeddings_list), np.vstack(text_embeddings_list)


def main(args: argparse.Namespace):
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load data
    _, val_dataloader, _ = create_coco_dataloader(
        args.dataset_path,
        args.batch_size,
        inference=False,
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

    image_embeddings, text_embeddings = run_epoch(val_dataloader, model, device)
    plot_both_embeddings(image_embeddings, text_embeddings)


if __name__ == "__main__":
    args = __parse_args()
    main(args)
