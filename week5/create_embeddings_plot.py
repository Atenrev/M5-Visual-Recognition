import tqdm
import torch
import random
import argparse
import numpy as np


from src.embedding_viz import plot_both_embeddings
from src.models.triplet_nets import ImageToTextTripletModel, ImageToTextWithTempModel
from src.models.resnet import ResNetWithEmbedder
from src.models.bert_text_encoder import BertTextEncoder
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
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    # Dataset configuration
    parser.add_argument('--dataset_path', type=str, default='./datasets/COCO',
                        help='Path to the dataset.')
    # Model configuration
    parser.add_argument('--checkpoint', type=str, default="./checkpoints/epoch_1.pt",
                        help='Path to the checkpoint to load.')
    parser.add_argument('--model', type=str, default='image_to_text_triplet',
                        help='Model to use. Options: image_to_text_triplet, image_to_text_with_temp.')
    parser.add_argument('--image_encoder', type=str, default='resnet_18',
                        help='Model to use. Options: resnet_X, vgg.')
    parser.add_argument('--text_encoder', type=str, default='clip',
                        help='Model to use. Options: clip, bert.')
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
    for batch in tqdm.tqdm(dataloader, desc='Epoch', leave=False):
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
    # train_dataloader, val_dataloader, _ = create_coco_dataloader(
    #     args.dataset_path,
    #     args.batch_size,
    #     inference=False,
    # )
    # Create dummy data for testing
    def create_dummy_data():
        import string
        anchors = torch.randn((100, 3, 224, 224))
        # generate random strings
        positives = ["".join(random.choices(string.ascii_letters, k=80))
                     for _ in range(100)]
        negatives = ["".join(random.choices(string.ascii_letters, k=80))
                     for _ in range(100)]
        data = list(zip(anchors, positives, negatives))
        return data

    train_dataloader = torch.utils.data.DataLoader(
        create_dummy_data(),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = torch.utils.data.DataLoader(
        create_dummy_data(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Create model
    # Remember to make sure both models project to the same embedding space
    image_encoder = ResNetWithEmbedder(embed_size=args.embedding_size)

    if args.text_encoder == 'clip':
        text_encoder = CLIPTextEncoder(embed_size=args.embedding_size)
    elif args.text_encoder == 'bert':
        text_encoder = BertTextEncoder(embed_size=args.embedding_size)
    else:
        raise ValueError(f"Unknown text encoder {args.text_encoder}")

    if args.model == 'image_to_text_with_temp':
        model = ImageToTextWithTempModel(
            image_encoder,
            text_encoder,
        )
    elif args.model == 'image_to_text_triplet':
        model = ImageToTextTripletModel(
            image_encoder,
            text_encoder,
        )
    else:
        raise ValueError(f"Unknown model {args.model}")

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
