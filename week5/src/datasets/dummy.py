import string
import torch
import random


def create_dummy_data(mode: str = "image_to_text"):
        if mode == "image_to_text" or mode == "symmetric":
            anchors = torch.randn((100, 3, 224, 224))
            # generate random strings
            positives = ["".join(random.choices(string.ascii_letters, k=80))
                        for _ in range(100)]
            negatives = ["".join(random.choices(string.ascii_letters, k=80))
                        for _ in range(100)]
        else:
            anchors = ["".join(random.choices(string.ascii_letters, k=80))
                        for _ in range(100)]
            # generate random strings
            positives = torch.randn((100, 3, 224, 224))
            negatives = torch.randn((100, 3, 224, 224))
        data = list(zip(anchors, positives, negatives))
        return data


def create_dataloader(args):
    train_dataloader = torch.utils.data.DataLoader(
        create_dummy_data(args.mode),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = torch.utils.data.DataLoader(
        create_dummy_data(args.mode),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    return train_dataloader, val_dataloader, val_dataloader
