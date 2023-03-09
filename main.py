import argparse
import torch
import numpy as np
import importlib
import logging

from torchinfo import summary
from torchviz import make_dot
# from transformers import AutoTokenizer

from src.common.registry import Registry
from src.common.configuration import get_dataset_configuration, get_model_configuration, get_trainer_configuration
from src.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="xception",
                        help='Model to run')
    parser.add_argument('--dataset_config', type=str, default="mit_split_base",
                        help='Dataset config to use')
    parser.add_argument('--trainer_config', type=str, default="default",
                        help='Trainer params to use')
    parser.add_argument('--dataset_dir', type=str, default="datasets/MIT_split/",
                        help='Dataset directory path')
    parser.add_argument('--mode', type=str, default="summary",
                        help='Execution mode ("training" or "eval")')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Seed to use')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    torch.manual_seed(0)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"SELECTED DEVICE: {device}")

    # Configuration loading
    model_config = get_model_configuration(args.model)
    Registry.register("model_config", model_config)
    dataset_config = get_dataset_configuration(args.dataset_config)
    Registry.register("dataset_config", dataset_config)

    logging.info(f"SELECTED MODEL: {model_config.classname}")
    logging.info(f"SELECTED DATASET: {dataset_config.name}")

    # Dataset preprocessing
    tokenizer = None
    if model_config.tokenizer:
        # tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer)
        raise NotImplementedError("Tokenizers are not implemented yet.")

    dataset_kwargs = {}
    if tokenizer:
        dataset_kwargs["tokenizer"] = tokenizer

    # Model loading
    ModelClass = getattr(importlib.import_module(
        f"src.models.{args.model}"), model_config.classname)
    model = ModelClass(model_config, device).to(device)

    if tokenizer:
        model.tokenizer = tokenizer

    # Load model checkpoint
    checkpoint = None

    if args.load_checkpoint is not None:
        logging.info("Loading checkpoint.")

        try:
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
        except Exception as e:
            logging.error("The checkpoint could not be loaded.")
            print(e)
            return
            
        model.load_checkpoint(checkpoint["model_state_dict"])

    # if torch.cuda.device_count() > 1:
        # model = torch.nn.DataParallel(model)

    # Trainer specific configuration loading
    trainer_config = get_trainer_configuration(args.trainer_config)
    trainer_config.batch_size = args.batch_size
    Registry.register("trainer_config", trainer_config)

    # DataLoaders
    create_dataloader = getattr(importlib.import_module(
        f"src.datasets.{dataset_config.name}"), "create_dataloader")
    train_dataloader, val_dataloader, test_dataloader = create_dataloader(
        args.batch_size,
        args.dataset_dir,
        device,
        dataset_config,
        dataset_kwargs=dataset_kwargs
    )

    trainer = Trainer(model, train_dataloader, val_dataloader,
                        test_dataloader, device, trainer_config, checkpoint)

    if args.mode == "train":
        trainer.train(trainer_config.epochs)
    elif args.mode == "eval":
        assert checkpoint is not None, "ERROR: No checkpoint provided."
        trainer.eval()
    elif args.mode == "summary":
        batch = next(iter(train_dataloader))
        batch = batch["data"] if "data" in batch else batch
        preds = model(**batch)["logits"]
        graph = make_dot(preds, params=dict(list(model.named_parameters())))
        graph.render("model_architecture", format="png")
        summary(model, input_size=batch["image"].shape)
    else:
        raise ValueError(
            f"Unknown mode: {args.mode}. Please select one of the following: train, eval, inference")


if __name__ == "__main__":
    args = parse_args()
    main(args)
