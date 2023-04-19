import tqdm
import torch
import numpy as np

from src.metrics import LossMetric
from src.trackers.tracker import Stage, ExperimentTracker
from src.models.triplet_nets import TripletModel


class Runner:
    def __init__(self, model, loss_fn, optimizer, device, train=True, tracker: ExperimentTracker = None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.train = train
        self.tracker = tracker

        self.run_count = 0
        self.reset()

    def run_epoch(self, dataloader):
        if self.train:
            self.model.train()
        else:
            self.model.eval()

        # Print loss with tqdm
        for batch in (pbar := tqdm.tqdm(dataloader, desc='Epoch', leave=False)):
            batch_size = len(batch[0])
            anchors, positives, negatives = batch
            anchors = anchors.to(self.device)
            positives = self.model.tokenize(positives).to(self.device)
            negatives = self.model.tokenize(negatives).to(self.device)

            # Forward
            if isinstance(self.model, TripletModel):
                anchor_embeddings, positive_embeddings, negative_embeddings = self.model(
                    anchors, positives.input_ids, positives.attention_mask,
                    negatives.input_ids, negatives.attention_mask
                )
                loss = self.loss_fn(anchor_embeddings,
                            positive_embeddings, negative_embeddings)
            else:
                logits, image_embeddings, text_embeddings = self.model(
                    anchors, positives.input_ids, positives.attention_mask)
                loss = self.loss_fn(logits)

            # Backward
            if self.train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Metrics
            self.metrics['loss'].update(loss.item())

            # Calculate accuracy
            if isinstance(self.model, TripletModel):
                positive_distances = torch.sum(
                    (anchor_embeddings - positive_embeddings) ** 2, dim=1)
                negative_distances = torch.sum(
                    (anchor_embeddings - negative_embeddings) ** 2, dim=1)
                accuracy = torch.mean(
                    (positive_distances < negative_distances).float()).item()
                self.metrics['accuracy'].update(accuracy)
            else:
                # Compute pairwise distances between all images and texts
                distances = torch.cdist(image_embeddings.detach().cpu(), text_embeddings.detach().cpu()).numpy()
                # Find the index of the smallest distance for each image embedding and text embedding
                image_matches = np.argmin(distances, axis=1)
                text_matches = np.argmin(distances, axis=0)
                # Compute accuracy
                correct_matches = 0
                for i in range(batch_size):
                    if text_matches[image_matches[i]] == i:
                        correct_matches += 1
                accuracy = correct_matches / batch_size
                self.metrics['accuracy'].update(accuracy)

            # Log metrics
            if self.tracker is not None:
                for i, metric in enumerate(self.metrics.values()):
                    self.tracker.add_batch_metric(
                        metric.name, metric.values[-1], self.run_count, commit=i == len(self.metrics) - 1)

            self.run_count += 1
            pbar.set_postfix(
                {metric_name: metric_value.values[-1] for metric_name, metric_value in self.metrics.items()})
    
    def reset(self):
        self.metrics = {
            'loss': LossMetric(),
            'accuracy': LossMetric()
        }


def train(train_dataloader, val_dataloader, model, loss_fn, optimizer, device, epochs, tracker: ExperimentTracker = None, current_epoch: int = 0):
    best_val_value = np.inf
    train_runner = Runner(model, loss_fn, optimizer, device, train=True, tracker=tracker)
    val_runner = Runner(model, loss_fn, optimizer, device, train=False, tracker=tracker)

    for epoch in range(current_epoch, epochs):
        # Train
        tracker.set_stage(Stage.TRAIN)
        train_runner.run_epoch(train_dataloader)

        for metric_name, metric_value in train_runner.metrics.items():
            tracker.add_epoch_metric(
                metric_name, metric_value.average, epoch)

        # Validate
        tracker.set_stage(Stage.VAL)
        with torch.no_grad():
            val_runner.run_epoch(val_dataloader)

        for metric_name, metric_value in val_runner.metrics.items():
            tracker.add_epoch_metric(
                metric_name, metric_value.average, epoch)

        # Save checkpoint
        if val_runner.metrics['loss'].average < best_val_value:
            best_val_value = val_runner.metrics['loss'].average
            tracker.save_checkpoint(
                epoch,
                model,
                optimizer
            )

        summary = f"Epoch {epoch}/{epochs} - Train loss: {train_runner.metrics['loss'].average:.4f} - Val loss: {val_runner.metrics['loss'].average:.4f}"
        print("\n", summary, "\n")

        train_runner.reset()
        val_runner.reset()
        tracker.flush()

    tracker.finish()