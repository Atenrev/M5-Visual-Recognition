import tqdm
import torch
import numpy as np

from src.metrics import LossMetric, BasicAccuracyMetric
from src.trackers.tracker import Stage, ExperimentTracker
from src.models.triplet_nets import TripletModel, SymmetricSiameseModel, ImageToTextTripletModel, TextToImageTripletModel


class Runner:
    def __init__(self, model: TripletModel, optimizer, device, train=True, tracker: ExperimentTracker = None):
        self.model: TripletModel = model
        self.optimizer = optimizer
        self.device = device
        self.train = train
        self.tracker = tracker

        self.run_count = 0
        self.reset()

    def calculate_accuracy(self, anchor_embeddings, positive_embeddings, batch_size, negative_embeddings=None):
        # if negative_embeddings is not None:
        #     positive_distances = torch.sum(
        #         (anchor_embeddings - positive_embeddings) ** 2, dim=1)
        #     negative_distances = torch.sum(
        #         (anchor_embeddings - negative_embeddings) ** 2, dim=1)
        #     accuracy = torch.mean(
        #         (positive_distances < negative_distances).float()).item()
        #     self.metrics['accuracy'].update(accuracy)
        # else:
        # Compute pairwise distances
        anchor_embeddings = anchor_embeddings.detach().cpu().numpy()
        positive_embeddings = positive_embeddings.detach().cpu().numpy()
        distances = np.dot(anchor_embeddings, positive_embeddings.T)
        distances = 1 - distances  # convert dot product to cosine distance

        # Find most similar embeddings
        anchor_to_positive = np.argmin(distances, axis=1)
        positive_to_anchor = np.argmin(distances, axis=0)

        # Compute accuracy score
        correct_matches = np.sum(np.arange(batch_size) == anchor_to_positive[positive_to_anchor])
        accuracy = correct_matches / batch_size
        self.metrics['accuracy'].update(accuracy)

    def run_epoch(self, dataloader):
        image_embeddings_list = []
        text_embeddings_list = []

        if self.train:
            self.model.train()
        else:
            self.model.eval()

        for batch in (pbar := tqdm.tqdm(dataloader, desc='Epoch', leave=False)):
            batch_size = len(batch[0])
            anchors, positives, negatives = batch

            if type(anchors[0]) == str:
                anchors = self.model.tokenize(anchors).to(self.device)
                positives = positives.to(self.device)
                negatives = negatives.to(self.device)
            else:
                anchors = anchors.to(self.device)
                positives = self.model.tokenize(positives).to(self.device)
                negatives = self.model.tokenize(negatives).to(self.device)

            # Forward
            if isinstance(self.model, SymmetricSiameseModel):
                model_output = self.model.forward(anchors, positives.input_ids, positives.attention_mask)
                if not self.train:
                    image_embeddings_list.append(model_output.anchor_embedding.detach().cpu().numpy())
                    text_embeddings_list.append(model_output.positive_embedding.detach().cpu().numpy())
            elif isinstance(self.model, ImageToTextTripletModel):
                model_output = self.model.forward(anchors, positives.input_ids, positives.attention_mask, negatives.input_ids, negatives.attention_mask)
                if not self.train:
                    image_embeddings_list.append(model_output.anchor_embedding.detach().cpu().numpy())
                    text_embeddings_list.append(model_output.positive_embedding.detach().cpu().numpy())
            elif isinstance(self.model, TextToImageTripletModel):
                model_output = self.model.forward(anchors.input_ids, anchors.attention_mask, positives, negatives)
                if not self.train:
                    image_embeddings_list.append(model_output.positive_embedding.detach().cpu().numpy())
                    text_embeddings_list.append(model_output.anchor_embedding.detach().cpu().numpy())
            else:
                raise Exception("Model type not supported")

            loss = model_output.loss
            self.calculate_accuracy(model_output.anchor_embedding, model_output.positive_embedding, batch_size)

            # Backward
            if self.train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Metrics
            self.metrics['loss'].update(loss.item())                

            if self.tracker is not None:
                for i, metric in enumerate(self.metrics.values()):
                    self.tracker.add_batch_metric(
                        metric.name, metric.values[-1], self.run_count, commit=i == len(self.metrics) - 1)

            self.run_count += 1
            pbar.set_postfix(
                {metric_name: metric_value.values[-1] for metric_name, metric_value in self.metrics.items()})
        
        if not self.train:
            image_embeddings_list = np.vstack(image_embeddings_list)
            text_embeddings_list = np.vstack(text_embeddings_list)
            return image_embeddings_list, text_embeddings_list
        
        return None, None
    
    def reset(self):
        self.metrics = {
            'loss': LossMetric(),
            'accuracy': BasicAccuracyMetric()
        }


def train(train_dataloader, val_dataloader, model, optimizer, device, epochs, tracker: ExperimentTracker = None, current_epoch: int = 0):
    best_val_value = np.inf
    train_runner = Runner(model, optimizer, device, train=True, tracker=tracker)
    val_runner = Runner(model, optimizer, device, train=False, tracker=tracker)

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
            image_embeddings, text_embeddings = val_runner.run_epoch(val_dataloader)

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

        # Plot embeddings
        if image_embeddings is not None and text_embeddings is not None:
            tracker.plot_embeddings(image_embeddings, text_embeddings, epoch)

        summary = f"Epoch {epoch}/{epochs} - Train loss: {train_runner.metrics['loss'].average:.4f} - Val loss: {val_runner.metrics['loss'].average:.4f} - Train accuracy: {train_runner.metrics['accuracy'].average:.4f} - Val accuracy: {val_runner.metrics['accuracy'].average:.4f}"
        print("\n", summary, "\n")

        train_runner.reset()
        val_runner.reset()
        tracker.flush()

    tracker.finish()