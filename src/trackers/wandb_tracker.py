import os
import numpy as np
import torch
import wandb

from pathlib import Path
from typing import List, Tuple

from src.trackers.tracker import Stage, ExperimentTracker
from src.common.utils import create_experiment_dir
from src.common.registry import Registry


class WandbTracker(ExperimentTracker):
    """
    Creates a tracker that implements the ExperimentTracker protocol and logs to wandb.
    """

    def __init__(self, log_path: str, experiment_name: str, project_name: str, tags: List[str] = None):
        self.stage = Stage.TRAIN
        configs = {
            "trainer": Registry.get("trainer_config"),
            "model": Registry.get("model_config"),
            "dataset": Registry.get("dataset_config")
        }
        self.run = wandb.init(project=project_name,
                              name=experiment_name, 
                              config=configs,
                              tags=tags)
        log_dir, self.models_dir = create_experiment_dir(
            root=log_path, experiment_name=experiment_name)
        self._validate_log_dir(log_dir, create=True)

        wandb.define_metric("batch_step")
        wandb.define_metric("epoch")

        for metric in ["loss"] + Registry.get("model_config").metrics:
            for stage in Stage:
                wandb.define_metric(f"{stage.name}/batch_{metric}", step_metric='batch_step')
                wandb.define_metric(f"{stage.name}/epoch_{metric}", step_metric='epoch')
        
    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True):
        log_path = Path(log_dir).resolve()
        if log_path.exists():
            return
        elif not log_path.exists() and create:
            log_path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

    def set_stage(self, stage: Stage):
        self.stage = stage

    def save_checkpoint(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        print("\nNEW BEST MODEL, saving checkpoint.")

        save_path = os.path.join(self.models_dir, f"epoch_{epoch + 1}.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        wandb.run.summary["best_epoch"] = epoch + 1

    def add_batch_metric(self, name: str, value: float, step: int, commit: bool = True):
        wandb.log({f"{self.stage.name}/batch_{name}": value, "batch_step": step}, commit=commit)

    def add_epoch_metric(self, name: str, value: float, step: int):
        wandb.log({f"{self.stage.name}/epoch_{name}": value, "epoch": step})

    @staticmethod
    def collapse_batches(
        y_true: List[np.ndarray], y_pred: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return np.concatenate(y_true), np.concatenate(y_pred)

    def add_epoch_confusion_matrix(
        self, y_true: List[np.ndarray], y_pred: List[np.ndarray], step: int
    ):
        y_true, y_pred = self.collapse_batches(y_true, y_pred)
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=y_true, preds=y_pred, )}) # class_names=class_names
        
    def flush(self):
        pass

    def finish(self):
        wandb.finish()
