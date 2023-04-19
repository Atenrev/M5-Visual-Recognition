import os
import pathlib
import numpy as np
import torch
import wandb

from pathlib import Path
from typing import List, Tuple

from src.trackers.tracker import Stage, ExperimentTracker


def create_experiment_dir(root: str, experiment_name: str,
                          parents: bool = True) -> Tuple[str, str]:
    root_path = pathlib.Path(root).resolve()
    child = root_path / experiment_name
    child.mkdir(parents=parents)
    models_path = child / "models"
    models_path.mkdir()
    return child.as_posix(), models_path.as_posix()


class WandbTracker(ExperimentTracker):
    """
    Creates a tracker that implements the ExperimentTracker protocol and logs to wandb.
    """

    def __init__(self, log_path: str, experiment_name: str, project_name: str, config: dict, tags: List[str] = None):
        self.stage = Stage.TRAIN
        self.run = wandb.init(project=project_name,
                              name=experiment_name, 
                              entity="m5-group1",  
                              config=config,
                              tags=tags)
        log_dir, self.models_dir = create_experiment_dir(
            root=log_path, experiment_name=experiment_name)
        self._validate_log_dir(log_dir, create=True)

        wandb.define_metric("batch_step")
        wandb.define_metric("epoch")

        for metric in ["loss"] + config["metrics"]:
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
        
    def flush(self):
        pass

    def finish(self):
        wandb.finish()
