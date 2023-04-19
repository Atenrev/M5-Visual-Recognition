import numpy as np
# import evaluate

from typing import Any, List
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score


@dataclass
class Metric:
    """
    Taken from https://github.com/ArjanCodes/2021-data-science-refactor/blob/main/after/ds/metrics.py
    """
    name: str = field(init=False)
    inpyt_type: str = field(init=False)
    values: List[float] = field(default_factory=list)
    running_total: float = 0.0
    num_updates: float = 0.0
    average: float = 0.0

    def update(self, value: float, batch_size: int = 1) -> None:
        self.values.append(value)
        self.running_total += value * batch_size
        self.num_updates += batch_size
        self.average = self.running_total / self.num_updates

    def calculate_and_update(self, targets: Any, predictions: Any) -> float:
        raise NotImplementedError


class LossMetric(Metric):
    name: str = "loss"
    inpyt_type: str = "float"
    pass