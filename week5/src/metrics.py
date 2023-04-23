import numpy as np
# import evaluate

from typing import Any, List
from dataclasses import dataclass, field


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


class BasicAccuracyMetric(Metric):
    name: str = "accuracy"
    inpyt_type: str = "float"
    pass


from sklearn.metrics import average_precision_score, recall_score, precision_score
import cv2
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt


def calculate_mean_average_precision(prediction, gt): 
    return average_precision_score(prediction, gt)


def calculate_recall(prediction, gt):
    return recall_score(prediction, gt)


def calculate_precision(prediction, gt): 
    return precision_score(prediction, gt)


def calculate_top_k_accuracy(prediction, k = 5):
    return sum(prediction[:k]) / k# > 0


def plot_retrieved_images(query, retrieved, true_positives = None, green_p = .05, shape = 512, out = 'tmp.png'):
    # TP: [1, 0, 0, 1 ...] so we can grayscale FP
    query = cv2.resize(query, (shape, shape))
    until = int(shape * green_p)
    query[:until, :] = np.array([0, 0, 255])
    query[shape - until:, :] = np.array([0, 0, 255])

    query[:, :until] = np.array([0, 0, 255])
    query[:, shape - until:] = np.array([0, 0, 255])
    
    final_image = [query]

    for n, image in enumerate(retrieved):
        resized = cv2.resize(image, (shape, shape))

        if not (true_positives is None):
            color = np.array([255, 0, 0]) if not true_positives[n] else np.array([0, 255, 0])

            if not true_positives[n]: 
                resized = cv2.cvtColor(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2BGR)

            resized[:until, :] = color
            resized[shape - until:, :] = color

            resized[:, :until] = color
            resized[:, shape - until:] = color
        
        final_image.append(resized)

    stacked = np.hstack(final_image)
    cv2.imwrite(out, cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))


def plot_prec_rec_curve_multiclass(gt, pred, labels, output_path='prec_rec_curve.png'):
    precision = dict()
    recall = dict()

    for i in range(8):
        ap = average_precision_score(gt[:, i], pred[:, i])
        precision[i], recall[i], _ = precision_recall_curve(gt[:, i],
                                                            pred[:, i])
        plt.plot(recall[i], precision[i], lw=2, label=f'{labels[i]} - {ap:.3f}')
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.title(f"Precision vs Recall curve")
    plt.show()
    plt.savefig(output_path)