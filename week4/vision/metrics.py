import torch
from sklearn.metrics import average_precision_score, recall_score, precision_score, top_k_accuracy_score
# Just wrappers if I want to set up a certain format


def calculate_mean_average_precision(prediction, gt): 
    return average_precision_score(prediction, gt)

def calculate_recall(prediction, gt):
    return recall_score(prediction, gt)

def calculate_precision(prediction, gt): 
    return precision_score(prediction, gt)

def calculate_top_k_accuracy(prediction, target, k = 5):
    return top_k_accuracy_score(prediction, target, k = k)

def plot_retrieved_images(query, retrieved, gt = None): pass