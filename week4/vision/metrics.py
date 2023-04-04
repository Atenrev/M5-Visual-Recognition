import torch
from sklearn.metrics import average_precision_score, recall_score, precision_score, top_k_accuracy_score
import cv2
import numpy as np
# Just wrappers if I want to set up a certain format


def calculate_mean_average_precision(prediction, gt): 
    return average_precision_score(prediction, gt)

def calculate_recall(prediction, gt):
    return recall_score(prediction, gt)

def calculate_precision(prediction, gt): 
    return precision_score(prediction, gt)

def calculate_top_k_accuracy(prediction, target, k = 5):
    return top_k_accuracy_score(prediction, target, k = k)

def plot_retrieved_images(query, retrieved, true_positives = None, green_p = .05, shape = 512, out = 'tmp.png'):
    # TP: [1, 0, 0, 1 ...] so we can grayscale FP
    query = cv2.resize(query, (shape, shape))
    until = shape * green_p
    query[:until, :until] = np.array([0, 255, 0])
    query[shape - until:, shape - until:] = np.array([0, 255, 0])
    
    final_image = [query]
    for n, image in enumerate(retrieved):
        resized = cv2.resize(image, (shape, shape))
        if not (true_positives is None):

            if not true_positives[n]: resized = cv2.cvtColor(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        
        final_image.append(resized)
    
    stacked = np.hstack(final_image)
    cv2.imwrite(out, stacked)

    return 1