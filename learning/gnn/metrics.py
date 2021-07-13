import os
import numpy as np
import json
import matplotlib.pyplot as plt

def accuracy(logits, labels, threshold=0.5):
    return np.mean((logits >= threshold) == labels)

def precision_recall(logits, labels):
    
    inds = np.argsort(logits)[::-1]

    labels = labels.copy()[inds]
    thresholds = logits.copy()[inds]

    true_positive_cumsum = np.cumsum(labels == 1)
    true_negative_cumsum = np.cumsum(labels == 0)


    precision = true_positive_cumsum / np.arange(1, 1 + len(inds))
    positive_recall = true_positive_cumsum / true_positive_cumsum[-1]
    negative_recall = 1 - (true_negative_cumsum / true_negative_cumsum[-1])

    return thresholds, precision, positive_recall, negative_recall

def generate_figures(stats, save_path):
    idx = stats['index_of_total_recall']
    plt.plot(stats['positive_recall'], stats['negative_recall'])
    plt.xlabel('Recall on Relevant')
    plt.ylabel('% of Irrelevant Excluded')
    plt.annotate(
        f"thresh={stats['thresholds'][idx]:.2f}\nacc={stats['accuracy_at_total_recall']:.2f}",
        (stats['positive_recall'][idx], stats['negative_recall'][idx])
    )
    plt.savefig(os.path.join(save_path, 'recall.png'), dpi=300)