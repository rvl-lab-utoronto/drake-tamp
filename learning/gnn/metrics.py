import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import json
import matplotlib.pyplot as plt

def accuracy(logits, labels, threshold=0.5):
    return np.mean((logits >= threshold) == labels)

def precision_recall(logits, labels):
    
    inds = np.argsort(logits)[::-1]

    labels = np.array(labels)[inds]
    thresholds = np.array(logits)[inds]

    true_positive_cumsum = np.cumsum(labels == 1)
    true_negative_cumsum = np.cumsum(labels == 0)


    precision = true_positive_cumsum / np.arange(1, 1 + len(inds))
    positive_recall = true_positive_cumsum / true_positive_cumsum[-1]
    negative_recall = 1 - (true_negative_cumsum / true_negative_cumsum[-1])

    return thresholds, precision, positive_recall, negative_recall

def generate_figures(problem_stats, save_path):
    for problem_key, stats in problem_stats.items():
        make_recall_plot(stats, save_path, prefix=str(problem_key))
        make_histogram_plot(stats, save_path, prefix=str(problem_key))

def make_recall_plot(stats, save_path, prefix=''):
    idx = stats['index_of_total_recall']
    irrelevant_recall = stats['negative_recall'][idx]
    plt.plot(stats['positive_recall'], stats['negative_recall'])
    plt.xlabel('Recall on Relevant')
    plt.ylabel('% of Irrelevant Excluded')
    plt.annotate(
        f"thresh={stats['thresholds'][idx]:.2f}\nacc={stats['accuracy_at_total_recall']:.2f}\nfiltered={irrelevant_recall:.2f}",
        (stats['positive_recall'][idx], stats['negative_recall'][idx])
    )
    plt.savefig(os.path.join(save_path, prefix + 'recall.png'), dpi=300)
    plt.clf()


def make_histogram_plot(stats, save_path, prefix=''):
    logits = stats["logits"]
    labels = stats["labels"]

    logits = np.array(logits)
    labels = np.array(labels)

    assert len(logits) == len(labels), "Logits and labels should be the same size!"

    pos_logits = logits[np.where(labels == 1)]
    total_recall_log = min(pos_logits)
    neg_logits = logits[np.where(labels == 0)]
    num_neg_inc = len(np.where(neg_logits >= total_recall_log))

    # proportion of irrelevant facts included @ total recall
    prop_neg_inc = 1 - num_neg_inc/len(neg_logits) 
    # Proportion of included facts that are irrelevant
    prop_irr= num_neg_inc/(num_neg_inc + len(pos_logits))

    bins = np.linspace(0,1,100)

    fig, ax = plt.subplots()

    ax.hist(pos_logits, bins, alpha = 0.5, color = "b", label = "Relevant")
    ax.hist(neg_logits, bins, alpha = 0.5, color = "r", label = "Irrelevant")
    ax.axvline(total_recall_log, linestyle = "--", color = "k", label = f"Minimum Threshold For Total Recall")
    ax.set_xlabel("Model Relevance Score (logits)")
    ax.set_ylabel("Number")
    ax.legend(prop = {"size": 7})
    """
    ax.annotate(
        f"Proportion Irrelevant Excluded {prop_neg_inc:.3f}\nProportion of Included That Are Relevant {1 - prop_irr:.3f}",
        (0.53,0.7),
        fontsize = 7,
        xycoords='figure fraction',
        horizontalalignment='left', verticalalignment='top',
    )
    """
    plt.savefig(os.path.join(save_path, prefix + "hist.png"), dpi = 300)
    plt.close(fig)