#%%

from learning.gnn.models import HyperClassifier
import torch
from learning.visualization import plot_simple_graph, visualize_scene_graph
from learning.gnn.data import (
    construct_object_hypergraph, get_ancestor_objects, parse_hyper_labels, construct_scene_graph
)
from learning.gnn.train import train_model_graphnetwork, StratifiedRandomSampler

import pickle
filepath = "/home/agrobenj/drake-tamp/learning/data/labeled/2021-07-12-23:26:54.038.pkl"
with open(filepath, "rb") as s:
    data = pickle.load(s)

label = data["labels"][162]
print(f"Output object: {label.result.output_objects}")
print(f"Ancestors: {get_ancestor_objects(label)}")
print(label.result.name)
print(label.result.certified)
print(len(data["labels"]))
nodes, node_attr, edges, edge_attr = construct_object_hypergraph(
    label,
    data["problem_info"],
    data["model_info"],
    reduced = True
)
#visualize_scene_graph(nodes,node_attr, edges, edge_attr,"tmp.html")
plot_simple_graph(
    nodes,
    edges,
    "tmp_red.html",
    edge_names = [e["predicate"] for e in edge_attr],
    levels = [n["level"] for n in node_attr]
)
print()
#for label in data["labels"]:
    #print(label[:-2])

#%%

"""
nodes, node_attr, edges, edge_attr = construct_object_hypergraph(
    data["labels"][300], data["problem_info"], data["model_info"]
)

plot_simple_graph(
    nodes,
    edges,
    "tmp.html",
    edge_names = [e["predicate"] for e in edge_attr],
    levels = [n["level"] for n in node_attr]
)
print()
"""

#%%

"""
#dataset, model_info  = parse_hyper_labels("/home/agrobenj/drake-tamp/learning/data/labeled/2021-07-11-13:45:03.864.pkl")
#dataset += parse_hyper_labels("/home/agrobenj/drake-tamp/learning/data/labeled/2021-07-11-14:29:08.719.pkl")[0]
#dataset += parse_hyper_labels("/home/agrobenj/drake-tamp/learning/data/labeled/2021-07-11-14:29:18.821.pkl")[0]
dataset, model_info = parse_hyper_labels("/home/agrobenj/drake-tamp/learning/data/labeled/2021-07-11-14:29:41.795.pkl")
model = HyperClassifier(
    node_feature_size=model_info.node_feature_size,
    edge_feature_size=model_info.edge_feature_size,
    stream_domains=model_info.stream_domains[1:],
    stream_num_inputs=model_info.stream_num_inputs[1:],
)
neg = [d for d in dataset if d.y[0] == 0]
pos = [d for d in dataset if d.y[0] == 1]
data = dict(
    train=StratifiedRandomSampler(pos, neg, prop=0.5),
    val=StratifiedRandomSampler(pos, neg, prop=0.5),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=.8*torch.ones([1]))
train_model_graphnetwork(
    model,
    data,
    criterion=criterion,
    optimizer=optimizer,
    save_every=10,
    epochs=1000
)

import numpy as np

logits = []
labels = []
model.eval()
for d in dataset:
    logit = torch.sigmoid(model(d)).detach().numpy()[0]
    logits.append(logit)
    labels.append(d.y.detach().numpy().item())
    # print(d.certified, logit, d.y)
logits = np.array(logits)
labels = np.array(labels)
inds = np.argsort(logits)
labels = labels[inds]
scores = logits[inds]
num_irrelevant_excluded = best_threshold_index = list(labels).index(1)
best_threshold = scores[best_threshold_index]
num_irrelevant_included = np.sum(labels[best_threshold_index:] == 0)
#for ind in inds[best_threshold_index:]:
    #print(dataset[ind].certified, dataset[ind].y)
print(num_irrelevant_excluded/(num_irrelevant_excluded + num_irrelevant_included))
"""