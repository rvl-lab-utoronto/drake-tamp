#%%

from learning.gnn.models import HyperClassifier
import torch
from learning.visualization import plot_simple_graph
from learning.gnn.data import construct_object_hypergraph, parse_hyper_labels
from learning.gnn.train import train_model_graphnetwork, StratifiedRandomSampler

import pickle
filepath = "/home/agrobenj/drake-tamp/learning/data/labeled/2021-07-11-13:45:03.864.pkl"
with open(filepath, "rb") as s:
    data = pickle.load(s)


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

dataset, model_info  = parse_hyper_labels(filepath)
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
    epochs=100
)
