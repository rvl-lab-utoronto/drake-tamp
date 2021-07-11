#%%

import torch
from learning.visualization import plot_simple_graph

import pickle
with open("/home/agrobenj/drake-tamp/learning/data/labeled/2021-07-10-23:43:28.517.pkl", "rb") as s:
    data = pickle.load(s)

print(data.keys())

#for label in data["labels"]:
    #print(label[:-2])

#%%

from learning.gnn.data import construct_object_hypergraph
nodes, node_attr, edges, edge_attr = construct_object_hypergraph(
    data["model_info"], data["problem_info"], data["labels"][300]
)

plot_simple_graph(
    nodes,
    edges,
    "tmp.html",
    edge_names = [e["predicate"] for e in edge_attr],
    levels = [n["level"] for n in node_attr]
)
print()