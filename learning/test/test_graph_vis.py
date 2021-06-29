#!/usr/bin/env python3
import os
from learning import visualization
import json
from learning import oracle
file_path, _ = os.path.split(os.path.realpath(__file__))
file_path = "/".join(file_path.split("/")[:-1])

visualization.stats_to_graph(f"{file_path}/data/stats.json", "test_graph.html", verbose = True)

"""
stream = open(f"{file_path}/data/stats.json")
data = json.load(stream)
stream.close()
last_preimage = data["last_preimage"]
atom_map = oracle.item_to_dict(data["atom_map"])

graph = visualization.DirectedGraph(atom_map)
for fact in last_preimage:
    fact = tuple(fact)
    graph.add_node(fact)
print(graph)
print()
"""