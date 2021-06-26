#!/usr/bin/env python3
import os
from learning import visualization
file_path, _ = os.path.split(os.path.realpath(__file__))
file_path = "/".join(file_path.split("/")[:-1])

visualization.stats_to_graph(f"{file_path}/data/stats.json", "test_graph.html")