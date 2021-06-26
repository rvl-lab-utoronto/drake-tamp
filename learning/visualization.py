import json
from pyvis.network import Network
from . import oracle


def stats_to_graph(path_to_stats, save_path):
    """
    Given the path to the `stats.json` file for
    a particular trial, create the graph for that
    run and save it at `save_path`
    """
    stream = open(path_to_stats)
    data = json.load(stream)
    stream.close()
    net = Network(
        directed = True,
        width = 1500,
        height = 1500,
        #layout = "hierarchy"
    )
    net.repulsion(node_distance = 150, spring_strength = 0)
    last_preimage = data["last_preimage"]
    atom_map = oracle.item_to_dict(data["atom_map"])
    level = 0
    for fact in last_preimage:
        level +=1
        fact = tuple(fact)
        net.add_node(str(fact), label = str(fact), level = level)
        ancestors = set()
        if fact in atom_map:
            ancestors = oracle.ancestors(fact, atom_map)
        else:
            print(fact)
        for ancestor in ancestors:
            net.add_node(str(ancestor), label = str(ancestor), level = level)
            net.add_edge(str(ancestor), str(fact))

    #net.show_buttons()
    net.save_graph(save_path)
