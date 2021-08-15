import json
import numpy as np
from pyvis.network import Network
from . import oracle as ora


class DirectedGraph:
    def __init__(self, atom_map, verbose = False):
        """
        atom_map is a dictionary of the form
        {
            (<object_tuple>): [<list_of_parent_facts>],
            ('item', 'v7'): [],
            ('handpose', 'v1', '#x0'):[('item', 'v1')],
        }
        """
        self.adjaceny_list = {}
        self.ans_cache = {}
        self.atom_map = atom_map
        self.max_level = 0
        self.max_children = 0
        self.verbose = verbose

    def add_node(self, fact):
        """
        Adds a node and all of its parents
        """
        if fact not in self.atom_map:
            if self.verbose:
                print(f"Failed to add {fact}, not in atom_map")
            return
        if fact in self.adjaceny_list:
            return
        self.adjaceny_list[fact] = []
        parents = self.atom_map[fact]
        for p in parents:
            if p not in self.adjaceny_list:
                self.add_node(p)
            self.adjaceny_list[p].append(fact)
            self.max_children = max(self.num_children(p), self.max_children)

    def num_children(self, fact):
        assert fact in self.adjaceny_list, "Fact not in this graph"
        return len(self.adjaceny_list[fact])

    def __str__(self):
        res = ""
        for fact in self.adjaceny_list:
            res += f"{fact}: "
            if len(self.adjaceny_list[fact]) == 0:
                res += "\n"
            for i in range(len(self.adjaceny_list[fact])):
                child = self.adjaceny_list[fact][i]
                if i == len(self.adjaceny_list[fact]) - 1:
                    res += f"{child}"
                    res += "\n"
                else:
                    res += f"{child}, "
        return res

    def num_ans(self, fact):
        if fact not in self.ans_cache:
            self.ans_cache[fact] = ora.ancestors(fact, self.atom_map)
        return len(self.ans_cache[fact])

    def fact_to_color(self, fact):
        # TODO(agro): do this better
        max_children = self.max_children
        if max_children == 0:
            max_children = 1
        r = 200*(max_children - self.num_children(fact))/max_children
        g = 0
        b = 100*(max_children - self.num_children(fact))/max_children
        if (self.num_children(fact) == 0) and (self.num_ans(fact) == 0):
            r = 0
            b = 200
        color = f"rgba({r},{g},{b},1)"
        return color

    def make_pyvis_net(self, net, add_loners=True, node_repulsion=True):
        added = []
        level_x = {}
        for fact in self.adjaceny_list:
            if (len(self.adjaceny_list[fact]) == 0 and self.num_ans(fact) == 0) and not add_loners:
                if self.verbose:
                    print(f"Not adding {fact} to network visualization: No children")
                continue
            if fact not in added:
                # print(str(fact))
                level = self.num_ans(fact)
                net.add_node(
                    str(fact),
                    label=str(fact),
                    level=level,
                    physics=node_repulsion,
                    shape="box",
                    color=self.fact_to_color(fact),
                )
                added.append(fact)
            for child in self.adjaceny_list[fact]:
                if child not in added:
                    added.append(fact)
                    level = self.num_ans(child)
                    net.add_node(
                        str(child),
                        label=str(child),
                        level=level,
                        physics=node_repulsion,
                        shape="box",
                        color = self.fact_to_color(child),
                    )
                net.add_edge(str(fact), str(child), physics=False)


def ancestors(fact, atom_map):
    """
    Given a fact, return a set of
    that fact's ancestors
    """
    parents = atom_map[fact]
    res = set(parents)
    for parent in parents:
        res |= ancestors(parent, atom_map)
    return set(res)


def tuplize_atom_map(atom_map):
    res = {}
    for item in atom_map:
        t_item = tuple(item[0])
        res[t_item] = []
        for par in item[1]:
            res[t_item].append(tuple(par))
    return res


def tuplize_preimage(preimage):
    res = []
    for f in preimage:
        res.append(tuple(f))
    return res

def visualize_atom_map(atom_map, save_path):
    net = Network(
        directed=True, width="100%", height="100%", layout="hierarchy", font_color="white"
    )
    net.repulsion(
        node_distance=300, spring_strength=0, spring_length=400, central_gravity=0
    )
    graph = DirectedGraph(atom_map, verbose = False)
    for fact in atom_map:
        graph.add_node(fact)
    graph.make_pyvis_net(net)
    net.set_options(
        """
        var options = {
            "layout": {
                "hierarchical": {
                "enabled": true,
                "nodeSpacing": 250,
                "treeSpacing": 225
                }
            }
        }
        """ 
    )
    # net.show(name = "graph.html")
    #net.show_buttons(filter_ = ["layout"])
    net.save_graph(save_path)


def stats_to_graph(path_to_stats, save_path, verbose = False):
    """
    Given the path to the `stats.json` file for
    a particular trial, create the graph for that
    run and save it at `save_path`
    """
    stream = open(path_to_stats)
    data = json.load(stream)
    stream.close()
    net = Network(directed=True, width="100%", height="100%", layout="hierarchy", font_color="white")
    net.repulsion(
        node_distance=300, spring_strength=0, spring_length=400, central_gravity=0
    )
    last_preimage = tuplize_preimage(data["last_preimage"])
    atom_map = tuplize_atom_map(data["atom_map"])
    graph = DirectedGraph(atom_map, verbose = verbose)
    for fact in last_preimage:
        graph.add_node(fact)
    graph.make_pyvis_net(net)
    net.set_options(
        """
        var options = {
            "layout": {
                "hierarchical": {
                "enabled": true,
                "nodeSpacing": 250,
                "treeSpacing": 225
                }
            }
        }
        """ 
    )
    # net.show(name = "graph.html")
    #net.show_buttons(filter_ = ["layout"])
    net.save_graph(save_path)


def plot_simple_graph(nodes, edges, save_path, edge_names = None, levels = None):
    """
    Make a simple graph

    nodes: list of node names
    edges: list of tuples of node indicies the edges are connecting 
    """
    if levels is not None:
        net = Network(width="100%", height="100%", layout = "hierarchy")
        net.set_options(
            """
            var options = {
                "layout": {
                    "hierarchical": {
                    "enabled": true,
                    "nodeSpacing": 250,
                    "treeSpacing": 225
                    }
                }
            }
            """ 
        )
        assert len(levels) == len(nodes)
    else:
        net = Network(width="100%", height="100%")
    if edge_names is not None:
        assert len(edge_names) == len(edges)
    for i, node in enumerate(nodes):
        if levels is None:
            net.add_node(node)
        else:
            net.add_node(
                node, level = levels[i], label = node, shape = "box"
            )
    if edge_names is None:
        for edge in edges:
            net.add_edge(nodes[edge[0]], nodes[edge[1]])
    else:
        for edge, title in zip(edges, edge_names):
            net.add_edge(nodes[edge[0]], nodes[edge[1]], title = title)

    net.save_graph(save_path)

def visualize_scene_graph(
    nodes,
    node_attr,
    edges,
    edge_attr,
    save_path
):
    net = Network(width="100%", height="100%")

    xs = []
    ys = []
    for attr in node_attr:
        xs.append(attr["worldpose"].translation()[0])
        ys.append(attr["worldpose"].translation()[1])

    x_shift = -min(xs)
    y_shift = -min(ys)
    scale = 1000

    for attr in node_attr:
        net.add_node(
            attr["name"],
            lbel = attr["name"],
            shape = "box",
            x = scale*(attr["worldpose"].translation()[0] + x_shift),
            y = scale*(attr["worldpose"].translation()[1] + y_shift),
            physics = False
        )
    
    net.save_graph(save_path)
