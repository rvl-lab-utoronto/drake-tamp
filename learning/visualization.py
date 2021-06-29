import os
import json
from pyvis.network import Network
from . import oracle


class DirectedGraph:
    def __init__(self, atom_map):
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

    def add_node(self, fact):
        """
        Adds a node and all of its parents
        """
        if fact not in self.atom_map:
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
            self.ans_cache[fact] = oracle.ancestors(fact, self.atom_map)
        return len(self.ans_cache[fact])

    def fact_to_color(self, fact):
        # TODO(agro): do this better
        max_children = self.max_children
        if max_children == 0:
            max_children = 1
        r = 255*(max_children - self.num_children(fact))/max_children
        g = 0
        b = 0
        color = f"rgba({r},{g},{b},1)"
        return color

    def make_pyvis_net(self, net, add_loners=False, node_repulsion=True):
        added = []
        level_x = {}
        for fact in self.adjaceny_list:
            if (len(self.adjaceny_list[fact]) == 0 and self.num_ans(fact) == 0) and not add_loners:
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
                        color = self.fact_to_color(fact),
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


def stats_to_graph(path_to_stats, save_path):
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
    graph = DirectedGraph(atom_map)
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
