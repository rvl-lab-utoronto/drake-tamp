from learning.oracle import objects_from_facts
import itertools
import torch
from torch_geometric.data import Data
import pickle
from dataclasses import dataclass

def construct_fact_graph(goal_facts, atom_map, stream_map):
    goal_objects = objects_from_facts(goal_facts)
    nodes = []
    edges = []
    node_attributes_list = []
    edge_attributes_list = []
    fact_to_idx = {}
    initial_objects = {}
    for i, fact in enumerate(atom_map):
        nodes.append(fact)
        fact_to_idx[fact] = i
    for fact, i in fact_to_idx.items():
        fact_objects = objects_from_facts([fact])
        node_attributes = {
            "predicate": fact[0],
            "concerns": fact_objects,
            "overlap_with_goal": fact_objects.intersection(goal_objects),
            "is_initial": not atom_map[fact]
        }
        node_attributes_list.append(node_attributes)

        if node_attributes["is_initial"]:
            for o in node_attributes["concerns"]:
                initial_objects.setdefault(o, set()).add(i)
        else:
            for domain_idx, parent in enumerate(atom_map[fact]):
                parent_idx = fact_to_idx[parent]
                edges.append((parent_idx, i))
                edge_attributes = {
                    "is_directed": True,
                    "stream": stream_map[fact],
                    "via_objects": objects_from_facts([parent]).intersection(fact_objects),
                    "domain_index": domain_idx # meant to encode the position in which 
                }
                edge_attributes_list.append(edge_attributes)
    
    # add object sharing edges in initial objects
    object_edges = {} 
    for o in initial_objects:
        fact_idxs = initial_objects[o]
        for i, j in itertools.combinations(fact_idxs, 2):
            key = tuple(sorted((i, j)))
            object_edges.setdefault(key, set()).add(o)
    for (i, j) in object_edges:

        edge_attributes = {
            "is_directed": False,
            "stream": None,
            "domain_index": -1,
            "via_objects": object_edges[(i, j)]
        }
        edges.append((i, j))
        edge_attributes_list.append(edge_attributes)
        edges.append((j, j))
        edge_attributes_list.append(edge_attributes)
    return nodes, node_attributes_list, edges, edge_attributes_list

@dataclass
class ModelInfo:
    goal_facts: list
    predicates: list
    streams: list
    stream_input_sizes: list

    @property
    def node_feature_size(self):
        return len(self.predicate_to_index) + 2
    
    @property
    def edge_feature_size(self):
        return len(self.stream_to_index) + 3

    @property
    def stream_to_index(self):
        return {s:i for i,s in enumerate(self.streams)}
    @property
    def predicate_to_index(self):
        return {s:i for i,s in enumerate(self.predicates)}


def construct_input(parents, candidate_stream, atom_map, stream_map, model_info):
    nodes, node_attributes_list, edges, edge_attributes_list = construct_fact_graph(model_info.goal_facts, atom_map, stream_map)

    parent_idxs = [nodes.index(p) for p in parents]
    candidate = (model_info.stream_to_index[candidate_stream], ) + tuple(parent_idxs)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_features = torch.zeros((len(nodes), model_info.node_feature_size), dtype=torch.float)
    for i, (node, attr) in enumerate(zip(nodes, node_attributes_list)):
        node_features[i, model_info.predicate_to_index[attr['predicate']]] = 1
        node_features[i, -1] = len(attr['overlap_with_goal'])
        node_features[i, -2] = int(attr['is_initial'])

    edge_features = torch.zeros((len(edges), model_info.edge_feature_size), dtype=torch.float)
    for i, (edge, attr) in enumerate(zip(edges, edge_attributes_list)):
        edge_features[i, model_info.stream_to_index[attr['stream']]] = 1
        edge_features[-3] = attr['domain_index']
        edge_features[-2] = int(attr['is_directed'])
        edge_features[-1] = len(attr['via_objects'])

    return Data(
        nodes=nodes,
        x=node_features,
        edge_attr=edge_features,
        edge_index=edge_index,
        candidate=candidate
    )


def parse_labels(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    model_info = ModelInfo(
        goal_facts=data['goal_facts'],
        predicates=[p.name for p in data['domain'].predicates],
        streams=[None] + [e['name'] for e in data['externals']],
        stream_input_sizes=[None] + [len(e['domain']) for e in data['externals']]
    )

    dataset = []
    for label in data['labels']:
        fact, parents, candidate_stream, is_relevant, atom_map, stream_map = label
        d = construct_input(parents, candidate_stream, atom_map, stream_map, model_info)
        d.y = torch.tensor([float(is_relevant)])
        dataset.append(d)

    return dataset, model_info

if __name__ == '__main__':
    dataset = parse_labels('/home/mohammed/drake-tamp/learning/data/labeled/2021-07-05-14:35:28.341.pkl')