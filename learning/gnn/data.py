from learning.oracle import objects_from_facts
import itertools
import torch
from torch_geometric.data import Data
import pickle

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

def data_from_label(label, goal_facts, stream_to_index, predicate_to_index, node_feature_size, edge_feature_size):
    fact, parents, candidate_stream, is_relevant, atom_map, stream_map = label
    nodes, node_attributes_list, edges, edge_attributes_list = construct_fact_graph(goal_facts, atom_map, stream_map)

    parent_idxs = [nodes.index(p) for p in parents]
    candidate = (stream_to_index[candidate_stream], ) + tuple(parent_idxs)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_features = torch.zeros((len(nodes), node_feature_size), dtype=torch.float)
    for i, (node, attr) in enumerate(zip(nodes, node_attributes_list)):
        node_features[i, predicate_to_index[attr['predicate']]] = 1
        node_features[i, -1] = len(attr['overlap_with_goal'])
        node_features[i, -2] = int(attr['is_initial'])

    edge_features = torch.zeros((len(edges), edge_feature_size), dtype=torch.float)
    for i, (edge, attr) in enumerate(zip(edges, edge_attributes_list)):
        edge_features[i, stream_to_index[attr['stream']]] = 1
        edge_features[-3] = attr['domain_index']
        edge_features[-2] = int(attr['is_directed'])
        edge_features[-1] = len(attr['via_objects'])

    return Data(
        nodes=nodes,
        x=node_features,
        edge_attr=edge_features,
        edge_index=edge_index,
        fact=fact,
        candidate=candidate,
        y=torch.tensor([int(is_relevant)])
    )


def parse_labels(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    predicates = [p.name for p in data['domain'].predicates]
    predicate_to_index = {p:i for i,p in enumerate(predicates)}
    streams = [None] + [e['name'] for e in data['externals']]
    stream_input_sizes = [None] + [len(e['domain']) for e in data['externals']]
    stream_to_index = {p:i for i,p in enumerate(streams)}

    node_feature_size = len(predicate_to_index) + 2 # one-hot for predicate, 1 for is_initial and 1 for goal overlap
    edge_feature_size = len(predicate_to_index) + 3 # one-hot for stream, 1 for is_directed, 1 for object_overlap, and 1 for domain_index

    goal_facts = data['goal_facts']
    dataset = []
    for label in data['labels']:
        d = data_from_label(label, goal_facts, stream_to_index, predicate_to_index, node_feature_size, edge_feature_size)
        dataset.append(d)

    return dataset, (stream_input_sizes, node_feature_size, edge_feature_size)

if __name__ == '__main__':
    dataset = parse_labels('/home/mohammed/drake-tamp/learning/data/labeled/2021-07-05-14:35:28.341.pkl')