import itertools
import json
import os
import pickle
import sys
from dataclasses import dataclass
from copy import copy

import numpy as np
import torch
from learning.data_models import (HyperModelInfo, InvocationInfo, ModelInfo,
                                  ProblemInfo, StreamInstanceClassifierInfo)
from learning.pddlstream_utils import objects_from_facts
from torch_geometric.data import Data

FILEPATH, _ = os.path.split(os.path.realpath(__file__))


def construct_scene_graph(model_poses):
    """
    Construct a scene graph where the nodes are objects in the world and
    the edges are the relative transformations between those objects.

    params:
        model_poses: is a list of tuples of the form
        (<name>, X_WG), where X_WG is a pydrake RigidTransform

    node attributes:
        - is static or a manipuland (boolean)
    edge_attributes:
        - a length 3 vector representing the relative translation between
            the nodes the edge connects
            #TODO(agro): is rotation nessecary?

    #TODO(agro): how to encode object identity?
    """
    nodes = []
    node_attr = []
    edges = []
    edge_attr = []

    node_to_ind = {}
    for i, model_info in enumerate(model_poses):
        name = model_info["name"]
        node_to_ind[name] = i
        nodes.append(name)
        node_attr.append(
            {
                "static": model_info["static"],
                "name": name,
                "worldpose": model_info["X"]
            }
        )
    
    for o1, o2 in itertools.combinations(nodes, r= 2):
        i = node_to_ind[o1]
        j = node_to_ind[o2]
        Xi = model_poses[i]["X"]
        Xj = model_poses[j]["X"]
        edges.append((i,j))
        edge_attr.append({
            "p": Xj.translation() - Xi.translation()
        })
        edges.append((j,i))
        edge_attr.append({
            "p": Xi.translation() - Xj.translation()
        })

    return nodes, node_attr, edges, edge_attr

def construct_object_hypergraph(
    label: InvocationInfo, problem_info: ProblemInfo, model_info: ModelInfo
):
    """
    Construct an object hypergraph where nodes are pddl objects and edges are
    the facts that relate them. The hypergraph is represente as a normal graph
    with the hyperedges turned into multiple normal edges (HypergraphConv does
    not work)

    node attributes:
        - overlap_with_goal (bool)
        - stream (string)
        - level: the number of stream instantiations required to create this object (int)

    edge_attributes: 
        - predicate (string)
        - overlap_with_goal (bool)
        - level (int) 
        - stream (string)
        - actions it is precondition of (list[string])
        - actions it is poscondition of (list[string])
        - level: the number of stream insantiations required to certify this fact (int)

    TODO(agro): level vs initial?
    TODO(agro): index of domain fact
    """

    def obj_level(obj):
        if obj not in label.object_stream_map:
            return 0
        else:
            return 1 + max(
                [obj_level(o) for o in label.object_stream_map[obj]["input_objects"]]
            )

    def fact_level(fact):
        if len(label.atom_map[fact]) == 0:
            return 0
        else:
            return 1 + max([fact_level(d) for d in label.atom_map[fact]])

    goal_facts = problem_info.goal_facts
    nodes, edges = [], []
    node_attr, edge_attr = [], []
    node_to_ind = {}
    goal_objects = objects_from_facts(goal_facts)

    for fact in label.atom_map:
        fact_objects = objects_from_facts([fact])
        for o in fact_objects:
            if o in nodes:
                continue
            node_to_ind[o] = len(nodes)
            nodes.append(o)
            node_attr.append(
                {
                    "overlap_with_goal": o in goal_objects,
                    "stream": label.object_stream_map.get(o, None),
                    "level": obj_level(o)
                }
            )
        
        if len(fact_objects) == 1: #unary
            o = fact_objects.pop()
            edges.append((node_to_ind[o], node_to_ind[o]))
            edge_attr.append(
                {
                    "predicate": fact[0],
                    "overlap_with_goal": fact_objects.intersection(goal_objects),
                    "level": obj_level(fact),
                    "stream": label.stream_map[fact],
                    "relevant_actions": fact_to_relevant_actions(
                        fact, model_info.domain, indices = False
                    ),
                    "full_fact": fact # not used as an attribute, but for indexing
                }
            )
        else:
            for (o1,o2) in itertools.combinations(fact_objects, 2):
                edges.append((node_to_ind[o1], node_to_ind[o2]))
                edges.append((node_to_ind[o2], node_to_ind[o1]))
                attr = {
                    "predicate": fact[0],
                    "overlap_with_goal": fact_objects.intersection(goal_objects),
                    "level": obj_level(fact),
                    "stream": label.stream_map[fact],
                    "relevant_actions": fact_to_relevant_actions(
                        fact, model_info.domain, indices = False
                    ),
                    "full_fact": fact # not used as an attribute, but for indexing
                }
                edge_attr.append(attr)
                edge_attr.append(attr)
    return nodes, node_attr, edges, edge_attr


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
        edges.append((i, i))
        edge_attributes_list.append({
            "is_directed": False,
            "stream": None,
            "domain_index": -1,
            "via_objects": []
        })

    for fact, i in fact_to_idx.items():
        fact_objects = objects_from_facts([fact])
        node_attributes = {
            "predicate": fact[0],
            "concerns": fact_objects,
            "overlap_with_goal": fact_objects.intersection(goal_objects),
            "is_initial": not atom_map[fact],
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
                    "domain_index": domain_idx, # meant to encode the position in which
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
        edges.append((j, i))
        edge_attributes_list.append(edge_attributes)
    return nodes, node_attributes_list, edges, edge_attributes_list


def construct_hypermodel_input(
    label: InvocationInfo, problem_info: ProblemInfo, model_info: ModelInfo, 
):
    nodes, node_attr, edges, edge_attr = construct_object_hypergraph(
        label,
        problem_info,
        model_info
    )

    # indices of input objects to latest stream result
    fact_to_edge_ind = {}
    for i, attr in enumerate(edge_attr):
        fact_to_edge_ind.setdefault(attr['full_fact'], []).append(i)
    candidate_fn = lambda result: (model_info.stream_to_index[result.name], ) \
        + tuple([nodes.index(p) for p in result.input_objects]) \
        + tuple([i for dom_fact in result.domain for i in fact_to_edge_ind[dom_fact]])

    candidate = candidate_fn(label.result)

    # indices of domain facts of latest stream result
    node_features = torch.zeros(
        (len(nodes), model_info.node_feature_size), dtype=torch.float
    )
    edge_features = torch.zeros(
        (len(edges), model_info.edge_feature_size), dtype=torch.float
    )
    for i, attr in enumerate(node_attr):
        stream = None
        if attr["stream"] is not None:
            stream = attr["stream"]["name"]
        node_features[i, model_info.stream_to_index[stream]] = 1
        node_features[i, -2] = int(attr["overlap_with_goal"])
        node_features[i, -1] = attr["level"]
    num_preds = model_info.num_predicates
    num_actions = model_info.num_actions

    for i, attr in enumerate(edge_attr):
        ind = num_preds
        # predicate one hot
        edge_features[i, model_info.predicate_to_index[attr["predicate"]]] = 1
        # action precondition multi hot
        for action in attr["relevant_actions"][0]:
            edge_features[i, ind + model_info.action_to_index[action]] = 1
        ind += num_actions
        # action effect multi hot
        for action in attr["relevant_actions"][1]:
            edge_features[i, ind + model_info.action_to_index[action]] = 1
        ind += num_actions
        # stream one hot
        edge_features[i, ind + model_info.stream_to_index[attr["stream"]]] = 1
        ind += len(model_info.stream_to_index)
        # level
        edge_features[i, ind] = attr['level']
        # overlap_with_goal
        edge_features[i, ind + 1] = len(attr["overlap_with_goal"])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(
        nodes=nodes,
        x=node_features,
        edge_attr=edge_features,
        edge_index=edge_index,
        candidate=candidate,
        candidate_fn=candidate_fn
    )



def construct_input(data_obj, problem_info, model_info):
    nodes, node_attributes_list, edges, edge_attributes_list = construct_fact_graph(problem_info.goal_facts, data_obj.atom_map, data_obj.stream_map)

    candidate_fn = lambda result: (model_info.stream_to_index[result.name], ) \
        + tuple([nodes.index(p) for p in result.domain])
    candidate = candidate_fn(data_obj.result)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_features = torch.zeros((len(nodes), model_info.node_feature_size), dtype=torch.float)
    for i, (node, attr) in enumerate(zip(nodes, node_attributes_list)):
        node_features[i, model_info.predicate_to_index[attr['predicate']]] = 1
        node_features[i, -1] = len(attr['overlap_with_goal'])
        node_features[i, -2] = int(attr['is_initial'])

    edge_features = torch.zeros((len(edges), model_info.edge_feature_size), dtype=torch.float)
    for i, (edge, attr) in enumerate(zip(edges, edge_attributes_list)):
        edge_features[i, model_info.stream_to_index[attr['stream']]] = 1
        edge_features[i, -3] = attr['domain_index']
        edge_features[i, -2] = int(attr['is_directed'])
        edge_features[i, -1] = len(attr['via_objects'])

    objects_data = get_object_graph(data_obj.atom_map, model_info)

    nodes_ind = []
    for n in nodes:
        predicate = n[0]
        objects = n[1:]
        nodes_ind.append([model_info.predicate_to_index[predicate]] + [objects_data.object_to_index[o] for o in objects])

    return Data(
        nodes=nodes,
        x=node_features,
        edge_attr=edge_features,
        edge_index=edge_index,
        objects_data=objects_data,
        nodes_ind=nodes_ind,
        candidate=candidate,
        candidate_fn=candidate_fn,
    )

def get_object_predicates(object_name, atom_map):
    return set({a[0] for a in atom_map if object_name in a})

def get_object_graph(atom_map, model_info):
    objects = list(objects_from_facts([a for a in atom_map]))
    object_to_index = {o:i for i, o in enumerate(objects)}

    predicate_hot = np.zeros((len(objects), len(model_info.predicates)))
    for i, o in enumerate(objects):
        for predicate in get_object_predicates(o, atom_map):
            predicate_hot[i, model_info.predicate_to_index[predicate]] = 1
    embs = predicate_hot
    edges = set()
    for fact in atom_map:
        for source in objects:
            edges.add((object_to_index[source], object_to_index[source]))
        for source, dest in itertools.combinations(objects, 2):
            if source in fact and dest in fact:
                edges.add((object_to_index[source], object_to_index[dest]))
                edges.add((object_to_index[dest], object_to_index[source]))
    return Data(object_to_index=object_to_index,x=torch.tensor(embs, dtype=torch.float), edge_index=torch.tensor(list(edges), dtype=torch.long).t().contiguous())

def augment_labels(invocations):
    positive_results = {invocation.result for invocation in invocations if invocation.label == True}
    new_invocations = []
    for invocation in invocations:
        for result in positive_results:
            if invocation.result != result:
                if all(fact not in invocation.atom_map for fact in result.certified) \
                    and all(fact in invocation.atom_map for fact in result.domain):
                        new_invocation = copy(invocation)
                        new_invocation.result = result
                        new_invocation.label = True
                        new_invocations.append(new_invocation)
    print('New Invocations', len(new_invocations))
    return invocations + new_invocations

def parse_labels(pkl_path, augment=False):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    model_info = data['model_info']
    problem_info = data['problem_info']
    labels = data['labels'] if not augment else augment_labels(data['labels'])

    model_info = StreamInstanceClassifierInfo(
        predicates = model_info.predicates,
        streams = model_info.streams,
        stream_num_domain_facts=model_info.stream_num_domain_facts,
        stream_num_inputs=model_info.stream_num_inputs,
        stream_domains = model_info.stream_domains,
        domain = model_info.domain
    )

    dataset = []
    for invocation_info in labels: # label is now an instance of learning.gnn.oracle.Data
        d = construct_input(invocation_info, problem_info, model_info)
        d.y = torch.tensor([float(invocation_info.label)])
        dataset.append(d)
    return dataset, model_info

def parse_hyper_labels(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    model_info = data['model_info']
    problem_info = data['problem_info']

    model_info = HyperModelInfo(
        predicates = model_info.predicates,
        streams = model_info.streams,
        stream_num_domain_facts=model_info.stream_num_domain_facts,
        stream_num_inputs=model_info.stream_num_inputs,
        stream_domains = model_info.stream_domains,
        domain = model_info.domain
    )

    dataset = []
    for invocation_info in data['labels']: 
        d = construct_hypermodel_input(invocation_info, problem_info, model_info)
        d.y = torch.tensor([float(invocation_info.label)])
        dataset.append(d)
    
    return dataset, model_info

def fact_to_relevant_actions(fact, domain, indices = True):
    """
    Given a fact, determine which actions it is part of the
    preconditions and effects of.

    params:
        fact: A pddl stream fact represented as a tuple
        domain: a pddlstream.algorithms.downward.Domain object
        indices (optional): if true, return multi-hot encoding,
        else return a tuple of list of action names 
        ([appears in precondition of ], [appears in postcondition of])
    """

    assert len(fact) > 0, "must input a non-empty fact tuple"
    fact_name = fact[0]
    actions = domain.actions

    in_prec = []
    in_eff = []
    multi_hot = np.zeros(len(actions*2))

    action_index = 0
    for action in actions:
        for precondition in action.precondition.parts:
            precondition = precondition.predicate
            if fact_name == precondition:
                multi_hot[action_index] = 1
                in_prec.append(action.name)
                break
        for effect in action.effects:
            effect = effect.literal.predicate
            if fact_name == effect:
                in_eff.append(action.name)
                multi_hot[action_index + len(actions)] = 1
                break 
        action_index += 1

    if indices:
        return multi_hot

    return in_prec, in_eff

class Dataset:

    def __init__(self, construct_input_fn, model_info_class):
        self.construct_input_fn = construct_input_fn
        self.model_info_class = model_info_class
        self.problem_labels = [] 
        self.problem_infos = []
        self.datas = []
        self.model_info = None
        self.num_examples = 0

    def load_pkl(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        model_info = self.model_info_class(**data['model_info'].__dict__)
        problem_info = data['problem_info']
        if self.model_info is None:
            self.model_info = model_info
            self.model_info.domain_pddl = data['domain_pddl']
            self.model_info.stream_pddl = data['stream_pddl']
        else:
            assert self.model_info.domain_pddl == data['domain_pddl'] and self.model_info.stream_pddl == data['stream_pddl'], "Make sure the model infos in these pkls are identical!"

        self.problem_infos.append(problem_info)
        self.problem_labels.append(data['labels'])
        self.num_examples += len(data['labels'])

    def from_pkl_files(self, *file_paths):
        for file_path in file_paths:
            self.load_pkl(file_path)

    def construct_datas(self):
        datas = []
        for problem_info, labels in zip(self.problem_infos, self.problem_labels):
            data = []
            for invocation in labels:
                d = self.construct_input_fn(invocation, problem_info, self.model_info)
                d.y = torch.tensor([float(invocation.label)])
                data.append(d)
            datas.append(data)
        self.datas = datas

    def prepare(self):
        self.construct_datas()
        print(f'Prepared {self.num_examples} examples.')
    
    def __getitem__(self, index):
        if not (isinstance(index, tuple) and len(index) == 2):
            raise IndexError
        i, j = index
        return dict(
            problem_info=self.problem_infos[i],
            invocation=self.problem_labels[i][j],
            data=self.datas[i][j],
        )
    
    def __len__(self):
        return self.num_examples

    def __iter__(self):
        return (self[(i,j)]['data'] for i, labels in enumerate (self.problem_labels) for j in range(len(labels)))

class TrainingDataset(Dataset):
    def __init__(self, construct_input_fn, model_info_class, augment=False, stratify_prop=None, epoch_size=200):
        super().__init__(construct_input_fn, model_info_class)
        self.problem_labels_partitions = []
        self.input_result_mappings = []
        self.augment = augment
        self.epoch_size = epoch_size
        self.stratify_prop=stratify_prop
    
    def construct_input_result_map(self):
        self.input_result_mappings = []
        for labels in self.problem_labels:
            self.input_result_mappings.append(self.compute_possible_pairings(labels))
    
    def compute_possible_pairings(self, invocations):
        possible_pairs = {i:[i] for i in range(len(invocations))}

        if not self.augment:
            return possible_pairs

        for i, j in itertools.permutations(range(len(invocations)), 2):
            invocation1 = invocations[i]
            invocation2 = invocations[j]
            if invocation1.result == invocation2.result:
                continue
            # the atom map of invocation1 could support addition of result from invocation2
            if all(fact not in invocation1.atom_map for fact in invocation2.result.certified) \
                and all(fact in invocation1.atom_map for fact in invocation2.result.domain):
                    possible_pairs[j].append(i)
        return possible_pairs
    
    def construct_label_partition_map(self):
        self.problem_labels_partitions = []
        for labels in self.problem_labels:
            partitions = ([], [])
            self.problem_labels_partitions.append(partitions)
            for i,label in enumerate(labels):
                if not label.label:
                    partitions[0].append(i)
                else:
                    partitions[1].append(i)

    def construct_global_index(self):
        all_pos = []
        all_neg = []
        for i, (neg, pos) in enumerate(self.problem_labels_partitions):
            all_pos.extend([(i, j) for j in pos])
            all_neg.extend([(i, j) for j in neg])
        self.pos = all_pos
        self.neg = all_neg

    def __getitem__(self, index):
        if not (isinstance(index, tuple) and len(index) == 2):
            raise IndexError
        problem_index, invocation_index = index
        return dict(
            problem_info=self.problem_infos[problem_index],
            invocation=self.problem_labels[problem_index][invocation_index],
            data=self.datas[problem_index][invocation_index],
            possible_pairings=[(problem_index, k) for k in self.input_result_mappings[problem_index][invocation_index]]
        )


    def prepare(self):
        self.construct_input_result_map()
        self.construct_label_partition_map()
        self.construct_global_index()
        self.construct_datas()

    def combine_pair(self, graph_data, result_data, invocation):
        # use the graph from data1 and result from data2
        data = copy(graph_data)
        data.candidate = graph_data.candidate_fn(invocation.result)
        data.y = result_data.y
        return data


    def select_example(self):
        if self.stratify_prop is None:
            return self.global_index[self.i]

        if np.random.random() < self.stratify_prop:
            return self.pos[np.random.choice(len(self.pos))]
        else:
            return self.neg[np.random.choice(len(self.neg))]

    def initialize_random_order(self):
        self.global_index = self.pos + self.neg
        np.random.shuffle(self.global_index)

    def __iter__(self):
        if self.stratify_prop is None:
            self.initialize_random_order()
        self.i = 0
        return self

    def __len__(self):
        return self.epoch_size

    def __next__(self):
        if self.i > self.epoch_size:
            raise StopIteration
        example_idx = self.select_example()
        self.i += 1

        result_example = self[example_idx]
        pairings = result_example['possible_pairings']
        idx = pairings[np.random.choice(len(pairings))]
        if not self.augment or idx == example_idx:
            return result_example['data']
        graph_example = self[idx]
        d = self.combine_pair(
            graph_data=graph_example['data'],
            result_data=result_example['data'],
            invocation=result_example['invocation']
        )
        return d

@dataclass
class DifficultClasses:

    easy = [
        ("run_time", lambda t: t <= 15)
    ]
    medium = [
        ("run_time", lambda t: 15 < t <= 60)
    ]
    hard = [
        ("run_time", lambda t: 60 < t <= 180)
    ]
    very_hard = [
        ("run_time", lambda t: t > 180)
    ]

  

def query_data(pddl: str, query: list):
    """
    Returns a list of paths to pkl files that satisfy the query

    params:
        pddl: a concatentation of the strings of domain.pddl and stream.pddl
        query: a list of the form:
        [
            (run_attr1, run_attr2, ... , asses_func)
            ...
        ]
        where run_attr is a parameter of the run being searched for. The attributes
        of the returned pkl files must be a superset of the query attributes.
        A candidate meets the query in a certain attribute
        if asses_func(candidate[run_attr1], candidate[run_attr2], ...) is True

        Parameters for the kitchen include:
            - num_cabbages
            - num_glasses
            - num_raddishes
            - prob_sink
            - prob_tray
            - buffer_radius
            - num_goal
        parameters for blocks world include:
            - num_blocks
            - num_blockers
            - max_start_stack
            - max_goal_stack
            - buffer_radius
    """
    datapath = "/".join(FILEPATH.split("/")[:-1]) + "/data/labeled/"
    data_info_path = datapath + 'data_info.json'
    assert os.path.isfile(data_info_path), f"{data_info_path} does not exist yet"
    with open(data_info_path, "r") as f:
        info = json.load(f)
    assert pddl in info, "This domain.pddl and stream.pddl cannot be found in previous runs"
    info = info[pddl]
    datafiles = []
    for (run_attr, filename) in info:
        sat = True
        for item in query:
            attrs = item[:-1]
            asses = item[-1]
            args = tuple()
            for attr in attrs:
                if attr not in run_attr:
                    sat = False
                    break
                else:
                    args += (run_attr[attr],)
            if not sat or not asses(*args):
                sat = False
                break
        if sat:
            datafiles.append(filename)
    return datapath, datafiles

if __name__ == '__main__':

    datafile = open("/home/agrobenj/drake-tamp/learning/data/labeled/data_info.json", "r")
    data = json.load(datafile)
    blocks_world = None
    kitchen = None
    for pddl in data:
        if "blocks_world" in pddl:
            blocks_world = pddl
        if "kitchen" in pddl:
            kitchen = pddl

    if kitchen is not None:
        query = DifficultClasses.easy
        print(query_data(kitchen, query))
        query = DifficultClasses.medium
        print(query_data(kitchen, query))

    """
    if len(sys.argv) < 2:
        print('Usage error: Pass in a pkl path.')
        sys.exit(1)
    dataset = TrainingDataset(construct_hypermodel_input, HyperModelInfo, augment=True)
    dataset.from_pkl_files(*sys.argv[1:])
    dataset.prepare()
    for x in dataset:
        print(x)
    """