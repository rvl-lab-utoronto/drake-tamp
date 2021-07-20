import itertools
import json
from operator import mod
import os
import pickle
import sys
from dataclasses import dataclass
from copy import copy

import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import matplotlib
from torch.utils.data.sampler import Sampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from learning.data_models import (
    HyperModelInfo,
    InvocationInfo,
    ModelInfo,
    ProblemInfo,
    StreamInstanceClassifierInfo,
)
from learning.pddlstream_utils import objects_from_facts, ancestors, siblings, elders
from torch_geometric.data import Data
from tqdm import tqdm

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
            {"static": model_info["static"], "name": name, "worldpose": model_info["X"]}
        )

    for o1, o2 in itertools.combinations(nodes, r=2):
        i = node_to_ind[o1]
        j = node_to_ind[o2]
        Xi = model_poses[i]["X"]
        Xj = model_poses[j]["X"]
        edges.append((i, j))
        edge_attr.append({"p": Xj.translation() - Xi.translation()})
        edges.append((j, i))
        edge_attr.append({"p": Xi.translation() - Xj.translation()})

    return nodes, node_attr, edges, edge_attr


def get_ancestor_objects(label: InvocationInfo):
    """
    Given an InvocationInfo for a stream result,
    return all ancestor objects to that result
    as a set of string
    """

    def obj_ans(obj):
        if obj not in label.object_stream_map:
            return set()
        else:
            ans = set()
            for o in label.object_stream_map[obj]["input_objects"]:
                ans |= obj_ans(o) | {o}

            return ans

    res = set()
    for inp in label.result.input_objects:
        res.add(inp)
        res |= obj_ans(inp)

    return res


def get_initial_objects(label: InvocationInfo):

    """
    Given a stream result, return all the initial objects
    in that problem as a set.
    """

    initial_objects = set()
    for fact in label.atom_map:
        if len(label.atom_map[fact]) == 0:
            initial_objects |= objects_from_facts([fact])
    return initial_objects


def obj_level(obj, label):
    if obj not in label.object_stream_map:
        return 0
    else:
        return 1 + max(
            [obj_level(o, label) for o in label.object_stream_map[obj]["input_objects"]]
        )


def fact_level(fact, label):
    if len(label.atom_map[fact]) == 0:
        return 0
    else:
        return 1 + max([fact_level(d, label) for d in label.atom_map[fact]])


def construct_object_hypergraph(
    label: InvocationInfo,
    problem_info: ProblemInfo,
    model_info: ModelInfo,
    reduced: bool = True,
):
    """
    Construct an object hypergraph where nodes are pddl objects and edges are
    the facts that relate them. The hypergraph is represente as a normal graph
    with the hyperedges turned into multiple normal edges (HypergraphConv does
    not work)

    reduced: if True, the returned graph consists of only the ancestors of the stream result
        and the initial conditions (objects + facts)

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
        - the one hot encoding of the source object argument index
        - the one hot encoding of the destination object argument index

    TODO(agro): level vs initial?
    TODO(agro): index of domain fact
    """

    goal_facts = problem_info.goal_facts
    nodes, edges = [], []
    node_attr, edge_attr = [], []
    node_to_ind = {}
    goal_objects = objects_from_facts(goal_facts)

    # reduced_obj = get_ancestor_objects(label) | get_initial_objects(label)
    fact_ans = set()
    for dom_fact in label.result.domain:
        fact_ans.add(dom_fact)
        fact_ans |= siblings(dom_fact, label.atom_map)
        fact_ans |= elders(dom_fact, label.atom_map)

    for fact in label.atom_map:
        fact_objects = objects_from_facts([fact])
        if reduced and label.atom_map[fact]:
            if not fact in fact_ans:
                continue
        for o in fact_objects:
            if o in nodes:
                continue
            node_to_ind[o] = len(nodes)
            nodes.append(o)
            node_attr.append(
                {
                    "overlap_with_goal": o in goal_objects,
                    "stream": label.object_stream_map.get(o, None),
                    "level": obj_level(o, label),
                }
            )

        if len(fact_objects) == 1:  # unary
            o = fact_objects.pop()
            edges.append((node_to_ind[o], node_to_ind[o]))
            edge_attr.append(
                {
                    "predicate": fact[0],
                    "overlap_with_goal": fact_objects.intersection(goal_objects),
                    "level": fact_level(fact, label),
                    "stream": label.stream_map[fact],
                    "relevant_actions": fact_to_relevant_actions(
                        fact, model_info.domain, indices=False
                    ),
                    "full_fact": fact,  # not used as an attribute, but for indexing
                    "src_obj_arg_ind": 0,
                    "dest_obj_arg_ind": 0,
                }
            )
        else:
            for (o1, o2) in itertools.permutations(fact_objects, 2):
                edges.append((node_to_ind[o1], node_to_ind[o2]))
                attr = {
                    "predicate": fact[0],
                    "overlap_with_goal": fact_objects.intersection(goal_objects),
                    "level": fact_level(fact, label),
                    "stream": label.stream_map[fact],
                    "relevant_actions": fact_to_relevant_actions(
                        fact, model_info.domain, indices=False
                    ),
                    "full_fact": fact,  # not used as an attribute, but for indexing
                    "src_obj_arg_ind": fact.index(o1) - 1,
                    "dest_obj_arg_ind": fact.index(o2) - 1,
                }
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
        edge_attributes_list.append(
            {
                "is_directed": False,
                "stream": None,
                "domain_index": -1,
                "via_objects": [],
            }
        )

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
                    "via_objects": objects_from_facts([parent]).intersection(
                        fact_objects
                    ),
                    "domain_index": domain_idx,  # meant to encode the position in which
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
            "via_objects": object_edges[(i, j)],
        }
        edges.append((i, j))
        edge_attributes_list.append(edge_attributes)
        edges.append((j, i))
        edge_attributes_list.append(edge_attributes)
    return nodes, node_attributes_list, edges, edge_attributes_list


def construct_hypermodel_input(
    label: InvocationInfo,
    problem_info: ProblemInfo,
    model_info: ModelInfo,
    reduced: bool = False,
):
    nodes, node_attr, edges, edge_attr = construct_object_hypergraph(
        label, problem_info, model_info, reduced=reduced
    )

    # indices of input objects to latest stream result
    fact_to_edge_ind = {}
    for i, attr in enumerate(edge_attr):
        fact_to_edge_ind.setdefault(attr["full_fact"], []).append(i)

    candidate = (
        (model_info.stream_to_index[label.result.name],)
        + tuple([nodes.index(p) for p in label.result.input_objects])
        + tuple(
            [i for dom_fact in label.result.domain for i in fact_to_edge_ind[dom_fact]]
        )
    )

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
        edge_features[i, ind] = attr["level"]
        # overlap_with_goal
        edge_features[i, ind + 1] = len(attr["overlap_with_goal"])
        ind += 2
        edge_features[i, ind + attr["src_obj_arg_ind"]] = 1
        ind += model_info.max_predicate_num_args
        edge_features[i, ind + attr["dest_obj_arg_ind"]] = 1

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(
        nodes=nodes,
        x=node_features,
        edge_attr=edge_features,
        edge_index=edge_index,
        candidate=candidate,
    )


def construct_problem_graph(problem_info: ProblemInfo):
    """
    Construct an object graph where nodes are pddl objects and edges are facts.
    Nodes are included if they are in the problem's initial conditions, and
    facts are included if they appear either in the goal or initial conditions.

    node attributes:
        - has_pose: whether the object is something that can have a world pose
        - pose: The RigidTransform of the object in the world frame if has_pose
            is true, otherwise None

    edge attributes:
        - predicate: the name of the pddl predicate (including axioms)
        - is_initial: True if the fact is part of the initial conditions
    """
    model_pose_dict = {pose["name"]: pose for pose in problem_info.model_poses}
    initial_facts = problem_info.initial_facts
    edges = []
    edge_attributes = []
    nodes = []
    node_attributes = []
    node_to_index = {}
    for fact in initial_facts:
        fact_objects = objects_from_facts([fact])
        for obj in fact_objects:
            if obj not in nodes:
                attr = {}
                nodes.append(obj)
                node_attributes.append(attr)
                node_to_index[obj] = len(nodes) - 1

                name = problem_info.object_mapping[obj]
                attr["has_pose"] = name.__hash__ is not None and name in model_pose_dict
                if attr["has_pose"]:
                    attr["pose"] = model_pose_dict[name]["X"]
                else:
                    attr["pose"] = None
        if len(fact_objects) == 1:
            obj = fact_objects.pop()
            edge_attributes.append({"predicate": fact[0], "is_initial": True})
            edges.append((node_to_index[obj],) * 2)
        for obj1, obj2 in itertools.permutations(fact_objects, 2):
            edges.append((node_to_index[obj1], node_to_index[obj2]))
            edge_attributes.append({"predicate": fact[0], "is_initial": True})

    for fact in problem_info.goal_facts:
        fact_objects = objects_from_facts([fact])
        if len(fact_objects) == 1:
            edge_attributes.append({"predicate": fact[0], "is_initial": False})
            obj = fact_objects.pop()
            edges.append((node_to_index[obj],) * 2)
        for obj1, obj2 in itertools.permutations(fact_objects, 2):
            edges.append((node_to_index[obj1], node_to_index[obj2]))
            edge_attributes.append({"predicate": fact[0], "is_initial": False})
    return nodes, node_attributes, edges, edge_attributes


def construct_problem_graph_input(problem_info: ProblemInfo, model_info: ModelInfo):
    nodes, node_attr, edges, edge_attr = problem_info.problem_graph
    # construct_problem_graph(problem_info)

    node_features = torch.zeros((len(nodes), 4), dtype=torch.float)
    edge_features = torch.zeros(
        (len(edges), model_info.num_predicates + 1), dtype=torch.float
    )
    for i, attr in enumerate(edge_attr):
        edge_features[i, model_info.predicate_to_index[attr["predicate"]]] = 1
        edge_features[i, -1] = int(attr["is_initial"])

    for i, attr in enumerate(node_attr):
        node_features[i, 0] = int(attr["has_pose"])
        if attr["has_pose"]:
            for k, v in enumerate(attr["pose"].translation()):
                node_features[i, 1 + k] = v

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(
        nodes=nodes,
        x=node_features,
        edge_attr=edge_features,
        edge_index=edge_index,
    )


def construct_with_problem_graph(input_fn):
    def new_fn(invocation, problem, model_info):
        data = input_fn(invocation, problem, model_info)
        data.problem_graph = construct_problem_graph_input(problem, model_info)
        return data

    return new_fn


def construct_input(data_obj, problem_info, model_info):
    nodes, node_attributes_list, edges, edge_attributes_list = construct_fact_graph(
        problem_info.goal_facts, data_obj.atom_map, data_obj.stream_map
    )

    candidate_fn = lambda result: (model_info.stream_to_index[result.name],) + tuple(
        [nodes.index(p) for p in result.domain]
    )
    candidate = candidate_fn(data_obj.result)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_features = torch.zeros(
        (len(nodes), model_info.node_feature_size), dtype=torch.float
    )
    for i, (node, attr) in enumerate(zip(nodes, node_attributes_list)):
        node_features[i, model_info.predicate_to_index[attr["predicate"]]] = 1
        node_features[i, -1] = len(attr["overlap_with_goal"])
        node_features[i, -2] = int(attr["is_initial"])

    edge_features = torch.zeros(
        (len(edges), model_info.edge_feature_size), dtype=torch.float
    )
    for i, (edge, attr) in enumerate(zip(edges, edge_attributes_list)):
        edge_features[i, model_info.stream_to_index[attr["stream"]]] = 1
        edge_features[i, -3] = attr["domain_index"]
        edge_features[i, -2] = int(attr["is_directed"])
        edge_features[i, -1] = len(attr["via_objects"])

    objects_data = get_object_graph(data_obj.atom_map, model_info)

    nodes_ind = []
    for n in nodes:
        predicate = n[0]
        objects = n[1:]
        nodes_ind.append(
            [model_info.predicate_to_index[predicate]]
            + [objects_data.object_to_index[o] for o in objects]
        )

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
    object_to_index = {o: i for i, o in enumerate(objects)}

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
    return Data(
        object_to_index=object_to_index,
        x=torch.tensor(embs, dtype=torch.float),
        edge_index=torch.tensor(list(edges), dtype=torch.long).t().contiguous(),
    )


def augment_labels(invocations):
    positive_results = {
        invocation.result for invocation in invocations if invocation.label == True
    }
    new_invocations = []
    for invocation in invocations:
        for result in positive_results:
            if invocation.result != result:
                if all(
                    fact not in invocation.atom_map for fact in result.certified
                ) and all(fact in invocation.atom_map for fact in result.domain):
                    new_invocation = copy(invocation)
                    new_invocation.result = result
                    new_invocation.label = True
                    new_invocations.append(new_invocation)
    print("New Invocations", len(new_invocations))
    return invocations + new_invocations


def parse_labels(pkl_path, augment=False):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    model_info = data["model_info"]
    problem_info = data["problem_info"]
    labels = data["labels"] if not augment else augment_labels(data["labels"])

    model_info = StreamInstanceClassifierInfo(
        predicates=model_info.predicates,
        streams=model_info.streams,
        stream_num_domain_facts=model_info.stream_num_domain_facts,
        stream_num_inputs=model_info.stream_num_inputs,
        stream_domains=model_info.stream_domains,
        domain=model_info.domain,
    )

    dataset = []
    for (
        invocation_info
    ) in labels:  # label is now an instance of learning.gnn.oracle.Data
        d = construct_input(invocation_info, problem_info, model_info)
        d.y = torch.tensor([float(invocation_info.label)])
        dataset.append(d)
    return dataset, model_info


def parse_hyper_labels(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    model_info = data["model_info"]
    problem_info = data["problem_info"]

    model_info = HyperModelInfo(
        predicates=model_info.predicates,
        streams=model_info.streams,
        stream_num_domain_facts=model_info.stream_num_domain_facts,
        stream_num_inputs=model_info.stream_num_inputs,
        stream_domains=model_info.stream_domains,
        domain=model_info.domain,
    )

    dataset = []
    for invocation_info in data["labels"]:
        d = construct_hypermodel_input(invocation_info, problem_info, model_info)
        d.y = torch.tensor([float(invocation_info.label)])
        dataset.append(d)

    return dataset, model_info


def fact_to_relevant_actions(fact, domain, indices=True):
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
    multi_hot = np.zeros(len(actions * 2))

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


class DataStore:

    def __init__(self, model_info: ModelInfo, problem_info: ProblemInfo, label_paths:list, do_cache: bool = False):
        self.model_info = model_info
        self.problem_info = problem_info
        self.label_paths = label_paths
        self.do_cache = do_cache
        self.cache = [None for _ in label_paths]

    def __next__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError(
                f"You have asked for label at index {i} when there are only {len(self)} labels"
            )
        if self.cache[i] is not None:
            return self.cache[i]
        with open(self.label_paths[i], "rb") as f:
            label = pickle.load(f)
        if self.do_cache:
            self.cache[i] = label
        return label

    def __len__(self):
        return len(self.label_paths)


class Dataset:
    def __init__(
        self,
        construct_input_fn,
        model_info_class,
        preprocess_all=True,
        clear_memory=True,
    ):
        self.construct_input_fn = construct_input_fn
        self.model_info_class = model_info_class
        #self.problem_labels = []
        #self.problem_infos = []
        self.datas = []
        self.model_info = None
        self.num_examples = 0
        self.preprocess_all = preprocess_all
        self.clear_memory = clear_memory
        self.datastores = []

    def load_pkl(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        model_info = self.model_info_class(**data["model_info"].__dict__)
        problem_info = data["problem_info"]
        if self.model_info is None:
            self.model_info = model_info
            self.model_info.domain_pddl = data["domain_pddl"]
            self.model_info.stream_pddl = data["stream_pddl"]
        else:
            assert (
                self.model_info.domain_pddl == data["domain_pddl"]
                and self.model_info.stream_pddl == data["stream_pddl"]
            ), "Make sure the model infos in these pkls are identical!"

        dirpath = os.path.splitext(file_path)[0]
        label_paths = [
            os.path.join(dirpath, f"label_{i}.pkl") for i in range(data["num_labels"])
        ]

        self.datastores.append(
            DataStore(
                model_info,
                problem_info,
                label_paths,
            )
        )

        # self.problem_infos.append(problem_info)
        # self.problem_labels.append(data['labels'])
        self.num_examples += data['num_labels']

    def from_pkl_files(self, *file_paths):
        """
        Take in a list of *top level* pickle files (ie containing
        ModelInfo, ProblemInfo, ...). These will all be loaded to start.
        """
        # for file_path in tqdm(file_paths):
        # self.load_pkl(file_path)
        print("Loading top level pkls")
        for file_path in tqdm(file_paths):
            self.load_pkl(file_path)

    def construct_datas(self):
        datas = []
        print(f"Constructing datas. self.preprocess_all: {self.preprocess_all}")
        pbar = tqdm(total=self.num_examples)
        for datastore in self.datastores:
            data = []
            for i in range(len(datastore)):
                if self.preprocess_all:
                    invocation = datastore[i]
                    d = self.construct_datum(invocation, datastore.problem_info)
                else:
                    d = None
                pbar.update(1)
                data.append(d)
            datas.append(data)
        self.datas = datas

    def construct_datum(self, invocation, problem_info):
        d = self.construct_input_fn(invocation, problem_info, self.model_info)
        d.y = torch.tensor([float(invocation.label)])
        return d

    def prepare(self):
        self.construct_datas()
        print(f"Prepared {self.num_examples} examples.")

    def get_dict(self, index):
        if not (isinstance(index, tuple) and len(index) == 2):
            raise IndexError
        i, j = index
        if not self.preprocess_all and self.datas[i][j] is None:
            datastore = self.datastores[i]
            self.datas[i][j] = self.construct_datum(
                datastore[j], datastore.problem_info
            )
        result = dict(
            problem_info=self.datastores[i].problem_info,
            invocation=self.datastores[i][j],
            data=self.datas[i][j],
        )
        if not self.preprocess_all and self.clear_memory:
            self.datas[i][j] = None
        return result

    def __getitem__(self, index):
        data = self.get_dict(index)["data"]
        data.problem_index = [
            index[0]
        ]  # This list crazyness is to prevent torch geometric from batching
        return data

    def __len__(self):
        return self.num_examples


class EvaluationDatasetSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return (
            (problem_index, invocation_index)
            for problem_index, datastore in enumerate(self.dataset.datastores)
            for invocation_index in range(len(datastore))
        )

    def __len__(self):
        return len(self.dataset)

class TrainingDataset(Dataset):
    def __init__(
        self,
        construct_input_fn,
        model_info_class,
        preprocess_all=True,
        clear_memory=True,
    ):
        super().__init__(
            construct_input_fn,
            model_info_class,
            preprocess_all=preprocess_all,
            clear_memory=clear_memory,
        )
        self.problem_labels_partitions = []
        self.pos = []
        self.neg = []

    def construct_datas(self):
        datas = []
        self.problem_labels_partitions = []
        print(f"Constructing training datas. self.preprocess_all: {self.preprocess_all}")
        pbar = tqdm(total=self.num_examples)
        for datastore in self.datastores:
            data = []
            partitions = ([], [])
            self.problem_labels_partitions.append(partitions)
            for i, invocation in enumerate(datastore):
                partitions[int(invocation.label)].append(i)
                if self.preprocess_all:
                    d = self.construct_datum(invocation, datastore.problem_info)
                else:
                    d = None
                data.append(d)
                pbar.update(1)
            datas.append(data)
        self.datas = datas
        all_pos = []
        all_neg = []
        for i, (neg, pos) in enumerate(self.problem_labels_partitions):
            all_pos.extend([(i, j) for j in pos])
            all_neg.extend([(i, j) for j in neg])
        self.pos = all_pos
        self.neg = all_neg

    def prepare(self):
        self.construct_datas()
        print(f"Prepared {self.num_examples} examples.")


class TrainingDatasetSampler(Sampler):
    def __init__(self, dataset, epoch_size=200, stratify_prop=None):
        self.dataset = dataset
        self.i = None
        self.stratify_prop = stratify_prop
        self.global_index = None
        self.epoch_size = epoch_size

    def initialize_random_order(self):
        self.global_index = self.dataset.pos + self.dataset.neg
        np.random.shuffle(self.global_index)

    def select_example(self):
        if self.stratify_prop is None:
            return self.global_index[self.i]

        if np.random.random() < self.stratify_prop:
            return self.dataset.pos[np.random.choice(len(self.dataset.pos))]
        else:
            return self.dataset.neg[np.random.choice(len(self.dataset.neg))]

    def __iter__(self):
        if self.stratify_prop is None:
            self.initialize_random_order()
        self.i = 0
        return self

    def __next__(self):
        if self.i >= min(self.epoch_size, len(self.dataset)):
            raise StopIteration
        example_idx = self.select_example()
        self.i += 1
        return example_idx

    def __len__(self):
        return self.epoch_size

class DeviceAwareLoaderWrapper:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self

    def __next__(self):
        data = next(self.iterator)
        data.to(self.device)
        if hasattr(data, "problem_graph"):
            for problem in data.problem_graph:
                problem.to(self.device)
        return data

    def __len__(self):
        return len(self.dataloader)


@dataclass
class DifficultClasses:

    easy = [("run_time", lambda t: t <= 15)]
    medium = [("run_time", lambda t: 15 < t <= 60)]
    hard = [("run_time", lambda t: 60 < t <= 180)]
    very_hard = [("run_time", lambda t: t > 180)]


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
    datapath = get_base_datapath()
    data_info_path = datapath + "data_info.json"
    assert os.path.isfile(data_info_path), f"{data_info_path} does not exist yet"
    with open(data_info_path, "r") as f:
        info = json.load(f)
    assert (
        pddl in info
    ), "This domain.pddl and stream.pddl cannot be found in previous runs"
    info = info[pddl]
    datafiles = []
    for (run_attr, filename, _) in info:
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
    return datafiles


def get_base_datapath():
    sep = os.path.sep
    datapath = os.path.join(sep.join(FILEPATH.split(sep)[:-1]),"data/labeled/")
    return datapath


def get_pddl_key(domain):
    datapath = os.path.join(get_base_datapath(), "data_info.json")
    with open(datapath, "r") as datafile:
        data = json.load(datafile)
    for pddl in data:
        if domain in pddl:
            return pddl
    raise ValueError(f"{domain} not in data_info.json")


if __name__ == "__main__":

    datafile = open(
        "/home/agrobenj/drake-tamp/learning/data/labeled/data_info.json", "r"
    )
    data = json.load(datafile)
    blocks_world = None
    kitchen = None
    for pddl in data:
        if "blocks_world" in pddl:
            blocks_world = pddl
        if "kitchen" in pddl:
            kitchen = pddl

    if kitchen is not None:
        print(f"Num kitchen pkls: {len(data[kitchen])}")
        query = DifficultClasses.easy
        easy = len(query_data(kitchen, query))
        print(f"Number of easy runs {easy}")
        query = DifficultClasses.medium
        med = len(query_data(kitchen, query))
        print(f"Number of medium runs {med}")
        query = DifficultClasses.hard
        hard = len(query_data(kitchen, query))
        print(f"Number of hard runs {hard}")
        query = DifficultClasses.very_hard
        very_hard = len(query_data(kitchen, query))
        print(f"Number of very hard runs {very_hard}")

        kitchen_data = data[kitchen]

        num_c, num_r, num_g, num_goal = [], [], [], []
        sample_time, search_time, run_time = [], [], []

        for pt, _, _ in kitchen_data:
            num_c.append(pt["num_cabbages"])
            num_r.append(pt["num_raddishes"])
            num_g.append(pt["num_glasses"])
            num_goal.append(pt["num_goal"])
            sample_time.append(pt["sample_time"])
            search_time.append(pt["search_time"])
            run_time.append(pt["run_time"])

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(num_c, run_time, linestyle="", marker="o")
        axs[0, 0].set_xlabel("Number of Cabbages")
        axs[0, 0].set_ylabel("Run Time (s)")

        num_c = np.array(num_c).reshape(-1, 1)
        run_time = np.array(run_time)
        model = LinearRegression().fit(num_c, run_time)
        x = np.linspace(0, max(num_c), max(num_c) + 1)
        axs[0, 0].plot(
            x,
            x * model.coef_[0] + model.intercept_,
            color="k",
            linestyle="--",
            label=f"m: {model.coef_[0]:.2f}, b: {model.intercept_:.2f}",
        )
        axs[0, 0].legend()
        axs[0, 0].axhline(180, color="r")

        axs[0, 1].plot(num_r, run_time, linestyle="", marker="o")
        axs[0, 1].set_xlabel("Number of Raddishes")
        axs[0, 1].set_ylabel("Run Time (s)")

        num_r = np.array(num_r).reshape(-1, 1)
        run_time = np.array(run_time)
        model = LinearRegression().fit(num_r, run_time)
        x = np.linspace(0, max(num_r), max(num_r) + 1)
        axs[0, 1].plot(
            x,
            x * model.coef_[0] + model.intercept_,
            color="k",
            linestyle="--",
            label=f"m: {model.coef_[0]:.2f}, b: {model.intercept_:.2f}",
        )
        axs[0, 1].legend()
        axs[0, 1].axhline(180, color="r")

        axs[1, 0].plot(num_g, run_time, linestyle="", marker="o")
        axs[1, 0].set_xlabel("Number of Glasses")
        axs[1, 0].set_ylabel("Run Time (s)")
        num_g = np.array(num_g).reshape(-1, 1)
        run_time = np.array(run_time)
        model = LinearRegression().fit(num_g, run_time)
        x = np.linspace(0, max(num_g), max(num_g) + 1)
        axs[1, 0].plot(
            x,
            x * model.coef_[0] + model.intercept_,
            color="k",
            linestyle="--",
            label=f"m: {model.coef_[0]:.2f}, b: {model.intercept_:.2f}",
        )
        axs[1, 0].legend()
        axs[1, 0].axhline(180, color="r")

        axs[1, 1].plot(num_goal, run_time, linestyle="", marker="o")
        axs[1, 1].set_xlabel("Number of Goal Objects")
        axs[1, 1].set_ylabel("Run Time (s)")
        num_goal = np.array(num_goal).reshape(-1, 1)
        run_time = np.array(run_time)
        model = LinearRegression().fit(num_goal, run_time)
        x = np.linspace(0, max(num_goal), max(num_goal) + 1)
        axs[1, 1].plot(
            x,
            x * model.coef_[0] + model.intercept_,
            color="k",
            linestyle="--",
            label=f"m: {model.coef_[0]:.2f}, b: {model.intercept_:.2f}",
        )
        axs[1, 1].legend()
        axs[1, 1].axhline(180, color="r")

        fig.tight_layout()
        fig.set_size_inches(15, 15)
        plt.savefig(f"{FILEPATH}/plots/difficulty_plot1.png", dpi=300)

        plt.close(fig)

        fig, ax = plt.subplots()

        num_tot = list(map(lambda x: x[0] + x[1] + x[2], zip(num_c, num_r, num_g)))
        ax.plot(num_tot, run_time, linestyle="", marker="o")
        ax.set_xlabel("Total Number of Objects")
        ax.set_ylabel("Run Time (s)")
        num_tot = np.array(num_tot).reshape(-1, 1)
        run_time = np.array(run_time)
        model = LinearRegression().fit(num_tot, run_time)
        x = np.linspace(0, max(num_goal), max(num_goal) + 1)
        ax.plot(
            x,
            x * model.coef_[0] + model.intercept_,
            color="k",
            linestyle="--",
            label=f"m: {model.coef_[0]:.2f}, b: {model.intercept_:.2f}",
        )
        ax.legend()
        ax.axhline(180, color="r")
        fig.tight_layout()
        fig.set_size_inches(15, 15)
        plt.savefig(f"{FILEPATH}/plots/difficulty_plot2.png", dpi=300)

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
