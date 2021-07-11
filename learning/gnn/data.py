from learning.data_models import HyperModelInfo, InvocationInfo, ProblemInfo, ModelInfo, StreamInstanceClassifierInfo
import numpy as np
from learning.pddlstream_utils import objects_from_facts
import itertools
import torch
from torch_geometric.data import Data
import pickle
from copy import copy



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
    input_node_inds = [nodes.index(p) for p in label.result.input_objects]
    fact_to_edge_ind = {}
    for i, attr in enumerate(edge_attr):
        fact_to_edge_ind.setdefault(attr['full_fact'], []).append(i)
    dom_edge_inds = [i for dom_fact in label.result.domain for i in fact_to_edge_ind[dom_fact]]

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

    candidate = (model_info.stream_to_index[label.result.name], ) \
        + tuple(input_node_inds) + tuple(dom_edge_inds)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(
        nodes=nodes,
        x=node_features,
        edge_attr=edge_features,
        edge_index=edge_index,
        candidate=candidate,
        fact_to_edge_ind=fact_to_edge_ind
    )



def construct_input(data_obj, problem_info, model_info):
    nodes, node_attributes_list, edges, edge_attributes_list = construct_fact_graph(problem_info.goal_facts, data_obj.atom_map, data_obj.stream_map)

    parent_idxs = [nodes.index(p) for p in data_obj.result.domain]
    candidate = (model_info.stream_to_index[data_obj.result.name], ) + tuple(parent_idxs)

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
        candidate=candidate,
        objects_data=objects_data,
        nodes_ind=nodes_ind
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

class TrainingDataset:
    def __init__(self, construct_input_fn, model_info_class, augment=False):
        self.construct_input_fn = construct_input_fn
        self.model_info_class = model_info_class
        self.problem_labels = [] 
        self.problem_infos = []
        self.problem_labels_partitions = []
        self.input_result_mappings = []
        self.model_info = None
        self.num_examples = 0
        self.augment = augment
        self.epoch_size = 200
        self.prop = .5
    
    def load_pkl(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        model_info = self.model_info_class(**data['model_info'].__dict__)
        problem_info = data['problem_info']
        if self.model_info is None:
            self.model_info = model_info
        else:
            # TODO: this assertion doesnt work due to model_info.domain 
            # assert self.model_info == model_info
            print("Warning: Make sure the model infos in these pkls are identical!")

        self.problem_infos.append(problem_info)
        self.problem_labels.append(data['labels'])
        self.num_examples += len(data['labels'])

    def from_pkl_files(self, *file_paths):
        for file_path in file_paths:
            self.load_pkl(file_path)

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
        i, j = index
        return dict(
            invocation=self.problem_labels[i][j],
            data=self.datas[i][j],
            possible_pairings=[(i, k) for k in self.input_result_mappings[i][j]]
        )

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
        self.construct_input_result_map()
        self.construct_label_partition_map()
        self.construct_global_index()
        self.construct_datas()

    def combine_pair(self, graph_data, result_data, invocation):
        model_info = self.model_info
        # use the graph from data1 and result from data2
        input_node_inds = [graph_data.nodes.index(p) for p in invocation.result.input_objects]
        dom_edge_inds = [i for dom_fact in invocation.result.domain for i in graph_data.fact_to_edge_ind[dom_fact]]
        assert model_info.stream_to_index[invocation.result.name] == result_data.candidate[0]
        return Data(
            nodes=graph_data.nodes,
            x=graph_data.x,
            edge_attr=graph_data.edge_attr,
            edge_index=graph_data.edge_index,
            candidate=(model_info.stream_to_index[invocation.result.name], )  \
            + tuple(input_node_inds) + tuple(dom_edge_inds)
        )


    def __iter__(self):
        self.i = 0
        return self
    def __len__(self):
        return self.epoch_size
    def __next__(self):
        self.i += 1
        if self.i > self.epoch_size:
            raise StopIteration
        if np.random.random() < self.prop:
            example_idx = self.pos[np.random.choice(len(self.pos))]
        else:
            example_idx = self.neg[np.random.choice(len(self.neg))]

        result_example = self[example_idx]
        pairings = result_example['possible_pairings']
        idx = pairings[np.random.choice(len(pairings))]
        if idx == example_idx:
            return result_example['data']
        graph_example = self[idx]

        d = self.combine_pair(
            graph_data=graph_example['data'],
            result_data=result_example['data'],
            invocation=result_example['invocation']
        )
        d.y = torch.tensor([float(result_example['invocation'].label)])
        return d


    def __len__(self):
        return self.num_examples
    
    




if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage error: Pass in a pkl path.')
        sys.exit(1)
    dataset = TrainingDataset(construct_hypermodel_input, HyperModelInfo, augment=True)
    dataset.from_pkl_files(*sys.argv[1:])
    dataset.prepare()
    for x in dataset:
        print(x)