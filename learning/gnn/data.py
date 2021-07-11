from learning.data_models import HyperModelInfo, InvocationInfo, ProblemInfo, ModelInfo
import numpy as np
from learning.pddlstream_utils import objects_from_facts
import itertools
import torch
from torch_geometric.data import Data
import pickle



def construct_object_hypergraph(
    model_info: ModelInfo, problem_info: ProblemInfo, label: InvocationInfo
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
                }
            )
        else:
            for e_objs in itertools.combinations(fact_objects, 2):
                edges.append(
                    (node_to_ind[e_objs[0]], node_to_ind[e_objs[1]])
                )
                edge_attr.append(
                    {
                        "predicate": fact[0],
                        "overlap_with_goal": fact_objects.intersection(goal_objects),
                        "level": obj_level(fact),
                        "stream": label.stream_map[fact],
                        "relevant_actions": fact_to_relevant_actions(
                            fact, model_info.domain, indices = False
                        )
                    }
                )
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
            "full_fact": fact # not used as an attribute, but for indexing
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
                    "full_fact": fact # not used as an attribute, but for indexing
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
    model_info: ModelInfo, problem_info: ProblemInfo, label: InvocationInfo
):
    nodes, node_attr, edges, edge_attr = construct_object_hypergraph(
        model_info,
        problem_info,
        label
    )

    model_info = HyperModelInfo(
        predicates = model_info.predicates,
        streams = model_info.streams,
        stream_num_domain_facts=model_info.stream_num_domain_facts,
        domain = model_info.domain
    )

    # indices of input objects to latest stream result
    input_node_inds = [nodes.index(p) for p in label.result.input_objects]
    # indices of domain facts of latest stream result
    dom_edge_inds = []
    for i, attr in enumerate(edge_attr):
        if attr["full_fact"] in label.result.domain:
            dom_edge_inds.append(i)
    node_features = torch.zeros(
        (len(nodes), model_info.node_feature_size), dtype=torch.float
    )
    edge_features = torch.zeros(
        (len(edges), model_info.edge_feature_size), dtype=torch.float
    )
    for i, attr in enumerate(node_attr):
        node_features[i, model_info.stream_to_index[attr["stream"]]] = 1
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


def parse_labels(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    model_info = data['model_info']
    problem_info = data['problem_info']

    dataset = []
    for invocation_info in data['labels']: # label is now an instance of learning.gnn.oracle.Data
        d = construct_input(invocation_info, problem_info, model_info)
        d.y = torch.tensor([float(invocation_info.label)])
        dataset.append(d)
    return dataset, model_info

def parse_hyper_labels(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    model_info = data['model_info']
    problem_info = data['problem_info']

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

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage error: Pass in a pkl path.')
        sys.exit(1)
    dataset = parse_labels(sys.argv[1])
