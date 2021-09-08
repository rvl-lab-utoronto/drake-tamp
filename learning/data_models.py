from collections import namedtuple

from numpy.lib.arraysetops import isin
from learning.pddlstream_utils import make_atom_map, make_stream_map, fact_to_pddl, obj_to_pddl
from dataclasses import dataclass
from torch_geometric.data import Data
from pddlstream.algorithms.downward import Domain
import numpy as np

from pddlstream.language.constants import Evaluation

@dataclass
class ModelInfo:
    """This class is intended to keep all the information that should remain constant for a single model"""
    predicates: list
    streams: list
    stream_num_domain_facts: list
    stream_num_inputs: list
    stream_domains: list
    domain: Domain
    stream_num_outputs: list = None

    #TODO(agro)
    #@property
    #def stream_num_domain_facts(self):

    @property
    def stream_to_index(self):
        return {s:i for i,s in enumerate(self.streams)}
    @property
    def predicate_to_index(self):
        return {s:i for i,s in enumerate(self.predicates)}

    @property
    def object_node_feature_size(self):
        return len(self.predicates)

    @property
    def action_to_index(self):
        return {a.name:i for i, a in enumerate(self.domain.actions)}
    
    @property
    def num_streams(self):
        return len(self.streams)

    @property
    def num_predicates(self):
        return len(self.predicates)

    @property
    def num_actions(self):
        return len(self.domain.actions)

    @property
    def predicate_num_args(self):
        """Returns a list of the lengths of the predicate arguments"""
        return [len(predicate.arguments) for predicate in self.domain.predicates]

    @property
    def max_predicate_num_args(self):
        return max(self.predicate_num_args)

class StreamInstanceClassifierInfo(ModelInfo):

    @property
    def node_feature_size(self):
        return self.num_predicates + 2

    @property
    def edge_feature_size(self):
        return self.num_streams + 3

class HyperModelInfo(ModelInfo):

    @property
    def node_feature_size(self):
        return self.num_streams + 2

    @property
    def edge_feature_size(self):
        return self.num_predicates + self.num_actions * 2 + self.num_streams + 2  + 2 * self.max_predicate_num_args
    
    @property
    def problem_graph_node_feature_size(self):
        return 3 + 1 #xyz + a boolean
    
    @property
    def problem_graph_edge_feature_size(self):
        return self.num_predicates + 1

class StreamInstanceClassifierV2Info(ModelInfo):

    @property
    def problem_graph_node_feature_size(self):
        return 3 + 1 #xyz + a boolean
    
    @property
    def problem_graph_edge_feature_size(self):
        return self.num_predicates + 1



@dataclass
class ProblemInfo:
    goal_facts: list
    initial_facts: list
    model_poses: list
    problem_graph: Data = None
    object_mapping: dict = None

    def __eq__(self, other):
        if self.goal_facts != other.goal_facts:
            return False
        if self.initial_facts != other.initial_facts:
            return False
        if len(self.model_poses) != len(other.model_poses):
            return False
        for p1, p2 in zip(self.model_poses, other.model_poses):
            if p1["name"] != p2["name"]:
                return False
            if not np.all(p1["X"].GetAsMatrix34() == p2["X"].GetAsMatrix34()):
                return False
            if p1["static"] != p2["static"]:
                return False
        return True

    def __hash__(self):
        res = self.goal_facts
        res += self.initial_facts
        for pose in self.model_poses:
            res += (pose["name"], )
            res += (str(pose["X"]), )
            res += (pose["static"], )
        return hash(res)


# TODO: Do these shared classes need a new home?
SerializedResult = namedtuple(
    'SerializedResult', ["name", "certified", "domain", "input_objects", "output_objects"]
)
class InvocationInfo:
    def __init__(self, result, node_from_atom, label=None, atom_map = None, object_stream_map=None):
        if atom_map is None:
            self.atom_map = make_atom_map(node_from_atom) 
        else:
            self.atom_map = atom_map
        # self.stream_map = make_stream_map(node_from_atom)
        if object_stream_map is None:
            self.object_stream_map = self.make_obj_to_stream_map(node_from_atom)
        else:
            self.object_stream_map = object_stream_map
        self.result = self.result_to_serializable(result)
        self.label = label

    @staticmethod
    def result_to_serializable(result):
        return SerializedResult(
            name=result.name,
            certified=tuple([fact_to_pddl(f) for f in result.certified]),
            domain=tuple([fact_to_pddl(f) for f in result.domain]),
            input_objects=tuple([obj_to_pddl(f) for f in result.input_objects]),
            output_objects=tuple([obj_to_pddl(f) for f in result.output_objects]),
        )

    @staticmethod
    def make_obj_to_stream_map(node_from_atom):
        """
        return a mapping from pddl object to the
        stream that created it 
        """
        obj_to_stream_map = {}
        for atom in node_from_atom:
            node = node_from_atom[atom]
            result = node.result
            if result is None or isinstance(result, bool):
                continue
            for r in result.output_objects:
                obj_to_stream_map[obj_to_pddl(r)] = {
                    "name": result.name, 
                    "input_objects": [obj_to_pddl(f) for f in result.input_objects],
                    "output_objects": [obj_to_pddl(f) for f in result.output_objects]
                }
        return obj_to_stream_map

class RuntimeInvocationInfo(InvocationInfo):

    def __init__(self, result, atom_map, stream_map, obj_to_stream_map):
        self.atom_map = atom_map
        self.stream_map = stream_map
        self.object_stream_map = obj_to_stream_map
        self.result = self.result_to_serializable(result)
