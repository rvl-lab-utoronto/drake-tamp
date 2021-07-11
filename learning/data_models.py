from collections import namedtuple
from learning.pddlstream_utils import make_atom_map, make_stream_map, fact_to_pddl, obj_to_pddl
from dataclasses import dataclass
from pddlstream.algorithms.downward import Domain

@dataclass
class ModelInfo:
    """This class is intended to keep all the information that should remain constant for a single model"""
    predicates: list
    streams: list
    stream_num_domain_facts: list
    domain: Domain

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

    @property
    def object_node_feature_size(self):
        return len(self.predicates)


@dataclass
class ProblemInfo:
    goal_facts: list

# TODO: Do these shared classes need a new home?
SerializedResult = namedtuple('SerializedResult', ["name", "certified", "domain", "input_objects", "output_objects"])
class InvocationInfo:
    def __init__(self, result, node_from_atom, label=None):
        self.atom_map = make_atom_map(node_from_atom)
        self.stream_map = make_stream_map(node_from_atom)
        self.object_stream_map = self.make_obj_to_stream_map(node_from_atom)
        self.result = self.result_to_serializable(result)
        self.label = label

    @staticmethod
    def result_to_serializable(result):
        return SerializedResult(
            name=result.name,
            certified=[fact_to_pddl(f) for f in result.certified],
            domain=[fact_to_pddl(f) for f in result.domain],
            input_objects=[obj_to_pddl(f) for f in result.input_objects],
            output_objects=[obj_to_pddl(f) for f in result.output_objects],
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
                    "input_objects": [obj_to_pddl(f) for f in result.input_objects]
                }
        return obj_to_stream_map
