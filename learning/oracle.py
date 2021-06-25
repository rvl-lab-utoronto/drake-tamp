import json
from typing import AnyStr
from panda_station import (
    RigidTransformWrapper
)

def item_to_dict(atom_map):
    res = {}
    for item in atom_map:
        key = tuple(item[0]) 
        value = [tuple(i) for i in item[1]]
        res[key] = value
    return res

last_preimage = None
atom_map = None
with open("learning/data/stats.json") as stream:
    data = json.load(stream)
    last_preimage = data["last_preimage"]
    atom_map = item_to_dict(data["atom_map"])


def ancestors(fact, atom_map):
    """
    Given a fact, return a set of
    that fact's ancestors
    """
    parents = atom_map[fact]
    res = set(parents.copy())
    for parent in parents:
        res |= ancestors(parent, atom_map)
    return set(res)

def make_atom_map(node_from_atom):
    atom_map = {}
    for atom in node_from_atom:
        node = node_from_atom[atom]
        result = node.result
        if result is None:
            atom_map[atom] = []
            continue
        atom_map[atom] = result.domain
    return atom_map

def is_matching(can_fact, can_ans):
    """
    returns True iff 
    TODO(agro) 
    ... Insert definition here
    """
    pass

def is_relevant(result, node_from_atom):
    """
    returns True iff 
    TODO(agro) 
    ... Insert definition here
    """
    atom_map = make_atom_map(node_from_atom)
    can_ans = set()
    for domain_fact in result.domain:
        can_ans |= ancestors(domain_fact)
    for can_fact in result.get_certified():
        if is_matching(can_fact, can_ans):
            return True
    return False