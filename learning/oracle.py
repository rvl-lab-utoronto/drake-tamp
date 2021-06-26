import json
import os
from typing import AnyStr
from panda_station import (
    RigidTransformWrapper
)
file_path, _ = os.path.split(os.path.realpath(__file__))

def item_to_dict(atom_map):
    res = {}
    for item in atom_map:
        key = tuple(item[0]) 
        value = [tuple(i) for i in item[1]]
        res[key] = value
    return res

last_preimage = None
atom_map = None
with open(f"{file_path}/data/stats.json") as stream:
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
    returns True iff there exists a fact, f, in
    atom_map (global) and a substitution
    S such that S(can_fact) == f and for all 
    f_i in ancestors(f) there exists an li in can_fact
    such that S(l_i) = f_i.

    A subsitution is defined as a mapping from variable to object
    ie. (#o1 -> leg1, #g1 -> [0.1, 0.2, -0.5])
    """
    pass

def is_relevant(result, node_from_atom):
    """
    returns True iff either one of the
    certified facts in result.get_certified()
    is_matching() (see above function)
    """
    can_atom_map = make_atom_map(node_from_atom)
    can_ans = set()
    for domain_fact in result.domain:
        can_ans |= ancestors(domain_fact, can_atom_map)
    for can_fact in result.get_certified():
        if is_matching(can_fact, can_ans):
            return True
    return False