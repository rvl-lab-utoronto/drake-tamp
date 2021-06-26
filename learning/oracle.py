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
    res = set(parents)
    for parent in parents:
        res |= ancestors(parent, atom_map)
    return set(res)

def make_atom_map(node_from_atom):
    atom_map = {}
    for atom in node_from_atom:
        node = node_from_atom[atom]
        result = node.result
        print(node)
        if result is None:
            atom_map[atom] = []
            continue
        atom_map[atom] = result.domain
    return atom_map

def subsitution(l, g, sub_map):
    test_sub_map = sub_map.copy()
    if (l[0] != g[0]) or (len(l) != len(g)):
        return False
    for can_o, o in zip(l[1:], g[1:]) :
        if not (can_o in test_sub_map):
            test_sub_map[can_o] = o
            continue
        if test_sub_map[can_o] != o:
            return False
    sub_map.update(test_sub_map)
    return True

sub_map = {}
def is_matching(l, ans_l):
    """
    returns True iff there exists a fact, g, in
    atom_map (global) and a substitution
    S such that S(can_fact) == g and for all 
    g_i in ancestors(f) there exists an li in can_fact
    such that S(l_i) = g_i.

    A subsitution is defined as a mapping from variable to object
    ie. (#o1 -> leg1, #g1 -> [0.1, 0.2, -0.5])
    """

    for g in last_preimage:
        if not subsitution(l, g, sub_map): # facts are of same type
            continue
        ans_g = ancestors(g, atom_map)
        all = True
        for g_i in ans_g: # forall g_i in ans(g)
            l_exists = False # does there exists and l_i such that
            for l_i in ans_l:
                if subsitution(l_i, g_i, sub_map):
                    l_exists = True
                    break
            if not l_exists:
                all = False
                break # go to next g
        if all:
            return True
    return False


def is_relevant(result, node_from_atom):
    """
    returns True iff either one of the
    certified facts in result.get_certified()
    is_matching() (see above function)
    """
    can_atom_map = make_atom_map(node_from_atom)
    can_ans = set()
    for domain_fact in result.domain:
        can_ans.add(domain_fact)
        can_ans |= ancestors(domain_fact, can_atom_map)
    for can_fact in result.get_certified():
        print(f"candidate fact: {can_fact}")
        print(f"ancestors: {can_ans}")
        if is_matching(can_fact, can_ans):
            print("Relevant")
            return True
    print("Irrelevant")
    return False