import json
import os
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
        if result is None:
            atom_map[fact_to_pddl(atom)] = []
            continue
        atom_map[fact_to_pddl(atom)] = [fact_to_pddl(f) for f in result.domain]
    return atom_map

def subsitution(l, g, sub_map):
    test_sub_map = sub_map.copy()
    if (l[0] != g[0]) or (len(l) != len(g)):
        return False
    for can_o, o in zip(l[1:], g[1:]):
        if not can_o.startswith('#'):
            if can_o != o:
                return False
            continue
        if not (can_o in test_sub_map):
            assert can_o.startswith('#'), "Expected substitution keys to only include variables"
            test_sub_map[can_o] = o
            continue
        if test_sub_map[can_o] != o:
            return False
    sub_map.update(test_sub_map)
    return True

from pddlstream.language.object import Object, OptimisticObject
def fact_to_pddl(fact):
    new_fact = [fact[0]]
    for obj in fact[1:]:
        new_fact.append(obj.pddl if isinstance(obj, Object) or isinstance(obj, OptimisticObject) else obj)
    return tuple(new_fact)
    
def is_matching(l, ans_l, preimage, atom_map, return_all=False):
    """
    returns True iff there exists a fact, g, in
    atom_map (global) and a substitution
    S such that S(can_fact) == g and for all 
    g_i in ancestors(f) there exists an li in can_fact
    such that S(l_i) = g_i.

    A subsitution is defined as a mapping from variable to object
    ie. (#o1 -> leg1, #g1 -> [0.1, 0.2, -0.5])
    """
    result = []
    for g in preimage:
        sub_map = {}
        g = tuple(g)
        if not subsitution(l, g, sub_map):  # facts are of same type
            continue
        
        if not(g in atom_map):
            raise NotImplementedError("Something wrong here.")
        ans_g = ancestors(g, atom_map)
        all = True
        for g_i in ans_g: # forall g_i in ans(g)
            l_exists = False # does there exists and l_i such that
            for l_i in ans_l:
                if subsitution(l_i, g_i, sub_map):
                    l_exists = True
                    break
                # Do we need the l_i, g_i matching to be bijective?
            if not l_exists:
                all = False
                break # go to next g
        if not return_all and all:
            return True, g
        elif return_all:
            result.append(g)
    if return_all:
        return len(result) > 0, result
    return False, None


def is_relevant(result, node_from_atom):
    """
    returns True iff either one of the
    certified facts in result.get_certified()
    is_matching() (see above function)
    """
    can_atom_map = make_atom_map(node_from_atom)
    can_ans = set()
    for domain_fact in result.domain:
        can_ans.add(fact_to_pddl(domain_fact))
        can_ans |= ancestors(fact_to_pddl(domain_fact), can_atom_map)
    for can_fact in result.get_certified():
        #print(f"candidate fact: {can_fact}")
        #print(f"ancestors: {can_ans}")
        is_match, match = is_matching(fact_to_pddl(can_fact), can_ans, last_preimage, atom_map)
        if is_match:
            # print('yes', fact_to_pddl(can_fact), match)
            return True
        else:
            pass
            # print('no', fact_to_pddl(can_fact))
    #print("Irrelevant")
    return False


file_path, _ = os.path.split(os.path.realpath(__file__))
last_preimage = None
atom_map = None
with open(f"{file_path}/data/stats.json") as stream:
    data = json.load(stream)
    last_preimage = data["last_preimage"]
    atom_map = item_to_dict(data["atom_map"])
    to_add = set()
    for fact in last_preimage:
        fact = tuple(fact)
        if fact not in atom_map:
            print(f'Warning {fact} not in atom_map')
            continue
        to_add |= ancestors(fact, atom_map)
    print(len(last_preimage))
    last_preimage += list(to_add)
    print(len(last_preimage))
