import json
import os
import sys
from typing import AnyStr
from panda_station import (
    RigidTransformWrapper
)

file_path, _ = os.path.split(os.path.realpath(__file__))

def save_stats(
    domain_pddl,
    stream_pddl,
    str_init,
    str_goal,
    stats_path
):
    """
    Saves the file path to the stats.json to 
    index.json, with the key being the unique identifier
    string for that problem 
    """
    if not os.path.isdir(f"{file_path}/data"):
        os.mkdir(f"{file_path}/data")
    index = {}
    exists = os.path.isfile(f"{file_path}/data/index.json")
    if exists:
        with open(f"{file_path}/data/index.json") as stream:
            index = json.load(stream)

    pddl = domain_pddl + stream_pddl
    str_index = str_init + str_goal

    if pddl not in index:
        index[pddl] = {}

    index[pddl][str_index] = stats_path
    with open(f"{file_path}/data/index.json", "w") as stream:
        json.dump(index, stream, indent = 4, sort_keys= True)
    
def get_stats(
    domain_pddl,
    stream_pddl, 
    str_init,
    str_goal,
):
    """
    Gets the stats.json given
    str_index = str_init + str_goal 
    """
    with open(f"{file_path}/data/index.json") as stream:
        index = json.load(stream)
    pddl = domain_pddl + stream_pddl
    str_index = str_init+str_goal
    if pddl not in index:
        raise KeyError(
            ("Oracle does not have information of this"
            "domain.pddl/stream.pddl")
        )
    if str_index not in index[pddl]:
        raise KeyError(
            "Oracle does not have information of this problem.pddl"
        )
    filepath = index[pddl][str_index]
    return filepath

def logical_to_string(logical):
    """
    Turns an init/goal list for a
    pddl problem into a consistent string
    identifier
    (ie. the order of the predicates will not
    matter for the output string)
    """
    logical = sorted(
        logical,
        key = lambda x: str(x)
    )
    res = ""
    num = 0
    for item in logical:
        num += 1
        if isinstance(item, str):
            res += item + ","
            continue
        res += "("
        for i in range(len(item)):
            s = item[i]
            res += str(s)
            if i < len(item) - 1:
                res += ", "
        res += ")"
        if num < len(logical):
            res += ", "
    res += "\n"
    return res
            


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

def ancestors_tuple(fact, atom_map):
    """
    Given a fact, return a pre-order from bottom-up tuple of
    that fact's branch
    """
    parents = atom_map[fact]
    ancestors = tuple()
    for parent in parents:
        ancestors += (parent,)
        ancestors += ancestors_tuple(parent, atom_map)

    return ancestors

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

def apply_substitution(fact, substitution):
    return tuple(substitution.get(arg, arg) for arg in fact)

def is_matching(l, ans_l, preimage, atom_map):
    """
    returns True iff there exists a fact, g, in
    atom_map (global) and a substitution
    S such that S(can_fact) == g and for all 
    g_i in ancestors(f) there exists an li in can_fact
    such that S(l_i) = g_i.

    A subsitution is defined as a mapping from variable to object
    ie. (#o1 -> leg1, #g1 -> [0.1, 0.2, -0.5])
    """
    for g in preimage:
        sub_map = {}
        g = tuple(g)
        if not subsitution(l, g, sub_map):  # facts are of same type
            continue
        
        if not(g in atom_map):
            raise NotImplementedError(f"Something wrong here. {g} is not in atom_map.")
        ans_g = ancestors_tuple(g, atom_map)
        if not ans_g:
            continue # never need to match initial conditions
        if len(ans_g) != len(ans_l):
            continue # avoid cost of unnecessary computation
        all = True
        for g_i in set(ans_g): # forall g_i in ans(g)
            l_exists = False # does there exists and l_i such that
            for l_i in ans_l:
                if subsitution(l_i, g_i, sub_map):
                    l_exists = True
                    break
                # Do we need the l_i, g_i matching to be bijective?
            if not l_exists:
                all = False
                break # go to next g
        if all:
            substituted_ancestors = tuple(apply_substitution(fact, sub_map) for fact in ans_l)
            if substituted_ancestors == ans_g:
                return True, g
            continue
    return False, None


def is_relevant(result, node_from_atom, preimage):
    """
    returns True iff either one of the
    certified facts in result.get_certified()
    is_matching() (see above function)
    """
    can_atom_map = make_atom_map(node_from_atom)
    # assert {x for x in atom_map if not atom_map[x]} == {x for x in can_atom_map if not can_atom_map[x]}
    can_ans = tuple()
    for domain_fact in result.domain:
        can_ans += (fact_to_pddl(domain_fact),)
        can_ans += ancestors_tuple(fact_to_pddl(domain_fact), can_atom_map)
    
    for can_fact in result.get_certified():
        #print(f"candidate fact: {can_fact}")
        #print(f"ancestors: {can_ans}")
        is_match, match = is_matching(fact_to_pddl(can_fact), can_ans, preimage, atom_map)
        if is_match:
            # print(f'Relevant: \n\t {fact_to_pddl(can_fact)}: {can_ans} \n\t {match}: {ancestors(match, atom_map)}')            
            return True, (can_fact, match)
        else:
            pass
            # print('no', fact_to_pddl(can_fact))
    #print("Irrelevant")
    return False, None

def make_is_relevant_checker(remove_matched=True):
    if last_preimage is None or atom_map is None:
        load_stats()
    preimage = last_preimage.copy()
    def unique_is_relevant(result, node_from_atom):
        is_match, match = is_relevant(result, node_from_atom, preimage)
        
        if remove_matched and is_match:
            lifted, grounded = match
            preimage.remove(grounded)
        return is_match
    return unique_is_relevant


last_preimage = None
atom_map = None

def load_stats():
    global last_preimage
    global atom_map
    if os.path.isfile(f"{file_path}/data/stats.json"):
        with open(f"{file_path}/data/stats.json") as stream:
            data = json.load(stream)
            last_preimage = list(map(tuple, data["last_preimage"]))
            atom_map = item_to_dict(data["atom_map"])
            to_add = set()
            for fact in last_preimage:
                if fact not in atom_map:
                    print(f'Warning {fact} not in atom_map')
                    continue
                to_add |= ancestors(fact, atom_map)
            print(len(last_preimage))
            last_preimage += list(to_add)
            print(len(last_preimage))
    else:
        print(
            (f"File {file_path}/data/stats.json does not exist\n"
            "Not using oracle")
        )
