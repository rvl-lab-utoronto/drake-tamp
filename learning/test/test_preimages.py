#%%

import json
import numpy as np
import os
import itertools
import pickle

#%%

# interesting runs: 
p0 = "/home/agrobenj/drake-tamp/experiments/kitchen_no_fluents/logs/2021-07-20-11:12:12" #<- this one has the problem.yaml
p1 = "/home/agrobenj/drake-tamp/experiments/kitchen_no_fluents/logs/2021-07-20-11:12:29"
#p2 = "/home/agrobenj/drake-tamp/experiments/kitchen_no_fluents/logs/2021-07-18-23:21:48"

paths = [p0, p1]#, p2]

from learning.oracle import item_to_dict, ancestors, ancestors_tuple, is_matching

def load_stats(fullpath):
    with open(fullpath, "r") as stream:
        data = json.load(stream)
    last_preimage = list(map(tuple, data["last_preimage"]))
    atom_map = item_to_dict(data["atom_map"])
    to_add = set()
    for fact in last_preimage:
        if fact not in atom_map:
            continue
        to_add |= ancestors(fact, atom_map)
    last_preimage += list(to_add)
    return last_preimage, atom_map
preimages, atom_maps = [],[]
for p in paths:
    last_preimage, atom_map = load_stats(os.path.join(p, "stats.json"))
    preimages.append(last_preimage)
    atom_maps.append(atom_map)

# we want to find which facts in the preimage/atom_map of one run would the other find irrelevant


def is_relevant(can_fact, can_fact_atom_map, ground_truth_preimage, ground_truth_atom_map):
    """
    Check if can_fact is relevant wrt ground_truth_preimage 
    """
    can_ans = ancestors_tuple(can_fact, can_fact_atom_map)
    ground_truth_init = {x for x in ground_truth_atom_map if not ground_truth_atom_map[x]}
    if can_fact in ground_truth_init:
        return True, can_fact

    return is_matching(
        can_fact,
        can_ans,
        ground_truth_preimage,
        ground_truth_atom_map,
        ground_truth_init
    )

VERBOSE = False

def printv(s):
    if VERBOSE:
        print(s)

for can_ind, gt_ind in itertools.permutations(range(len(paths)), r = 2):
    can_preimage = preimages[0]
    can_atom_map = atom_maps[0]
    gt_preimage = preimages[gt_ind]
    gt_atom_map = atom_maps[gt_ind]

    print(f"Candidate: {paths[can_ind]}. Preimage size: {len(preimages[can_ind])}")
    print(f"Ground Truth: {paths[gt_ind]}. Preimage size: {len(preimages[gt_ind])}.")
    can_num_relevant = 0
    for can_fact in can_preimage:
        rel, match = is_relevant(can_fact, can_atom_map, gt_preimage, gt_atom_map)
        if rel:
            printv(f"Relevant: {can_fact, ancestors_tuple(can_fact, can_atom_map)}")
            printv(f"Match: {match, ancestors_tuple(match, gt_atom_map)}")
        else:
            printv(f"Irrelevant: {can_fact, ancestors(can_fact, can_atom_map)}")
        can_num_relevant += int(rel)
    print(f"Fraction relevant: {can_num_relevant/len(can_preimage)}")

l0 = "/home/agrobenj/drake-tamp/learning/data/labeled/2021-07-20-11:12:13.912.pkl"
l1 = "/home/agrobenj/drake-tamp/learning/data/labeled/2021-07-20-11:12:30.953.pkl"
#l2 = "/home/agrobenj/drake-tamp/learning/data/labeled/2021-07-20-11:12:45.932.pkl"
label_paths = [l0, l1]#, l2]

def load_pkl(fullpath):
    with open(fullpath, "rb") as f:
        data = pickle.load(f)
    folder, _ = os.path.splitext(fullpath)
    print(f"Loading: {folder}")
    invs = []
    for i in range(data["num_labels"]):
        name = f"label_{i}.pkl"
        with open(f"{os.path.join(folder, name)}" , "rb") as f:
            inv = pickle.load(f)
            invs.append(inv)

    last_preimage, atom_map = load_stats(data["stats_path"])
    return invs, {"last_preimage": last_preimage, "atom_map": atom_map}


def is_inv_relevant(inv, ground_truth_preimage, ground_truth_atom_map):
    can_facts = inv.result.certified
    domain_facts = inv.result.domain
    can_ans = tuple()
    for domain_fact in domain_facts:
        can_ans += (domain_fact, )
        can_ans += ancestors_tuple(domain_fact, inv.atom_map)
    for can_fact in can_facts:
        ground_truth_init = {x for x in ground_truth_atom_map if not ground_truth_atom_map[x]}
        is_match, match = is_matching(
            can_fact,
            can_ans,
            ground_truth_preimage,
            ground_truth_atom_map,
            ground_truth_init
        )
        if is_match:
            break
    return is_match, match, inv.label

print()

"""
invs = []
for l in label_paths:
    invs.append(load_pkl(l))

for can_ind, gt_ind in itertools.permutations((0,1,2), r = 2):
    gt_pre = preimages[gt_ind]
    gt_atom = atom_maps[gt_ind]
    num_irr = 0
    num_irr_ex = 0
    num_rel = 0
    num_rel_inc = 0
    for inv in invs[can_ind]:
        is_match, match, label = is_inv_relevant(inv, gt_pre, gt_atom)
        if not label:
            num_irr += 1
            if not is_match:
                num_irr_ex += 1
        if label:
            num_rel += 1
            if is_match:
                num_rel_inc += 1
        #if is_match != label:
            #print(is_match, label)
    print(f"Prop irrelevant excluded: {num_irr_ex/num_irr}")
    print(f"Prop relevant included: {num_rel_inc/num_rel}")
    print()

"""

def rel_to_others(inv, can_pkl, pkl_to_data):
    for pkl, data in pkl_to_data.items():
        if pkl == can_pkl:
            continue
        rel = is_inv_relevant(inv, data["last_preimage"], data["atom_map"])[0]
        if rel:
            return True
    return False

def merge_labels(pkl_list):
    invs_list = []

    pkl_to_data = {}
    for pkl in pkl_list:
        invs, stats = load_pkl(pkl)
        invs_list.append(invs)
        pkl_to_data[pkl] = stats

    merged = []
    

    original = {"pos": 0, "neg" : 0}
    after = {"pos": 0, "neg" : 0}
    #num_become_positive = 0
    for pkl, invs in zip(pkl_list ,invs_list):
        for inv in invs:
            original["pos"] += int(inv.label)
            original["neg"] += int(not inv.label)
            if not inv.label:
                inv.label = rel_to_others(inv, pkl, pkl_to_data) # do any of the other oracles think this invocation is relevant?
            after["pos"] += int(inv.label)
            after["neg"] += int(not inv.label)
            merged.append(inv)
    tot = sum(list(original.values()))
    print("Before consensus:")
    pos, neg = original["pos"], original["neg"]
    print(f"Fraction positive: {pos/tot}. Fraction negative: {neg/tot}")
    print("After consensus:")
    pos, neg = after["pos"], after["neg"]
    print(f"Fraction positive: {pos/tot}. Fraction negative: {neg/tot}")

    return merged

merged = merge_labels(label_paths)

# %%
        

l1 = "/home/agrobenj/drake-tamp/learning/data/labeled/2021-07-20-11:05:16.964.pkl"
l2 = "/home/agrobenj/drake-tamp/learning/data/labeled/2021-07-20-11:05:36.588.pkl"
l3 = "/home/agrobenj/drake-tamp/learning/data/labeled/2021-07-20-11:06:47.634.pkl"
d1 = pickle.load(open(l1, "rb"))
d2 = pickle.load(open(l2, "rb"))
d3 = pickle.load(open(l3, "rb"))

#%%

def same_problem(problem_info1, problem_info2):
    if problem_info1.goal_facts != problem_info2.goal_facts:
        return False
    if problem_info1.initial_facts != problem_info2.initial_facts:
        return False
    if len(problem_info1.model_poses) != len(problem_info2.model_poses):
        return False
    for p1, p2 in zip(problem_info1.model_poses, problem_info2.model_poses):
        if p1["name"] != p2["name"]:
            return False
        if not np.all(p1["X"].GetAsMatrix34() == p2["X"].GetAsMatrix34()):
            return False
        if p1["static"] != p2["static"]:
            return False
    return True
    
#%%


r = d1["problem_info"] == d2["problem_info"]
print(r)
print(d1["problem_info"] == d3["problem_info"])
print(list(d1.keys()))
