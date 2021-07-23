#!/usr/bin/env python3
import json
from datetime import datetime
import shutil
from multiprocessing import Pool
from functools import partial
import numpy as np
import os
import itertools
import pickle
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from learning.oracle import item_to_dict, ancestors, ancestors_tuple, is_matching
from learning.gnn.data import get_base_datapath, query_data

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
    return is_match

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

def same_problem(problem_info1, problem_info2):
    """
    Takes two problem infos and returns True iff they corrispond to
    the same problem.
    """
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

def get_all_datas():
    datapath = get_base_datapath()
    data_info_path = os.path.join(datapath, "data_info.json")
    assert os.path.isfile(data_info_path), f"{data_info_path} does not exist yet"
    with open(data_info_path, "r") as f:
        info = json.load(f)
    pkl_names = []
    groups = {}
    for lst in info.values():
        for item in lst:
            pkl = item[1]
            file = os.path.join(datapath, pkl)
            if "merged" in file:
                continue
            pkl_names.append(pkl)
            with open(file, "rb") as f:
                data = pickle.load(f)
            data["dir"] = os.path.splitext(file)[0]
            groups.setdefault(data["problem_info"], []).append(data)
    return pkl_names,groups 

def load_invs(data):
    invs = []
    for i in range(data["num_labels"]):
        path = os.path.join(
            data["dir"], f"label_{i}.pkl"
        )
        with open(path, "rb") as f:
            invs.append(pickle.load(f))
    return invs

def consensus(inv, can_pkl, pkl_to_data):
    for pkl, data in pkl_to_data.items():
        if pkl == can_pkl:
            continue
        rel = is_inv_relevant(inv, data[0], data[1])
        if rel:
            return True
    return False

def conditional_consensus(inv, pkl, pkl_to_stats):
    if not inv.label:
        inv.label = consensus(inv, pkl, pkl_to_stats)

def merge_labels(group):
    if len(group) <= 1:
        return
    pkl_to_data = {}
    pkl_to_stats = {}
    for data in group:
        # (last_preimage, atom_map)
        pkl_to_stats[data["dir"]] = load_stats(data["stats_path"]) 
        pkl_to_data[data["dir"]] = data
    original = {"pos": 0, "neg" : 0}
    after = {"pos": 0, "neg" : 0}
    #merged = []
    newname = "merged" + "_" + datetime.utcnow().strftime("%Y-%m-%d-%H:%M:%S.%f")[:-3] 
    newdir = os.path.join(get_base_datapath(), newname)


    os.mkdir(newdir)
    num_labels = 0
    for pkl in pkl_to_data.keys():
        invs = load_invs(pkl_to_data[pkl])
        for inv in tqdm(invs):
            original["pos"] += int(inv.label)
            original["neg"] += int(not inv.label)
            conditional_consensus(inv, pkl, pkl_to_stats)
            after["pos"] += int(inv.label)
            after["neg"] += int(not inv.label)

            name = f"label_{num_labels}.pkl"
            with open(os.path.join(newdir, name), "wb") as stream:
                pickle.dump(inv, stream)
            num_labels +=1
        
        #merged += invs
    tot = sum(list(original.values()))
    dp = (after["pos"] - original["pos"])/tot
    dn = (after["neg"] - original["neg"])/tot
    print(f"Change in proportion positive: {dp:.5f}. Change in proportion negative {dn:.5f}")

    # write new high level data file
    newdata = {}
    newdata["problem_info"] = group[0]["problem_info"]
    newdata["model_info"] = group[0]["model_info"]
    newdata["stats_paths"] = [
        d["stats_path"] for d in group
    ]
    newdata["merged"] = True
    newdata["domain_pddl"] = group[0]["domain_pddl"]
    newdata["stream_pddl"] = group[0]["stream_pddl"]
    newdata["num_labels"] = num_labels
    newdata["dir"] = newdir
    with open(newdir + ".pkl", "wb") as stream:
        pickle.dump(newdata, stream)

    # update data_info.json

    with open(os.path.join(get_base_datapath(), "data_info.json")) as stream:
        data_info = json.load(stream)
    pddl = newdata["domain_pddl"] + newdata["stream_pddl"]
    names = [os.path.split(d["dir"])[1] + ".pkl" for d in group]
    i = 0
    sum_comp = 0
    sum_evals = 0
    sum_iters = 0
    sum_run = 0
    sum_sample = 0
    sum_search = 0
    rem = False
    while i < len(data_info[pddl]):
        info = data_info[pddl][i]
        if info[1] in names:
            rem = True
            stats = data_info[pddl].pop(i)[0]
            sum_comp += stats["complexity"]
            sum_evals += stats["evaluations"]
            sum_iters += stats["iterations"]
            sum_run += stats["run_time"]
            sum_sample += stats["sample_time"]
            sum_search += stats["search_time"]
            continue
        i += 1
    if not rem: # must have already merged these files
        return
    length = len(group)
    stats["complexity"] = sum_comp/length
    stats["evaluations"] = sum_evals/length
    stats["iterations"] = sum_iters/length
    stats["run_time"] = sum_run/length
    stats["sample_time"] = sum_sample/length
    stats["search_time"] = sum_search/length
    data_info[pddl].append(
        [
            stats,
            newname + ".pkl",
            num_labels
        ]
    )
    with open(os.path.join(get_base_datapath(), "data_info.json"), "w") as stream:
        json.dump(data_info, stream, indent = 4, sort_keys = True)

    # delete old pickles

    for dir in pkl_to_data:
        os.remove(dir + ".pkl")
        shutil.rmtree(
            dir
        )

def merge_groups(groups):
    num_groups = len(groups.values())
    for i,group in enumerate(list(groups.values())):
        print(f"Processing group number {i+1} our of {num_groups}")
        if len(group) <= 1:
            continue
        print(f"Group size: {len(group)}")
        merge_labels(group)

if __name__ == "__main__":
    """
    1. iterate over top level pickles and group based on problem
    2. run merge_labels on all pickles in each group to get merged label
    3. For each merged group, delete all data and save new top-level pickle and 
       all of its labels
    """
    names, groups = get_all_datas()
    print("merging")
    merge_groups(groups)
    #for k,v in groups.items():
        #if len(v) > 1:
            #print(len(v))
    pass