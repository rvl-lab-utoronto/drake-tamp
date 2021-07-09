import json
import os
import sys
import pickle
from typing import AnyStr

import torch
from panda_station import RigidTransformWrapper
from datetime import datetime
from pddlstream.language.object import Object, OptimisticObject

FILEPATH, _ = os.path.split(os.path.realpath(__file__))


def logical_to_string(logical):
    """
    Turns an init/goal list for a
    pddl problem into a consistent string
    identifier
    (ie. the order of the predicates will not
    matter for the output string)
    """
    logical = sorted(logical, key=lambda x: str(x))
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
        # TODO: Figure out how to deal with these bools?
        if result is None:
            atom_map[fact_to_pddl(atom)] = []
            continue
        if isinstance(result, bool):
            continue
        atom_map[fact_to_pddl(atom)] = [fact_to_pddl(f) for f in result.domain]
    return atom_map


def fact_to_pddl(fact):
    new_fact = [fact[0]]
    for obj in fact[1:]:
        pddl_obj = obj_to_pddl(obj)
        new_fact.append(pddl_obj)
    return tuple(new_fact)


def obj_to_pddl(obj):
    pddl_obj = (
        obj.pddl
        if isinstance(obj, Object) or isinstance(obj, OptimisticObject)
        else obj
    )
    return pddl_obj


def apply_substitution(fact, substitution):
    return tuple(substitution.get(arg, arg) for arg in fact)


def sub_map_from_init(init):
    objects = objects_from_facts(init)
    return {o: o for o in objects}


def objects_from_facts(init):
    objects = set()
    for fact in init:
        for arg in fact[1:]:
            objects.add(arg)
    return objects


def is_matching(l, ans_l, preimage, atom_map, init):
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
        sub_map = sub_map_from_init(init)
        g = tuple(g)
        if not subsitution(l, g, sub_map):  # facts are of same type
            continue

        if not (g in atom_map):
            raise NotImplementedError(f"Something wrong here. {g} is not in atom_map.")
        ans_g = ancestors_tuple(g, atom_map)
        if not ans_g:
            continue  # never need to match initial conditions
        if len(ans_g) != len(ans_l):
            continue  # avoid cost of unnecessary computation
        all = True
        for g_i, l_i in zip(ans_g, ans_l):  # forall g_i in ans(g)
            if not subsitution(l_i, g_i, sub_map):
                all = False
                break
        if all:
            return True, g
    return False, None


def subsitution(l, g, sub_map):
    test_sub_map = sub_map.copy()
    if (l[0] != g[0]) or (len(l) != len(g)):
        return False
    for can_o, o in zip(l[1:], g[1:]):
        if not (can_o in test_sub_map):
            test_sub_map[can_o] = o
            continue
        if test_sub_map[can_o] != o:
            return False
    sub_map.update(test_sub_map)
    return True


class Oracle:
    def __init__(self, domain_pddl, stream_pddl, initial_conditions, goal_conditions):
        self.domain_pddl = domain_pddl
        self.stream_pddl = stream_pddl
        self.initial_conditions = initial_conditions
        self.goal_conditions = goal_conditions
        self.str_init = logical_to_string(initial_conditions)
        self.str_goal = logical_to_string(goal_conditions)
        self.stats_path = None
        # self.labeled_path = labeled_path
        self.save_path = (
            FILEPATH
            + "/data/labeled/"
            + datetime.utcnow().strftime("%Y-%m-%d-%H:%M:%S.%f")[:-3]
            + ".pkl"
        )
        self.last_preimage = None
        self.atom_map = None
        self.init = None
        self.labels = []
        self.goal_facts = []
        self.domain = None
        self.externals = None

    def set_goal_facts(self, goal_exp):
        if not goal_exp[0] == "and":
            raise NotImplementedError(
                f"Expected goal to be a conjunction of facts. Got {goal_exp}.Need to parse this correctly."
            )
        self.goal_facts = tuple([fact_to_pddl(f) for f in goal_exp[1:]])

    def set_domain(self, domain):
        self.domain = domain

    def set_externals(self, externals):
        self.externals = []
        for external in externals:
            self.externals.append(
                {
                    "certified": external.certified,
                    "domain": external.domain,
                    "fluents": external.fluents,
                    "inputs": external.inputs,
                    "outputs": external.outputs,
                    "name": external.name,
                }
            )

    def save_labeled(self, path=None):
        if path is None:
            if not os.path.isdir(f"{FILEPATH}/data"):
                os.mkdir(f"{FILEPATH}/data")
            if not os.path.isdir(f"{FILEPATH}/data/labeled"):
                os.mkdir(f"{FILEPATH}/data/labeled")
            path = self.save_path
        data = {}
        data["stats_path"] = self.get_stats()
        data["domain_pddl"] = self.domain_pddl
        data["stream_pddl"] = self.stream_pddl
        data["initial_conditions"] = tuple(self.initial_conditions)
        data["goal_conditions"] = tuple(self.goal_conditions)
        data["goal_facts"] = tuple(self.goal_facts)
        data["labels"] = self.labels
        data["domain"] = self.domain
        data["externals"] = self.externals
        with open(path, "wb") as stream:
            pickle.dump(data, stream)

    def save_stats(self, stats_path):
        """
        Saves the file path to the stats.json to
        index.json, with the key being the unique identifier
        string for that problem
        """
        if not os.path.isdir(f"{FILEPATH}/data"):
            os.mkdir(f"{FILEPATH}/data")
        index = {}
        exists = os.path.isfile(f"{FILEPATH}/data/index.json")
        if exists:
            with open(f"{FILEPATH}/data/index.json") as stream:
                index = json.load(stream)

        pddl = self.domain_pddl + self.stream_pddl
        str_index = self.str_init + self.str_goal

        if pddl not in index:
            index[pddl] = {}

        index[pddl][str_index] = stats_path
        with open(f"{FILEPATH}/data/index.json", "w") as stream:
            json.dump(index, stream, indent=4, sort_keys=True)

    def get_stats(self):
        """
        Gets the stats.json given
        str_index = str_init + str_goal
        """
        if self.stats_path is not None:
            return self.stats_path

        with open(f"{FILEPATH}/data/index.json") as stream:
            index = json.load(stream)
        pddl = self.domain_pddl + self.stream_pddl
        str_index = self.str_init + self.str_goal
        if pddl not in index:
            raise KeyError(
                ("Oracle does not have information of this" "domain.pddl/stream.pddl")
            )
        if str_index not in index[pddl]:
            raise KeyError("Oracle does not have information of this problem.pddl")
        self.stats_path = index[pddl][str_index]
        return self.stats_path

    def load_stats(self):
        if os.path.isfile(self.get_stats()):
            with open(self.get_stats()) as stream:
                data = json.load(stream)
                self.last_preimage = list(map(tuple, data["last_preimage"]))
                self.atom_map = item_to_dict(data["atom_map"])
                self.init = {x for x in self.atom_map if not self.atom_map[x]}
                to_add = set()
                for fact in self.last_preimage:
                    if fact not in self.atom_map:
                        print(f"Warning {fact} not in atom_map")
                        continue
                    to_add |= ancestors(fact, self.atom_map)
                self.last_preimage += list(to_add)
        else:
            raise FileExistsError(
                f"File {self.get_stats()} does not exist, cannot using oracle"
            )

    def is_relevant(self, result, node_from_atom, preimage):
        """
        returns True iff either one of the
        certified facts in result.get_certified()
        is_matching() (see above function)
        """
        can_atom_map = make_atom_map(node_from_atom)
        can_stream_map = make_stream_map(node_from_atom)
        assert objects_from_facts(self.init) == objects_from_facts(
            {f for f in can_atom_map if not can_atom_map[f]}
        )
        can_ans = tuple()
        can_parents = tuple()
        for domain_fact in result.domain:
            can_ans += (fact_to_pddl(domain_fact),)
            can_parents += (fact_to_pddl(domain_fact),)
            can_ans += ancestors_tuple(fact_to_pddl(domain_fact), can_atom_map)

        for can_fact in result.get_certified():
            is_match, match = is_matching(
                fact_to_pddl(can_fact), can_ans, preimage, self.atom_map, self.init
            )
            res = []
            self.labels.append(
                (
                    fact_to_pddl(can_fact),
                    can_parents,
                    result.external.name,
                    is_match,
                    can_atom_map,
                    can_stream_map,
                    tuple([obj_to_pddl(r) for r in result.output_objects]),
                )
            )
            if is_match:
                return True, (can_fact, match)
        return False, None

    def make_is_relevant_checker(self, remove_matched=False):
        if self.last_preimage is None or self.atom_map is None:
            self.load_stats()
        preimage = self.last_preimage.copy()
        if remove_matched:
            raise NotImplementedError("Removed matched does not work ... yet")

        def unique_is_relevant(result, node_from_atom):
            is_match, _ = self.is_relevant(result, node_from_atom, preimage)
            # if remove_matched and is_match:
            #    lifted, grounded = match
            #    preimage.remove(grounded)
            return is_match

        return unique_is_relevant


def make_stream_map(node_from_atom):
    stream_map = {}
    for atom in node_from_atom:
        node = node_from_atom[atom]
        result = node.result
        # TODO: Figure out how to deal with these bools?
        if result is None:
            stream_map[fact_to_pddl(atom)] = None
            continue
        if isinstance(result, bool):
            continue
        stream_map[fact_to_pddl(atom)] = result.external.name
    return stream_map


from learning.gnn.data import ModelInfo
from learning.gnn.data import construct_input
from learning.gnn.models import StreamInstanceClassifier


class Model(Oracle):
    def __init__(
        self, domain_pddl, stream_pddl, initial_conditions, goal_conditions, model_path
    ):
        super().__init__(domain_pddl, stream_pddl, initial_conditions, goal_conditions)
        self.model = None
        self.model_info = None
        self.model_path = model_path

    def load_model(self):
        self.model_info = ModelInfo(
            goal_facts=self.goal_facts,
            predicates=[p.name for p in self.domain.predicates],
            streams=[None] + [e["name"] for e in self.externals],
            stream_input_sizes=[None] + [len(e["domain"]) for e in self.externals],
        )
        self.model = StreamInstanceClassifier(
            self.model_info.node_feature_size,
            self.model_info.edge_feature_size,
            self.model_info.stream_input_sizes[1:],
            feature_size=4,
            use_gcn=False,
        )
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def make_is_relevant_checker(self, remove_matched=False):
        if self.model is None:
            self.load_model()
            assert self.model is not None
        oracle_checker = super().make_is_relevant_checker(remove_matched=remove_matched)

        def checker(result, node_from_atom):
            label = oracle_checker(result, node_from_atom)
            logit = self.predict(result, node_from_atom)
            pred = logit > 0.5
            if pred != label:
                print("Bad pred!", logit, label)
            return True if label else pred

        return checker

    def predict(self, result, node_from_atom):
        can_atom_map = make_atom_map(node_from_atom)
        can_stream_map = make_stream_map(node_from_atom)

        can_ans = tuple()
        can_parents = tuple()
        for domain_fact in result.domain:
            can_ans += (fact_to_pddl(domain_fact),)
            can_parents += (fact_to_pddl(domain_fact),)
            can_ans += ancestors_tuple(fact_to_pddl(domain_fact), can_atom_map)

        data = construct_input(
            can_parents,
            result.external.name,
            can_atom_map,
            can_stream_map,
            self.model_info,
        )
        logit = self.model(data, score=True).detach().numpy()[0]
        return logit
