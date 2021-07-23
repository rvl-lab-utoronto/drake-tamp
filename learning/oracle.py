import json
import os
import pickle
from datetime import datetime
import numpy as np

import torch

from learning.data_models import InvocationInfo, ModelInfo, ProblemInfo
from learning.gnn.data import construct_input, construct_problem_graph
from learning.gnn.models import StreamInstanceClassifier
from learning.pddlstream_utils import *
from pddlstream.language.conversion import fact_from_evaluation

FILEPATH, _ = os.path.split(os.path.realpath(__file__))


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
    def __init__(
        self,
        domain_pddl,
        stream_pddl,
        initial_conditions,
        goal_conditions,
        model_poses = None 
    ):
        # model_poses is for scene graph, optional list of ["model_name", X_WM]
        # where X_WM is a RigidTransform
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
        self.model_info = None
        self.problem_info = None
        self.model_poses = model_poses
        self.run_attr = None

    def set_infos(self, domain, externals, goal_exp, evaluations):
        self.set_model_info(domain, externals)
        self.set_problem_info(goal_exp, evaluations)

    def set_problem_info(self, goal_exp, evaluations):
        if not goal_exp[0] == "and":
            raise NotImplementedError(
                f"Expected goal to be a conjunction of facts. Got {goal_exp}.Need to parse this correctly."
            )
        self.problem_info = ProblemInfo(
            goal_facts=tuple([fact_to_pddl(f) for f in goal_exp[1:]]),
            initial_facts=tuple([fact_to_pddl(fact_from_evaluation(f)) for f in evaluations]),
            model_poses=self.model_poses
        )

    def set_run_attr(self, run_attr):
        self.run_attr = run_attr

    def set_model_info(self, domain, externals):
        new_externals = []
        for external in externals:
            new_externals.append(
                {
                    "certified": external.certified,
                    "domain": external.domain,
                    "fluents": external.fluents,
                    "inputs": external.inputs,
                    "outputs": external.outputs,
                    "name": external.name,
                }
            )
        self.model_info = ModelInfo(
            predicates=[p.name for p in domain.predicates],
            streams=[None] + [e["name"] for e in new_externals],
            stream_num_domain_facts=[None] + [len(e["domain"]) for e in new_externals],
            stream_num_inputs = [None] + [len(e["inputs"]) for e in new_externals],
            stream_domains = [None] + [e["domain"] for e in new_externals],
            domain = domain
        )

    def save_labeled(self, stats_path, path=None):
        if not os.path.isdir(f"{FILEPATH}/data"):
            os.mkdir(f"{FILEPATH}/data")
        if not os.path.isdir(f"{FILEPATH}/data/labeled"):
            os.mkdir(f"{FILEPATH}/data/labeled")
        info_path = f"{FILEPATH}/data/labeled/data_info.json"
        data_info = {}
        if os.path.isfile(info_path):
            with open(info_path, "r") as f:
                data_info = json.load(f)
        if path is None:
            path = self.save_path

        _, datafile = os.path.split(path)

        # we use the stats,json for the non-oracle run for the listed
        # run attributes statistics
        with open(self.get_stats(), "r") as f:
            stats = json.load(f)

        if self.run_attr is not None:
            self.run_attr["sample_time"] = stats["summary"]["sample_time"]
            self.run_attr["search_time"] = stats["summary"]["search_time"]
            self.run_attr["run_time"] = stats["summary"]["run_time"]
            self.run_attr["iterations"] = stats["summary"]["iterations"]
            self.run_attr["complexity"] = stats["summary"]["complexity"]
            self.run_attr["evaluations"] = stats["summary"]["evaluations"]
            pddl = self.domain_pddl + self.stream_pddl 
            if pddl not in data_info:
                # only save name of pkl file
                data_info[pddl] = []
            data_info[pddl].append((self.run_attr, datafile, len(self.labels)))

            with open(info_path, "w") as f:
                json.dump(data_info, f, indent = 4, sort_keys = True)

        data = {}
        data["stats_path"] = stats_path
        data["domain_pddl"] = self.domain_pddl
        data["stream_pddl"] = self.stream_pddl
        #data["initial_conditions"] = tuple(self.initial_conditions)
        #data["goal_conditions"] = tuple(self.goal_conditions)
        data["model_info"] = self.model_info
        data["problem_info"] = self.problem_info
        self.problem_info.object_mapping = {k:v.value for k,v in Object._obj_from_name.items()}
        self.problem_info.problem_graph = construct_problem_graph(self.problem_info)#, self.model_info)
        data["num_labels"] = len(self.labels)

        with open(path, "wb") as stream:
            pickle.dump(data, stream)

        dirpath = os.path.splitext(path)[0]
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)

        for i, label in enumerate(self.labels):
            labelpath = os.path.join(dirpath, f"label_{i}.pkl")
            with open(labelpath, "wb") as lfile:
                pickle.dump(label, lfile)

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
        #can_stream_map = make_stream_map(node_from_atom)
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
            is_match, _ = is_matching(
                fact_to_pddl(can_fact), can_ans, preimage, self.atom_map, self.init
            )

            if is_match:
                break
        self.labels.append(InvocationInfo(result, node_from_atom, label=is_match))

        return is_match


    def make_is_relevant_checker(self, remove_matched=False):
        if self.last_preimage is None or self.atom_map is None:
            self.load_stats()
        preimage = self.last_preimage.copy()
        if remove_matched:
            raise NotImplementedError("Removed matched does not work ... yet")

        def unique_is_relevant(result, node_from_atom):
            is_match = self.is_relevant(result, node_from_atom, preimage)
            # if remove_matched and is_match:
            #    lifted, grounded = match
            #    preimage.remove(grounded)
            return is_match

        return unique_is_relevant


class Model(Oracle):
    def __init__(
        self, domain_pddl, stream_pddl, initial_conditions, goal_conditions, model_path
    ):
        super().__init__(domain_pddl, stream_pddl, initial_conditions, goal_conditions)
        self.model_path = model_path

    def load_model(self):
        self.model = StreamInstanceClassifier(
            self.model_info.node_feature_size,
            self.model_info.edge_feature_size,
            self.model_info.stream_num_domain_facts[1:],
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
        invocation_info = InvocationInfo(result, node_from_atom)
        data = construct_input(
            invocation_info,
            self.problem_info,
            self.model_info,
        )
        logit = self.model(data, score=True).detach().numpy()[0]
        return logit

class StupidModel(Oracle):
    def predict(self, instance, atom_map, instantiator):
        return np.random.uniform()

class ComplexityModel(Oracle):
    def predict(self, instance, atom_map, instantiator):
        complexity = instantiator.compute_complexity(instance)
        return complexity
