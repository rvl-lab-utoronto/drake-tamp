from collections import defaultdict
import json
import os
import pickle
from datetime import datetime
import numpy as np

import torch
from torch_geometric.data.data import Data

from learning.data_models import HyperModelInfo, InvocationInfo, ModelInfo, ProblemInfo, RuntimeInvocationInfo, StreamInstanceClassifierV2Info
from learning.gnn.data import construct_hypermodel_input_faster, construct_input, construct_problem_graph, construct_problem_graph_input, construct_with_problem_graph, fact_level
from learning.gnn.models import HyperClassifier, StreamInstanceClassifier, StreamInstanceClassifierV2
from learning.pddlstream_utils import *
from pddlstream.language.conversion import evaluation_from_fact, fact_from_evaluation
from torch_geometric.data.batch import Batch

from pddlstream.language.object import SharedOptValue, UniqueOptValue

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
        # if not ans_g:
        #     continue  # never need to match initial conditions
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
        model_poses = None,
        data_collection_mode=False
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
        self.data_collection_mode = data_collection_mode

    def set_infos(self, domain, externals, goal_exp, evaluations):
        self.set_problem_info(goal_exp, evaluations)
        self.set_model_info(domain, externals)

    def set_problem_info(self, goal_exp, evaluations):
        if not goal_exp[0] == "and":
            raise NotImplementedError(
                f"Expected goal to be a conjunction of facts. Got {goal_exp}.Need to parse this correctly."
            )
        self.problem_info = ProblemInfo(
            goal_facts=tuple([fact_to_pddl(f) for f in goal_exp[1:]]),
            initial_facts=tuple([fact_to_pddl(fact_from_evaluation(f)) for f in evaluations]),
            model_poses=self.model_poses,
            object_mapping = {k:v.value for k,v in Object._obj_from_name.items()}
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
            stream_num_outputs = [None] + [len(e["outputs"]) for e in new_externals],
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
            self.run_attr["stats_path"] = stats_path
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
        if not result.is_input_refined_recursive():
            return True, None
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
            is_match, match = is_matching(
                fact_to_pddl(can_fact), can_ans, preimage, self.atom_map, self.init
            )

            if is_match:
                break
        self.labels.append(InvocationInfo(result, node_from_atom, label=is_match))

        return is_match, match


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
            if self.data_collection_mode:
                return True
            return is_match

        return unique_is_relevant

    def after_run(self, **kwargs):
        pass

class OracleModel(Oracle):
    def make_is_relevant_checker(self):
        if self.last_preimage is None or self.atom_map is None:
            self.load_stats()
        preimage = self.last_preimage.copy()
        def unique_is_relevant(result, node_from_atom):
            is_match, match = self.is_relevant(result, node_from_atom, preimage)
            return is_match, match

        return unique_is_relevant
    def predict(self, result, node_from_atom, **kwargs):
        if not hasattr(self, 'relevant_checker'):
            self.relevant_checker = self.make_is_relevant_checker()
            self.previously_matched = defaultdict(int)
        if not all([d in node_from_atom for d in result.domain]):
            return 1
        is_match, match = self.relevant_checker(result, node_from_atom)
        if is_match:
            self.previously_matched[match] += 1
            score = 2 / self.previously_matched[match]
            return score
        return 0

class Model(Oracle):
    def __init__(
        self, domain_pddl, stream_pddl, initial_conditions, goal_conditions, model_path, model_poses
    ):
        super().__init__(domain_pddl, stream_pddl, initial_conditions, goal_conditions, model_poses = model_poses)
        self.model_path = model_path
        self.model = None

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
        self.model_info = HyperModelInfo(
            predicates=[p.name for p in domain.predicates],
            streams=[None] + [e["name"] for e in new_externals],
            stream_num_domain_facts=[None] + [len(e["domain"]) for e in new_externals],
            stream_num_inputs = [None] + [len(e["inputs"]) for e in new_externals],
            stream_domains = [None] + [e["domain"] for e in new_externals],
            domain = domain
        )

    def load_model(self):
        self.model = HyperClassifier(
            self.model_info,
            with_problem_graph= True,
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
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

    def predict(self, result, node_from_atom, **kwargs):
        if not result.is_refined() or not all([d in node_from_atom for d in result.domain]):
            return 1
        if self.model is None:
            self.load_model()
            assert self.model is not None
        invocation_info = InvocationInfo(result, node_from_atom)
        data = construct_with_problem_graph(construct_hypermodel_input_faster)(
            invocation_info,
            self.problem_info,
            self.model_info,
        )
        #TODO: fix this 
        data = Batch().from_data_list([data])
        logit = self.model(data, score=True).detach().numpy()[0][0]
        return logit

class CachingModel(Model):
    def set_model_info(self, domain, externals):
        super().set_model_info(domain, externals)
        self.load_model()
        self.counts = {}
        self.logits = {}
        self.init_objects = objects_from_facts(self.problem_info.initial_facts)

    def calculate_result_key(self, result, atom_map):
        facts = [fact_to_pddl(f) for f in result.get_certified()]
        domain = [fact_to_pddl(f) for f in result.domain]
        result_key = tuple()
        for fact in facts:
            atom_map[fact] = domain
            result_key += standardize_facts(ancestors_tuple(fact, atom_map=atom_map), self.init_objects)
        return result_key

    def predict(self, result, node_from_atom, levels, atom_map, **kwargs):
        l = max(levels[evaluation_from_fact(f)] for f in result.domain) + 1  + result.call_index
        if not all([d in node_from_atom for d in result.domain]):
            return 0.5/l
        result_key = self.calculate_result_key(result, atom_map)
        if result_key not in self.logits:
            invocation_info = InvocationInfo(result, node_from_atom)
            data = construct_with_problem_graph(construct_hypermodel_input_faster)(
                invocation_info,
                self.problem_info,
                self.model_info,
            )
            #TODO: fix this
            data = Batch().from_data_list([data])
            self.logits[result_key] = self.model(data, score=True).detach().numpy()[0][0]
        
        self.counts[result_key] = self.counts.get(result_key, 0) + 1
        return self.logits[result_key]/(l  + self.counts[result_key] - 1)

class StupidModel(Oracle):
    def predict(self, instance, atom_map, instantiator):
        return np.random.uniform()

class ComplexityModel(Oracle):
    def predict(self, instance, atom_map, instantiator):
        complexity = instantiator.compute_complexity(instance)
        return complexity


class ComplexityModelV2(Oracle):
    def predict(self, result, node_from_atom, **kwargs):
        if not result.is_refined() or not all([d in node_from_atom for d in result.domain]):
            return 1
        invocation_info = InvocationInfo(result, node_from_atom)
        level = -1
        for f in result.domain:
            level = max(level, 1 + fact_level(fact_to_pddl(f), invocation_info))

        return level/10

class ComplexityModelV3(Oracle):
    def predict(self, result, node_from_atom, levels, **kwargs):
        l = max(levels[evaluation_from_fact(f)] for f in result.domain) + 1  + result.call_index
        return 1  / l

class ComplexityModelV3StructureAware(Oracle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = {}

    def calculate_result_key(self, result, atom_map):
        if not hasattr(self, 'init_objects'):
            self.init_objects = objects_from_facts(self.problem_info.initial_facts)
        facts = [fact_to_pddl(f) for f in result.get_certified()]
        domain = [fact_to_pddl(f) for f in result.domain]
        result_key = tuple()
        for fact in facts:
            atom_map[fact] = domain
            result_key += standardize_facts(ancestors_tuple(fact, atom_map=atom_map), self.init_objects)
        return result_key

    def predict(self, result, node_from_atom, levels, atom_map, **kwargs):
        l = max(levels[evaluation_from_fact(f)] for f in result.domain) + 1  + result.call_index
        if not all([d in node_from_atom for d in result.domain]):
            return 1 / l
        result_key = self.calculate_result_key(result, atom_map)
        count = self.counts.get(result_key, 0) 
        self.counts[result_key] = count + 1
        return 1  / (l + count)

class OracleModelExpansion(OracleModel):
    def load_stats(self):
        super().load_stats()
        preimage_no_leaves = set()
        for atom in self.last_preimage:
            if not any(atom in parents and child in self.last_preimage for child, parents in self.atom_map.items()):
                print('Filtered leaf', atom)
            else:
                preimage_no_leaves.add(atom) 
                print('Added internal node', atom)

        self.last_preimage = preimage_no_leaves

    def predict(self, result, node_from_atom, **kwargs):
        if not hasattr(self, 'relevant_checker'):
            self.relevant_checker = self.make_is_relevant_checker()
            self.previously_matched = defaultdict(int)
        if not all([d in node_from_atom for d in result.domain]):
            return 1
        is_match, match = self.relevant_checker(result, node_from_atom)
        if is_match:
            self.previously_matched[match] += 1
            score = 2 / self.previously_matched[match]
            return score
        return 0

class OracleAndComplexityModelExpansion(OracleModelExpansion):
    """Uses an oracle + complexity for refined facts and just complexity for unrefined""" 
    def predict(self, result, node_from_atom, levels,**kwargs):
        if not hasattr(self, 'relevant_checker'):
            self.relevant_checker = self.make_is_relevant_checker()
            self.previously_matched = defaultdict(int)
        if not all([d in node_from_atom for d in result.domain]):
            score = 0.5
        else:
            is_match, match = self.relevant_checker(result, node_from_atom)
            if is_match:
                self.previously_matched[match] += 1
                score = 1 / self.previously_matched[match] 
            else:
                score = 0
        
        l = max(levels[evaluation_from_fact(f)] for f in result.domain) + 1  + result.call_index
        return score / l


class OracleDAggerModel(OracleModel):
    def is_relevant_fact(self, can_fact, can_atom_map, preimage):
        """
        returns True iff either one of the
        certified facts in result.get_certified()
        is_matching() (see above function)
        """
        assert objects_from_facts(self.init) == objects_from_facts(
            {f for f in can_atom_map if not can_atom_map[f]}
        )
        can_ans = ancestors_tuple(can_fact, can_atom_map)
        is_match, match = is_matching(
            can_fact, can_ans, preimage, self.atom_map, self.init
        )

        return is_match, match

    def predict(self, result, node_from_atom, levels, **kwargs):
        if not hasattr(self, 'relevant_checker'):
            self.relevant_checker = self.make_is_relevant_checker()
            self.previously_matched = defaultdict(int)
            self.scores = {}
        if all([d in node_from_atom for d in result.domain]):
            is_match, match = self.relevant_checker(result, node_from_atom)
            if is_match:
                self.previously_matched[match] += 1
                score = (True,  self.previously_matched[match], node_from_atom.copy())
                self.scores[result] = score
            else:
                self.scores[result] = (False, 0, node_from_atom.copy())
        else:
            assert False
        l = max(levels[evaluation_from_fact(f)] for f in result.domain) + 1  + result.call_index
        return 1 / l
    
    def after_run(self, store, expanded, **kwargs):
        ################
        print('# Expanded', len(expanded))
        print({e for e in expanded if e.call_index == 0} - set(self.scores))
        print('# Scored and Expanded / # Expaneded', len(expanded & set(self.scores))/len(expanded))
        print('# Irrelevant and Expanded / # Expanded', len(expanded & set(x for x in self.scores if not self.scores[x][0]))/len(expanded))
        print('# Irrelevant and Expanded (w/o motions) / # Expanded', len(expanded & set(x for x in self.scores if not self.scores[x][0] and x.name != 'find-traj'))/len(expanded))
        ############
        from learning.pddlstream_utils import fact_to_pddl, ancestors_tuple
        atom_map = store.node_from_atom_to_atom_map({})
        atom_map = {fact_to_pddl(key): list(map(fact_to_pddl, value)) for key, value in atom_map.items()}

        plan_preimage = set(map(fact_to_pddl, store.last_preimage))
        preimage_no_leaves = set()
        for atom in plan_preimage:
            if not any(atom in parents and child in plan_preimage for child, parents in atom_map.items()):
                pass
            else:
                preimage_no_leaves.add(atom) 
        if store.is_solved():
            print('######## ORACLE PREIMAGE ########')
            for atom in self.last_preimage:
                print(atom, ancestors_tuple(atom, self.atom_map))

            print('######## FOUND PREIMAGE ########')
            not_matched = 0
            for fact in preimage_no_leaves:
                is_match, match = self.is_relevant_fact(fact, atom_map, self.last_preimage)
                if not is_match:
                    print('No match', fact, ancestors_tuple(fact, atom_map))
                    not_matched += 1
            print(f'No match for {not_matched / len(preimage_no_leaves)}')

            self.last_preimage = preimage_no_leaves
            self.atom_map = atom_map
            scores = {}
            for result, (_, _, node_from_atom) in self.scores.items():
                if result in expanded:
                    scores[result] = self.is_relevant(result, node_from_atom, preimage_no_leaves)[0]

            print('# Irrelevant and Expanded / # Expanded', len(set(x for x in scores if not scores[x]))/len(expanded))
            print('# Irrelevant and Expanded (w/o motions) / # Expanded', len(set(x for x in scores if not scores[x] and x.name != 'find-traj'))/len(expanded))

class ComplexityDataCollector(ComplexityModelV3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_node_from_atoms = {}

    def label(self, results, preimage, atom_map, EAGER_MODE=False):
        assert not self.labels
        self.atom_map = {fact_to_pddl(key): list(map(fact_to_pddl, value)) for key, value in atom_map.items()}
        self.init = {x for x in self.atom_map if not self.atom_map[x]}
        self.last_preimage = set(map(fact_to_pddl, preimage))
        if EAGER_MODE:
            preimage_no_leaves = set()
            for atom in self.last_preimage:
                if not any(atom in parents and child in self.last_preimage for child, parents in self.atom_map.items()):
                    pass
                else:
                    preimage_no_leaves.add(atom) 
            self.last_preimage = preimage_no_leaves
        #TODO: only label expanded?
        for result, node_from_atom in self.saved_node_from_atoms.items():
            self.is_relevant(result, node_from_atom, self.last_preimage)

    def after_run(self, store, expanded, logpath):
        if store.is_solved():
            atom_map = store.node_from_atom_to_atom_map({})
            preimage = store.last_preimage
            self.label(expanded, preimage, atom_map)
            self.save_stats(logpath + "stats.json")
            self.save_labeled(logpath + "stats.json")

    def predict(self, result, node_from_atom, levels, **kwargs):
        if all([d in node_from_atom for d in result.domain]):
            self.saved_node_from_atoms[result] = node_from_atom.copy()
        return super().predict(result, node_from_atom, levels)

class MultiHeadModel(Oracle):
    def __init__(self, *args, **kwargs):
        model_path = kwargs.pop('model_path')
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.model = None
    
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
        self.model_info = StreamInstanceClassifierV2Info(
            predicates=[p.name for p in domain.predicates],
            streams=[None] + [e["name"] for e in new_externals],
            stream_num_domain_facts=[None] + [len(e["domain"]) for e in new_externals],
            stream_num_inputs = [None] + [len(e["inputs"]) for e in new_externals],
            stream_num_outputs = [None] + [len(e["outputs"]) for e in new_externals],
            stream_domains = [None] + [e["domain"] for e in new_externals],
            domain = domain
        )
        self.load_model()

    def load_model(self):
        self.model = StreamInstanceClassifierV2(
            self.model_info,
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()

        self.problem_info.problem_graph = construct_problem_graph(self.problem_info)#, self.model_info)
        problem_graph_input = construct_problem_graph_input(self.problem_info, self.model_info)
        self.object_reps = self.model.get_init_reps(problem_graph_input)
        self.history = {}
        self.counts = {}
        self.init_objects = objects_from_facts(self.problem_info.initial_facts)
    
    def calculate_result_key(self, result, atom_map):
        facts = [fact_to_pddl(f) for f in result.get_certified()]
        domain = [fact_to_pddl(f) for f in result.domain]
        result_key = tuple()
        for fact in facts:
            atom_map[fact] = domain
            result_key += standardize_facts(ancestors_tuple(fact, atom_map=atom_map), self.init_objects)
        return result_key

    def predict(self, result, node_from_atom, levels, atom_map, **kwargs):
        l = max(levels[evaluation_from_fact(f)] for f in result.domain) + 1  + result.call_index
        if not all([d in node_from_atom for d in result.domain]):
            return 0.5/l
        
        if result.instance in self.history:
            score, reps = self.history[result.instance]
            outputs = tuple()
            for o, r in zip(map(obj_to_pddl, result.output_objects), reps):
                self.object_reps[o] = r 
        else:
            inputs = tuple(map(obj_to_pddl, result.input_objects))
            outputs = tuple(map(obj_to_pddl, result.output_objects))
            data = Data(stream_schedule=[[{"name": result.name, "input_objects": inputs, "output_objects": outputs}]])
            score = self.model(data, object_reps=self.object_reps, score=True).detach().numpy()[0][0]
            self.history[result.instance] = (score, [self.object_reps[o] for o in outputs])

        result_key = self.calculate_result_key(result, atom_map)
        self.counts[result_key] = self.counts.get(result_key, 0) + 1
        count = self.counts[result_key]
        # count = 0
        return score/(l  + count  - 1)