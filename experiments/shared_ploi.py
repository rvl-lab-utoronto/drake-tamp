from collections import defaultdict
from learning.data_models import ModelInfo, ProblemInfo
from learning.gnn.data import construct_problem_graph, construct_problem_graph_input
from torch_geometric.data import Batch

import torch
from learning.gnn.models import PLOIAblationModel
from learning import oracle as ora
from pddlstream.algorithms.algorithm import parse_problem
from pddlstream.language.constants import PDDLProblem
from pddlstream.language.conversion import evaluation_from_fact, fact_from_evaluation
from pddlstream.language.object import Object
from learning.pddlstream_utils import fact_to_pddl
from pddlstream.algorithms.meta import solve

class PLOIFilter:
    def __init__(self, problem, path, model_poses):
        self.problem = problem
        # This has an important side effect which sets the object names in pddl
        # TODO: Make it explicit.
        self.parsed_problem = evaluations, goal_exp, domain, externals = parse_problem(problem)
        self.all_objects = self.parse_evaluation_objects(evaluations)
        self.goal_objects = self.parse_goal_objects(goal_exp)
        self.threshold = 1.
        self.model_path = path
        self.model_poses = model_poses
        self.relevant = None
        # self.preds = {o: 1 for o in self.all_objects}
        # self.preds[Object.from_value('blocker0')] = 0
        # self.preds[Object.from_value('blocker1')] = 0
        # self.preds[Object.from_value('blocker4')] = 0.5
        self.load_model()
        
        # self.preds = {o: 0 for o in self.all_objects}

    def parse_evaluation_objects(self, evaluations):
        init_objects = set()
        for ev in evaluations:
            x = fact_from_evaluation(ev)
            init_objects.update(set(x[1:]))
        return init_objects

    def parse_goal_objects(self, goal_exp):
        '''Takes a goal expression and returns a list of goal objects of type Object'''
        assert goal_exp[0] == 'and'
        object_set = {o for f in goal_exp[1:] for o in f[1:]}
        return object_set

    def reduce_problem(self):
        '''Take a PDDLProblem, a PLOI oracle and a threshold, and return the a reduced PDDLProblem
        where objects with a score below the threshold (and their associated facts) have been excluded.
        
        Makes exceptions for objects which are part of the goal state.'''
        objects_to_keep = {o for o, score in self.preds.items() if score >= self.threshold} | {o for o in self.goal_objects}
        self.relevant = [o.value for o in objects_to_keep]

    def reduce_threshold(self):
        max_below_threshold = max((self.preds[o] for o in self.preds if self.preds[o] < self.threshold), default=None)
        if max_below_threshold is None:
            return False

        self.threshold = min(self.threshold * 0.9, max_below_threshold)
        print('New threshold', self.threshold)
        return True

    def load_model(self):
        evaluations, goal_exp, domain, externals = self.parsed_problem
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
        self.problem_info = ProblemInfo(
            goal_facts=tuple([fact_to_pddl(f) for f in goal_exp[1:]]),
            initial_facts=tuple([fact_to_pddl(fact_from_evaluation(f)) for f in evaluations]),
            model_poses=self.model_poses,
            object_mapping = {k:v.value for k,v in Object._obj_from_name.items()}
        )
        self.model_info.problem_graph_node_feature_size = 3 + 1
        self.model_info.problem_graph_edge_feature_size = self.model_info.num_predicates + 1

        model = PLOIAblationModel(model_info=self.model_info)
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))

        self.problem_info.problem_graph = construct_problem_graph(self.problem_info)#, self.model_info)
        problem_graph_input = construct_problem_graph_input(self.problem_info, self.model_info)
        preds = model(Batch.from_data_list([problem_graph_input]))
        preds = torch.sigmoid(preds)
        self.preds = dict(zip(map(Object.from_name, problem_graph_input.nodes), preds.flatten().detach().numpy()))

def solve_ploi(problem, *args, **kwargs):
    if 'ploi_filter' in kwargs:
        kwargs.pop('algorithm')
        reset = kwargs.pop('reset')
        ploi_filter = PLOIFilter(problem, kwargs.pop('ploi_filter'), model_poses=kwargs.pop('model_poses'))
        while True:
            reduced_problem = ploi_filter.reduce_problem()
            reset(ploi_filter.relevant)
            solution = solve(
                reduced_problem,
                *args, 
                algorithm='adaptive',
                max_complexity=15,
                **kwargs
            )
            if solution.plan is not None:
                return solution
            if not ploi_filter.reduce_threshold():
                break
    else:
        return solve(problem, *args, **kwargs)
        

    return None

def construct_oracle(mode, pddl_problem, problem_info, model_poses, **kwargs):
    if mode in ["oracle", "save"]:
        oracle_class = ora.Oracle
    elif mode == "oraclemodel":
        oracle_class = ora.OracleModel
    elif mode == "oracleexpansion":
        oracle_class = ora.OracleModelExpansion
    elif mode == "complexityV3":
        oracle_class = ora.ComplexityModelV3
    elif mode == "complexityandstructure":
        oracle_class = ora.ComplexityModelV3StructureAware
    elif mode == "model":
        oracle_class = ora.Model
    elif mode == "cachingmodel":
        oracle_class = ora.CachingModel
    elif mode == "multiheadmodel":
        oracle_class = ora.MultiHeadModel
    elif mode == "complexitycollector":
        oracle_class = ora.ComplexityDataCollector
    elif mode == "complexitoracle":
        oracle_class = ora.OracleAndComplexityModelExpansion
    elif mode == "oracledagger":
        oracle_class = ora.OracleDAggerModel
    elif mode == "normal":
        return None
    elif mode == "statsablation":
        oracle_class = ora.StatsAblation
    else:
        raise ValueError(f"Unrecognized mode {mode}")
    oracle = oracle_class(
        pddl_problem.domain_pddl,
        pddl_problem.stream_pddl,
        pddl_problem.init,
        pddl_problem.goal,
        model_poses=model_poses,
        **kwargs
    )
    oracle.set_run_attr(problem_info.attr)
    return oracle
