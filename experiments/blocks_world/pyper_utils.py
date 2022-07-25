import getpass
USER = getpass.getuser()
from pyperplan.planner import Parser
from pyperplan.heuristics.relaxation import hAddHeuristic, hFFHeuristic, hMaxHeuristic
from pyperplan.heuristics.landmarks import LandmarkHeuristic
from pyperplan.pddl.pddl import Problem, Type, Effect, Action, Domain, Predicate
from pyperplan.planner import _ground
from pyperplan.grounding import _get_fact, _collect_facts
from pyperplan.search.searchspace import SearchNode
class BlocksWorldPyperTranslator:
    domain_file = f'/home/{USER}/drake-tamp/experiments/blocks_world/domain_logical.pddl'
    heuristics = {
        "hadd": hAddHeuristic,
        "hff": hFFHeuristic,
        "lama": LandmarkHeuristic
    }
    def __init__(self):
        parser = Parser(self.domain_file)
        self.domain = parser.parse_domain()
        self.object_type = self.domain.types['object']
    
    def parse_objects(self, init):
        objects = set()
        for f in init:
            if f.predicate in self.domain.predicates:
                args = [o for o in f.args]
                objects |= set(args)
        objects = {o:self.object_type for o in objects}
        return objects

    def parse_state(self, state):
        pyper_state = []
        for f in state:
            if f.predicate in self.domain.predicates:
                args = [(o, self.object_type) for o in f.args]
                pyper_state.append(Predicate(f.predicate, args))
        return pyper_state
    
    def create_pyper_problem(self, init, goal):
        problem = Problem(
            name="prob",
            domain=self.domain,
            objects=self.parse_objects(init),
            init=self.parse_state(init),
            goal=self.parse_state(goal)
        )
        return problem
    
    def create_heuristic(self, init, goal, h_name):
        problem = self.create_pyper_problem(init, goal)
        task = _ground(problem, remove_statics_from_initial_state=True)
        facts = _collect_facts(task.operators)
        h = self.heuristics[h_name](task)
        def h_lifted(child, goal):
            pyper_state = facts & frozenset({_get_fact(f) for f in self.parse_state(child.state)})
            pyper_search_node = SearchNode(pyper_state, None, None, 0)
            return h(pyper_search_node)
        return h_lifted