
from pddlstream.language.conversion import fact_from_evaluation, objects_from_evaluations
import sys

sys.path.insert(0, '/home/mohammed/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/')
import pddl
from pddl.conditions import Atom, Conjunction, NegatedAtom
from pddl.actions import PropositionalAction, Action
from pddl.effects import Effect
import pddl.conditions as conditions

from pddl.pddl_types import TypedObject
import pddl.conditions as conditions

class Identifiers:
    idx = 0
    @classmethod
    def next(cls):
        cls.idx += 1
        return f'?x{cls.idx}'

def combinations(candidates):
    """Given a dictionary from key (k^j) to a set of possible values D_k^j, yield all the
    tuples [(k^j, v_i^j)] where v_i^j is an element of D_k^j"""
    keys, domains = zip(*candidates.items())
    for combo in itertools.product(*domains):
        yield dict(zip(keys, combo))

def bind_action(action, assignment):
    for par in action.parameters[:action.num_external_parameters]:
        if par.name not in assignment:
            assignment[par.name] = Identifiers.next()

    assert isinstance(action.precondition, Conjunction)
    precondition_parts = []
    for atom in action.precondition.parts:
        args = [assignment.get(arg, arg) for arg in atom.args]
        atom = Atom(atom.predicate, args) if not atom.negated else NegatedAtom(atom.predicate, args)
        precondition_parts.append(atom)

    effects = []
    for effect in action.effects:
        atom = effect.literal
        args = [assignment.get(arg, arg) for arg in atom.args]
        if atom.negated:
            atom = NegatedAtom(atom.predicate, args)
        else:
            atom = Atom(atom.predicate, args)
        condition = effect.condition # TODO: assign the parameters in this
        if effect.parameters or not isinstance(effect.condition, conditions.Truth):
            raise NotImplementedError
        effects.append((condition, atom, effect, assignment.copy()))

    arg_list = [assignment.get(par.name, par.name)
                for par in action.parameters[:action.num_external_parameters]]
    name = "(%s %s)" % (action.name, " ".join(arg_list))
    cost = 1

    if not all(param in assignment for param in action.parameters):
        params = [TypedObject(name=assignment.get(par.name), type_name=par.type_name) for par in action.parameters]

        arg_list = [assignment.get(par.name, par.name)
                    for par in action.parameters[:action.num_external_parameters]]
        name = "(%s %s)" % (action.name, " ".join(arg_list))
        pre = Conjunction(precondition_parts)
        eff = [Effect([], cond, atom) for (cond, atom, _, _) in effects]
        return Action(name, params, len(params), pre, eff, cost)
    return PropositionalAction(name, Conjunction(precondition_parts), effects, cost, action, assignment)
    
def find_applicable_brute_force(action, state, allow_missing, object_stream_map={}, filter_precond=True):
    """Given an action schema and a state, return the list of partially grounded
    operators possible in the given state"""

    # Find all possible assignments of the state.objects to action.parameters
    # such that [action.preconditions - {atom | atom in certified}] <= state
    candidates = {}
    for atom in action.precondition.parts:
        if atom.predicate in allow_missing:
            continue
        for ground_atom in state:
            if ground_atom.predicate != atom.predicate:
                continue
            for arg, candidate in zip(atom.args, ground_atom.args):
                candidates.setdefault(arg, set()).add(candidate)
    if not candidates:
        yield bind_action(action, {})
        return
    # find all the possible versions
    for assignment in combinations(candidates):
        partial_op = bind_action(action, assignment)

        if filter_precond:
            missing_positive = {atom for atom in partial_op.precondition.parts if not atom.negated} - set(state)

            # grounded positive precondition not in state
            if any(atom.predicate not in allow_missing or all(arg in object_stream_map for arg in atom.args) for atom in missing_positive):
                continue
            
            # negative precondition is positive in state
            negative = {atom for atom in partial_op.precondition.parts if atom.negated} 
            if any(atom.negate() in state for atom in negative):
                continue
        yield partial_op



def partially_ground(action, state, fluents):
    """Given an action schema and a state, return the list of partially grounded
    operators keeping any objects in fluent predicates lifted"""

    # Find all possible assignments of the state.objects to action.parameters
    # such that [action.preconditions - {atom | atom in certified}] <= state
    candidates = {}
    for atom in action.precondition.parts:

        for ground_atom in state:
            if ground_atom.predicate != atom.predicate or ground_atom.predicate in fluents:
                continue
            for arg, candidate in zip(atom.args, ground_atom.args):
                candidates.setdefault(arg, set()).add(candidate)
    if not candidates:
        yield bind_action(action, {})
        return
    # find all the possible versions
    for assignment in combinations(candidates):
        partial_op = bind_action(action, assignment)
        yield partial_op

def apply(action, state):
    """Given an action, and a state, compute the successor state"""
    state = set(state)
    for effect in action.effects:
        if effect.parameters or not isinstance(effect.condition, conditions.Truth):
            raise NotImplementedError
        if effect.literal.negated:
            pos_literal = effect.literal.negate()
            if pos_literal in state:
                state.remove(pos_literal)
        else:
            state.add(effect.literal)
    return state

def objects_from_state(state):
    return { arg for atom in state for arg in atom.args }


class SearchState:
    def __init__(self, state, object_stream_map, unsatisfied):
        self.state = state.copy()
        self.object_stream_map = object_stream_map.copy()
        self.unsatisfied = unsatisfied
        self.unsatisfiable = False
        self.children = []
        self.parent = None
        self.action = None

    def __eq__(self, other):
        # compare based on fluent predicates and computation graph in corresponding objects
        return False

    def __repr__(self):
        return str((self.state, self.object_stream_map, self.unsatisfied))

    def get_path(self):
        actions = []
        node = self
        while node is not None:
            actions.insert(0, node.action)
            node = node.parent
        return actions

class ActionStreamSearch:
    def __init__(self, init, goal, externals, actions):
        self.init_objects = {o for o in objects_from_state(init)}
        self.init = SearchState(init, {o:None for o in self.init_objects}, set())
        self.goal = goal
        self.externals = externals
        self.actions = actions
        self.streams_by_predicate = {}
        for stream in externals:
            for fact in stream.certified:
                self.streams_by_predicate.setdefault(fact[0], set()).add(stream)

        self.fluent_predicates = set()
        for action in actions:
            for effect in action.effects:
                self.fluent_predicates.add(effect.literal.predicate)

    def test_equal(self, s1, s2):
        return False
        f1 = sorted(f for f in s1.state if f.predicate in self.fluent_predicates)
        f2 = sorted(f for f in s2.state if f.predicate in self.fluent_predicates)

        # are f1 and f2 the same, down to a substitiution?
        sub = {o:o for o in self.init_objects}
        for a,b in zip(f1, f2):
            if a.predicate != b.predicate:
                return False
            if a == b:
                continue
            for o1, o2 in zip(a.args, b.args):
                if o1 != o2 and not (o1.startswith('?') and o2.startswith('?')):
                    return False
                if sub.setdefault(o1, o2) != o2:
                    return False
        # TODO: are the stream created objects the same down to a substitution of their computation graph
        return True
    
    def test_goal(self, state):
        return self.goal <= state.state 

    def action_successor(self, state, action):
        new_state = apply(action, state.state)
        missing_positive = {atom for atom in action.precondition.parts if not atom.negated} - set(state.state)
        return SearchState(new_state, state.object_stream_map, missing_positive | state.unsatisfied)
    
    def successors(self, state):
        for action in self.actions:
            ops = find_applicable_brute_force(action, state.state | state.unsatisfied, self.streams_by_predicate, state.object_stream_map)
            for op in ops:
                op_state = self.action_successor(state, op)
                op_state = instantiate_depth_first(op_state, self.externals)
                yield (op, op_state)
    

def identify_groups(facts, stream):

    # given a set of facts, and a stream capable of producing facts
    # return a grouping over the facts where each group of facts might
    # can be certified together

    # question: is the solution unique?
    # question: is the "maximal" part very important?

    # question: does my graph consist only of disjoint cliques?
    # if so, then i can use a flood fill algorithm 
    # lets assume thats the case and see if we run into trouble
    preds = set(cert[0] for cert in stream.certified)
    facts = {f for f in facts if f.predicate in preds}
    if not facts:
        return []

    adjacency = {f:set() for f in facts}
    for (a, b) in itertools.combinations(facts, 2):
        assignment = {}
        for certified in stream.certified:
            if certified[0] == b.predicate:
                assignment.update({var:val for (var,val) in zip(certified[1:], b.args)})
            if certified[0] == a.predicate:
                partial = {var:val for (var,val) in zip(certified[1:], a.args)}
                if any(assignment.get(var) != partial.get(var) for var in partial):
                    break
        else:
            if assignment:
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)

    unlabeled = facts.copy()
    labeled = set()
    groups = []
    while unlabeled:
        fact = unlabeled.pop()
        group = adjacency[fact]
        for el in group:
            unlabeled.remove(el)
            # check that it's a disconnected clique
            assert (adjacency[el] | set([el])) == (group | set([fact]))
        group.add(fact)
        
        # double check my assumption that its a disconnected clique
        assert not (labeled & group)
        labeled = labeled | group
        groups.append(group)
    return groups


def get_assignment(group, stream):
    assignment = {}
    for certified in stream.certified:
        for fact in group:
            if certified[0] == fact.predicate:
                partial = {var:val for (var,val) in zip(certified[1:], fact.args)}
                if any(assignment.get(var, partial[var]) != partial[var] for var in partial):
                    return None
                assignment.update(partial)
    return assignment

def instantiate(state, group, stream):
    assignment = get_assignment(group, stream)
    if not assignment:
        return None
    inputs = tuple(assignment.get(arg) for arg in stream.inputs)
    outputs = tuple(assignment.get(arg) for arg in stream.outputs)
    if any(x is None for x in inputs + outputs):
        return None
    domain = {Atom(dom[0], [assignment.get(arg) for arg in dom[1:]]) for dom in stream.domain}
    certified = {Atom(cert[0], [assignment.get(arg) for arg in cert[1:]]) for cert in stream.certified}
    missing = domain - state.state
    if any(atom.predicate not in streams_by_predicate for atom in missing):
        # domain facts not satisfiable
        return None

    if any(out in state.object_stream_map for out in outputs):
        # the same object cant be output by different stream instances
        # TODO: need to prune this node completely from the search...
        state.unsatisfiable = True
        return None
    instance = (stream, inputs, outputs)

    new_literals = state.state | certified
    new_unsat = state.unsatisfied - certified | missing
    new_object_stream_map = dict(state.object_stream_map, **{out:instance for out in outputs})
    state = SearchState(new_literals, new_object_stream_map, new_unsat)
    return state

def instantiation_successors(state, streams):
    new_states = []
    for stream in streams:
        for group in identify_groups(state.unsatisfied, stream):
            new_state = instantiate(state, group, stream)
            if new_state:
                new_states.append(new_state)
    return new_states

def identify_stream_groups(facts, streams):
    facts_grouped = set()
    for stream in streams:
        for group in identify_groups(facts, stream):
            assert not (facts_grouped & group)
            yield (stream, group)
            facts_grouped |= group

def instantiate_depth_first(state, streams):
    changed = False
    for (stream, group) in identify_stream_groups(state.unsatisfied, streams):
        new_state = instantiate(state, group, stream)
        if new_state is not None:
            state = new_state
            changed = True
    if changed:
        return instantiate_depth_first(state, streams)
    else:
        return state

def try_bfs(search):
    q = [search.init]
    closed = []
    while q:
        state = q.pop(0)
        if search.test_goal(state):
            return state.get_path()
        state.children = []
        for (op, child) in search.successors(state):
            child.action = op
            child.parent = state
            state.children.append((op, child))
            if child.unsatisfiable or any(search.test_equal(child, node) for node in closed):
                continue 
            q.append(child)
        closed.append(state)
            

if __name__ == '__main__':
    from experiments.blocks_world.run import *
    from pddlstream.algorithms.algorithm import parse_problem
    url = 'tcp://127.0.0.1:6000'
    problem_file = 'experiments/blocks_world/data_generation/random/train/2_0_1_77.yaml'

    (
        sim,
        station_dict,
        traj_directors,
        meshcat_vis,
        prob_info,
    ) = make_and_init_simulation(url, problem_file)
    problem, model_poses = construct_problem_from_sim(sim, station_dict, prob_info)
    evaluations, goal_exp, domain, externals = parse_problem(problem)

    # print("Initial:", str_from_object(problem.init))
    # print("Goal:", str_from_object(problem.goal))

    init = set()
    for evaluation in evaluations:
        x = fact_from_evaluation(evaluation)
        init.add(Atom(x[0], [o.pddl for o in x[1:]]))

    goal = set()
    assert goal_exp[0] == 'and'
    for x in goal_exp[1:]:
        goal.add(Atom(x[0], [o.pddl for o in x[1:]]))

    print('Initial:', init)
    print('\n\nGoal:', goal)
    [pick, move, place, stack, unstack] = domain.actions

    # todo: move this somewhere like a class member
    streams_by_predicate = {}
    for stream in externals:
        for fact in stream.certified:
            streams_by_predicate.setdefault(fact[0], set()).add(stream)


    search = ActionStreamSearch(init, goal, externals, domain.actions)
    print(try_bfs(search))
    # for op, _ in search.successors(search.init):
    #     print(op)

    # state = search.init
    # stream = externals[0]
    # groups = identify_groups(state.unsatisfied, stream)
    # print(groups)

    # s1 = search.action_successor(search.init, moves[0])
    # state = instantiate_depth_first(s1, externals)

    # [action]  = list(find_applicable_brute_force(pick, state.state, streams_by_predicate))
    # s2 = search.action_successor(state, action)
    # state = instantiate_depth_first(s2, externals)
    # print(state)
