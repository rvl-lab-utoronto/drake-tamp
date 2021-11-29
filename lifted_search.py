from collections import defaultdict
from typing import DefaultDict
from pddlstream.algorithms.algorithm import check_problem, parse_constants, parse_stream_pddl
from pddlstream.algorithms.common import evaluations_from_init
from pddlstream.algorithms.constraints import add_plan_constraints
from pddlstream.algorithms.downward import get_identical_atoms, get_problem, has_costs, parse_goal, set_unit_costs
from pddlstream.language.constants import EQ
from pddlstream.language.conversion import fact_from_evaluation, obj_from_value_expression, objects_from_evaluations
import sys
import itertools
import copy
from datetime import datetime
from pddlstream.language.exogenous import compile_to_exogenous

from pddlstream.language.stream import Stream
from pddlstream.language.object import Object
from pddlstream.language.temporal import SimplifiedDomain, parse_domain

sys.path.insert(0, '/home/mohammed/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/')
sys.path.insert(0, '/home/atharv/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/')
import pddl
from pddl.conditions import Atom, Conjunction, NegatedAtom, Disjunction, UniversalCondition, ExistentialCondition
from pddl_parser.parsing_functions import check_for_duplicates
from pddl.actions import PropositionalAction, Action
from pddl.effects import Effect
import pddl.conditions as conditions

from pddl.pddl_types import TypedObject

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
# @profile
def bind_action(action, assignment):
    for par in action.parameters:
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

    if not all(param.name in assignment for param in action.parameters):
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

    def build_candidates(condition):
        candidates = {}
        def _build_candidates(c):
            if isinstance(c, Atom) or isinstance(c, NegatedAtom):
                if c.predicate not in allow_missing:
                    for ground_atom in state:
                        if ground_atom.predicate == c.predicate:
                            for arg, candidate in zip(c.args, ground_atom.args):
                                candidates.setdefault(arg, set()).add(candidate)
            else:
                for part in c.parts:
                    _build_candidates(part)
        _build_candidates(condition)
        return candidates


    # Find all possible assignments of the state.objects to action.parameters
    # such that [action.preconditions - {atom | atom in certified}] <= state
    # candidates = {}
    # for atom in action.precondition.parts:
    #     if atom.predicate in allow_missing:
    #         continue
    #     for ground_atom in state:
    #         if ground_atom.predicate != atom.predicate:
    #             continue
    #         for arg, candidate in zip(atom.args, ground_atom.args):
    #             candidates.setdefault(arg, set()).add(candidate)

    candidates = build_candidates(action.precondition)

    filtered_candidates = candidates.copy()
    def filter_candidates(_p):
        if isinstance(_p, UniversalCondition) or isinstance(_p, ExistentialCondition):
            for p in _p.parameters:
                if p.name in filtered_candidates:
                    del filtered_candidates[p.name]
            for part in _p.parts:
                filter_candidates(part)
        if isinstance(_p, Atom):
            return
        else:
            for part in _p.parts:
                filter_candidates(part)
    filter_candidates(action.precondition)
            
    if not candidates:
        assert False, "why am i here"
        yield bind_action(action, {})
        return
    
    def ground_recurse(condition, a):
    
        def _ground_recurse(c, g, _a):
            """ Procedure for generating partially grounded operators while handling
                conjuctions, disjunctions and quantifier.
            """
            if isinstance(c, Conjunction):
                for part in c.parts: 
                    res, g = _ground_recurse(part, g, _a)
                    if not res:
                        return False, g
                return True, g
            elif isinstance(c, Disjunction):
                for part in c.parts: 
                    res, _g = _ground_recurse(part, g, _a)
                    if res:
                        return True, _g
                return False, g
            elif isinstance(c, UniversalCondition):
                param_list = [candidates.get(param.name, {}) for param in c.parameters]
                for params in itertools.product(*param_list):
                    for idx, param in enumerate(c.parameters):
                        _a[param.name] = params[idx]
                    res, g = _ground_recurse(c.parts[0], g, _a)
                    if not res:
                        return False, g
                return True, g
            elif isinstance(c, ExistentialCondition):
                param_list = [candidates.get(param.name, {}) for param in c.parameters]
                for params in itertools.product(*param_list):
                    for idx, param in enumerate(c.parameters):
                        _a[param.name] = params[idx]
                    res, _g = _ground_recurse(c.parts[0], g, _a)
                    if res:
                        return True, _g
                return False, g
            elif isinstance(c, Atom) or isinstance(c, NegatedAtom):
                args = [_a.get(arg, arg) for arg in c.args]
                atom = Atom(c.predicate, args) if not c.negated else NegatedAtom(c.predicate, args)

                if not atom.negated:
                    if atom not in state and (
                        atom.negate() in state or
                        atom.predicate not in allow_missing 
                    ):
                        return False, g
                else:
                    if atom not in state and (
                        atom.negate() in state or
                        atom.predicate in allow_missing 
                    ):
                        return False, g

                # # grounded positive precondition not in state
                # if not atom.negated and atom not in state:
                #     if atom.predicate not in allow_missing:
                #         return False, g
                #     if all(arg in object_stream_map for arg in atom.args):
                #         return False, g
                # if atom.negated and atom.negate() in state:
                #     return False, g
 
                g.append(atom)
                return True, g

            else:
                raise NotImplementedError
    
        return _ground_recurse(condition, [], a)
   
   
    # find all the possible versions
    for assignment in combinations(filtered_candidates):
        for par in action.parameters:
            if par.name not in assignment:
                assignment[par.name] = Identifiers.next()

        # assert isinstance(action.precondition, Conjunction)
        # precondition_parts = []
        # for atom in action.precondition.parts:
        #     args = [assignment.get(arg, arg) for arg in atom.args]
        #     atom = Atom(atom.predicate, args) if not atom.negated else NegatedAtom(atom.predicate, args)
        #     precondition_parts.append(atom)
        #     # grounded positive precondition not in state
        #     if (not atom.negated and atom not in state):
        #         if(atom.predicate not in allow_missing):
        #             feasible = False
        #             break
        #         if all(arg in object_stream_map for arg in atom.args):
        #             feasible = False
        #             break

        #     if atom.negated and atom.negate() in state:
        #         feasible = False
        #         break


        feasible, precondition_parts = ground_recurse(action.precondition, assignment)
        if not feasible:
            continue

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


        partial_op = PropositionalAction(name, Conjunction(precondition_parts), effects, cost, action, assignment)
    
        
        # are any of the preconditions missing?
        # if any of the preconditions correspond to non-certifiable AAAND they include missing args
        # then there's no way this will work

        if filter_precond:
            missing_positive = {atom for atom in partial_op.precondition.parts if not atom.negated} - set(state)

            # grounded positive precondition not in state
            if any(atom.predicate not in allow_missing or all(arg in object_stream_map for arg in atom.args) for atom in missing_positive):
                assert False
                continue
            
            # negative precondition is positive in state
            negative = {atom for atom in partial_op.precondition.parts if atom.negated} 
            if any(atom.negate() in state for atom in negative):
                assert False
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
    return (state | {eff[1] for eff in action.add_effects}) - {eff[1] for eff in action.del_effects}
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
    def __init__(self, state, object_stream_map, independant_streams, unsatisfied):
        self.state = state.copy()
        self.object_stream_map = object_stream_map.copy()
        self.independant_streams = independant_streams.copy()
        self.unsatisfied = unsatisfied
        self.unsatisfiable = False
        self.children = []
        self.parent = None
        self.action = None
        self.start_distance = 0
        self.rhs = 0
        self.num_attempts = 1
        self.num_successes = 1

    def __eq__(self, other):
        # compare based on fluent predicates and computation graph in corresponding objects
        return False

    def __repr__(self):
        return str((self.state, self.object_stream_map, self.unsatisfied))

    def get_path(self):
        path = []
        node = self
        while node is not None:
            path.insert(0, (node.parent, node.action, node))
            node = node.parent
        return path

    def get_actions(self):
        actions = []
        node = self
        while node is not None:
            actions.insert(0, node.action)
            node = node.parent
        return actions

class ActionStreamSearch:
    def __init__(self, init, goal, externals, actions):
        self.init_objects = {o for o in objects_from_state(init)}
        self.init = SearchState(init, {o:None for o in self.init_objects}, set(), set())
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
            # for action.precondition.parts:
                
            ops = find_applicable_brute_force(action, state.state | state.unsatisfied, self.streams_by_predicate, state.object_stream_map)
            for op in ops:
                new_world_state = apply(op, state.state)
                missing_positive = {atom for atom in op.precondition.parts if not atom.negated} - set(state.state)
                # this does the partial order plan
                try:
                    partial_plan = certify(state.state, state.object_stream_map, missing_positive | state.unsatisfied, self.streams_by_predicate)
                    # this extracts the important bits from it (i.e. description of the state)
                    new_world_state, object_stream_map, independant_streams, missing = extract_from_partial_plan(new_world_state, partial_plan)
                    op_state = SearchState(new_world_state, object_stream_map, independant_streams, missing)
                    yield (op, op_state)
                except Unsatisfiable:
                    continue

                
    
class Unsatisfiable(Exception):
    pass
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

def instantiate_stream_from_assignment(stream, assignment, new_vars=False):
    assignment = assignment.copy()
    if new_vars:
        for param in stream.inputs + stream.outputs:
            assignment.setdefault(param, Identifiers.next())

    inputs = tuple(assignment.get(arg) for arg in stream.inputs)
    outputs = tuple(assignment.get(arg) for arg in stream.outputs)
    domain = {Atom(dom[0], [assignment.get(arg) for arg in dom[1:]]) for dom in stream.domain}
    certified = {Atom(cert[0], [assignment.get(arg) for arg in cert[1:]]) for cert in stream.certified}

    return (inputs, outputs, domain, certified)
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
    expand_count = 0
    evaluate_count = 0
    while q:
        state = q.pop(0)
        expand_count += 1
        if search.test_goal(state):
            print(f'Explored {expand_count}. Evaluated {evaluate_count}')
            # print(state.get_path(), end='\n\n')
            # continue
            return state
        state.children = []
        for (op, child) in search.successors(state):
            child.action = op
            child.parent = state
            state.children.append((op, child))
            if child.unsatisfiable or any(search.test_equal(child, node) for node in closed):
                continue 
            evaluate_count += 1
            q.append(child)
        closed.append(state)

import itertools

from dataclasses import dataclass, field
# I want to think about partial order planning
@dataclass
class PartialPlan:
    agenda: set
    actions: set
    bindings: dict
    order: list
    links: list
    def copy(self):
        return PartialPlan(self.agenda.copy(), self.actions.copy(), self.bindings.copy(), self.order.copy(), self.links.copy())

id_provider = itertools.count()
@dataclass
class StreamAction:
    stream: Stream = None
    inputs: tuple = field(default_factory=tuple)
    outputs: tuple = field(default_factory=tuple)
    id: int = field(default_factory=lambda: next(id_provider))
    pre: set = field(default_factory=set)
    eff: set = field(default_factory=set)

    def __hash__(self):
        return self.id

@dataclass
class Resolver:
    action: StreamAction = None
    links: list = field(default_factory=[])
    binding: dict = None

def equal_barring_substitution(atom1, atom2, bindings):
    if atom1.predicate != atom2.predicate:
        return None
    sub = bindings.copy()
    for a1, a2 in zip(atom1.args, atom2.args):
        if sub.setdefault(a1, a2) != a2:
            return None
    return sub

def get_resolvers(partial_plan, agenda_item, streams_by_predicate):
    (incomplete_action, missing_precond) = agenda_item
    for action in partial_plan.actions:
        if missing_precond in action.eff:
            assert False, "Didnt expect to get here, considering im doing the same work below"
            yield Resolver(links=[(action, missing_precond, incomplete_action)])
            continue

        # # check bindings
        # for eff in action.eff:
        #     # if equal barring substitution:
        #     sub = equal_barring_substitution(missing_precond, eff, partial_plan.bindings)
        #     if sub:
        #         if missing_precond.predicate in streams_by_predicate:
        #             # print(missing_precond, eff, action.stream)
        #             continue
        #         else:
        #             print(missing_precond, eff, action.stream)

        #         yield Resolver(binding=sub, link=(action, missing_precond, incomplete_action))

    

    for stream in streams_by_predicate.get(missing_precond.predicate, []):
        assignment = get_assignment((missing_precond, ), stream)

        (inputs, outputs, pre, eff) = instantiate_stream_from_assignment(stream, assignment, new_vars=True)

        binding = {o: o for o in outputs}

        action = StreamAction(stream, inputs, outputs, pre=pre, eff=eff)
        # TODO: continue if any of the atoms in eff are already produced by an action in the plan
        # In fact, it may be easier... that we continue if any of outputs are in any achieved facts?
        if any(o in partial_plan.bindings for o in outputs):
            continue
        if any(assignment.get(p) is None for p in stream.inputs):
            action.new_input_variables = True
            links = [(action, missing_precond, incomplete_action)]
        else:
            action.new_input_variables = False

            # could figure out all the links from this action to existing agenda items.
            # assumption: no other future action could certify the facts that this action certifies.

            # is it possible that:
            #  this action resolves a1 and a2
            #  a1 has many resolvers, so we dont do anything
            #  a2 has only one resolver, so we apply it, and resolve a1 along the way
            #  but now, because we resolved a1, we have made an irrevocable choice even though there might have been another way
            
            # I dont think this is a problem because the fact that one action will resolve a1 and a2 means that any other choice
            # for resolving a1 would have failed to resolve a2. Because there's only one resolver of a2. So had we resolved a1 by
            # some other means, then the only resolver of a2 would have to reproduce a1, but that's not allowed.
            links = []
            for (incomplete_action, missing_precond) in partial_plan.agenda:
                if missing_precond in eff:
                    links.append((action, missing_precond, incomplete_action))


            # could figure out all the links from existing actions to the preconditions of this action.
            # assumption: no facts are every provided by more than one existing action.
            for missing_precond in pre:
                for existing_action in partial_plan.actions:
                    if missing_precond in existing_action.eff:
                        links.append((existing_action, missing_precond, action))

        yield Resolver(action, links=links, binding=binding)

def successor(plan, resolver):
    plan = plan.copy()
    if resolver.action:
        plan.actions.add(resolver.action)
        plan.agenda |= {(resolver.action, f) for f in resolver.action.pre} 
    if resolver.links:
        for link in resolver.links:
            plan.links.append(link)
            plan.agenda = plan.agenda - {(link[2], link[1])}

    if resolver.binding:
        plan.bindings.update(resolver.binding)
    
    return plan

def certify(state, object_stream_map, missing, streams_by_predicate):

    init_action = StreamAction(eff=state)
    goal_action = StreamAction(pre=missing)
    p0 = PartialPlan(agenda={(goal_action, sub) for sub in missing}, actions={init_action, goal_action}, bindings={o:o for o in object_stream_map}, order=[], links=[])


    while p0.agenda:
        for agenda_item in p0.agenda:
            resolvers = list(get_resolvers(p0, agenda_item, streams_by_predicate))
            if not resolvers:
                raise Unsatisfiable('Deadend')
            if len(resolvers) > 1:
                # assumes that there is at least one fact that will uniquely identify the stream, doesnt it?
                # e.g if stream A certifies (p ?x ?y) and (q ?y ?z)
                # but stream B certifies (p ?x ?y) and (r ?y ?z)
                # and stream C certifies (q ?x ?y) and (r ?y ?z)
                # if we have two agenda items { (p ?x1 ?y1) and (q ?y1 ?z1)} each of the agenda items
                # will have 2 resolvers, so we wont identify stream A as being the right move
                continue
            [resolver] = resolvers
            if resolver.action and resolver.action.new_input_variables:
                continue
            p0 = successor(p0, resolver)
            break
        else:
            break
    return p0

def extract_from_partial_plan(new_world_state, partial_plan):
    object_stream_map = {o:None for o in partial_plan.bindings}
    independant_streams = set()
    for act in partial_plan.actions:
        if act.stream is None:
            continue
        new_world_state |= act.eff
        for out in act.outputs:
            object_stream_map[out] = act
        if len(act.outputs) == 0:
            independant_streams.add(act)
    missing = {m for (stream_action, m) in partial_plan.agenda}
    return new_world_state, object_stream_map, independant_streams, missing    

def extract_stream_plan(state):
    """Given a search state, return the list of object stream maps needed by each action
    along the path to state. The list contains one dictionary per action."""

    stream_plan = []
    while state is not None:
        objects_created = {k:v for k,v in state.object_stream_map.items() if v is not None}
        stream_plan.insert(0, objects_created)
        state = state.parent
    return stream_plan

def extract_stream_ordering(stream_plan):
    """Given a stream_plan, return a list of stream actions in order that they should be
    computed. The order is determined by the order in which the objects are needed for the
    action plan, modulo a topological sort."""

    all_object_map = {k: v for _, d, _ in stream_plan for k, v in d.items()}

    computed_objects = set()
    stream_ordering = []
    for edge, object_map, independant_streams in stream_plan:
        local_ordering = [(stream_action, edge) for stream_action in independant_streams]
        for object_name in object_map:
            stack = [object_name]

            while stack:
                current_object_name = stack.pop(0)
                if current_object_name in computed_objects:
                    continue
                stream_action = all_object_map.get(current_object_name)
                if stream_action is None:
                    continue

                local_ordering.insert(0, (stream_action, edge))
                for object_name in stream_action.outputs:
                    computed_objects.add(object_name)

                for parent_object in stream_action.inputs:
                    stack.insert(0, parent_object)
        stream_ordering.extend(local_ordering)

    return stream_ordering

@dataclass
class Binding:
    index: int
    stream_plan: list
    mapping: dict

def sample_depth_first(stream_plan, max_steps=10000, verbose=False):
    """Demo sampling a stream plan using a backtracking depth first approach.
    Returns a mapping if one exists, or None if infeasible or timeout. """
    queue = [Binding(0, [s for (s, _) in stream_plan], {})]
    steps = 0
    while queue and steps < max_steps:
        binding = queue.pop(0)
        steps += 1

        stream_action = binding.stream_plan[binding.index]

        input_objects = [binding.mapping.get(var_name) or Object.from_name(var_name) for var_name in stream_action.inputs]
        stream_instance = stream_action.stream.get_instance(input_objects)
    
        [new_stream_result], new_facts = stream_instance.next_results(verbose=verbose)
        output_objects = new_stream_result.output_objects

        new_mapping = binding.mapping.copy()
        new_mapping.update(dict(zip(stream_action.outputs, output_objects)))
        new_binding = Binding(binding.index + 1, binding.stream_plan, new_mapping)

        if len(new_binding.stream_plan) == new_binding.index:
            return new_binding.mapping

        queue.append(new_binding)
        queue.append(binding)
    return None # infeasible or reached step limit


def extract_stream_plan_from_path(path):
    stream_plan = []
    for edge in path:
        stream_map = {k:v for k,v in edge[2].object_stream_map.items() if v is not None}
        stream_plan.append((edge, stream_map, edge[2].independant_streams))
    return stream_plan

import heapq

class PriorityQueue:
    def __init__(self, init=[]):
        self.heap = []
        self.counter = itertools.count()
        for item in init:
            self.push(item, 0)

    def push(self, item, priority):
        heapq.heappush(self.heap, (priority, next(self.counter), item))
    
    def pop(self):
        return heapq.heappop(self.heap)[-1]

    def peep(self):
        return self.heap[0][-1]
    
    def __len__(self):
        return len(self.heap)

import math

def try_a_star(search, cost, heuristic, max_step=10000):
    start_time = datetime.now()
    q = PriorityQueue([search.init])
    closed = []
    expand_count = 0
    evaluate_count = 0
    
    while q and expand_count < max_step:
        state = q.pop()
        expand_count += 1

        if search.test_goal(state):
            av_branching_f = evaluate_count / expand_count
            approx_depth = math.log(evaluate_count) / math.log(av_branching_f)
            print(f'Explored {expand_count}. Evaluated {evaluate_count}')
            print(f"Av. Branching Factor {av_branching_f:.2f}. Approx Depth {approx_depth:.2f}")
            print(f"Time taken: {(datetime.now() - start_time).seconds} seconds")
            return state
        
        state.children = []
        for op, child in search.successors(state):
            child.action = op
            child.parent = state
            state.children.append((op, child))
            if child.unsatisfiable or any(search.test_equal(child, node) for node in closed):
                continue 
            evaluate_count += 1
            child.start_distance = state.start_distance + cost(state, op, child)
            q.push(child, child.start_distance + heuristic(child))

        closed.append(state)

def sample_depth_first_with_costs(stream_ordering, max_steps=10000, verbose=False):
    """Demo sampling a stream plan using a backtracking depth first approach.
    Returns a mapping if one exists, or None if infeasible or timeout. """
    queue = PriorityQueue([Binding(0, [a for a, e in stream_ordering], {})])
    stream_edge_map = {a:e for a, e in stream_ordering}
    edge_stats = DefaultDict(lambda: DefaultDict(lambda: 0))
    steps = 0
    while queue and steps < max_steps:
        binding = queue.pop()
        steps += 1

        stream_action = binding.stream_plan[binding.index]

        input_objects = [binding.mapping.get(var_name) or Object.from_name(var_name) for var_name in stream_action.inputs]
        stream_instance = stream_action.stream.get_instance(input_objects)
    
        if stream_instance.enumerated:
            print(f"Stream instance {stream_action} fully enumerated!. Reseting.")
            stream_instance.reset()

        result = stream_instance.next_results(verbose=verbose)

        edge = stream_edge_map[stream_action]
        edge[2].num_attempts += 1

        if len(result[0]) == 0:
            print(f"Invalid result for {stream_action}: {result}")
            queue.push(binding, (stream_instance.num_calls, len(stream_ordering) - binding.index))
            continue

        edge[2].num_successes += 1
        
        [new_stream_result], new_facts = result
        output_objects = new_stream_result.output_objects

        new_mapping = binding.mapping.copy()
        new_mapping.update(dict(zip(stream_action.outputs, output_objects)))
        new_binding = Binding(binding.index + 1, binding.stream_plan, new_mapping)

        if len(new_binding.stream_plan) == new_binding.index:
            return new_binding.mapping

        queue.push(new_binding, (0, len(stream_ordering) - new_binding.index))
        queue.push(binding, (stream_instance.num_calls, len(stream_ordering) - binding.index))
    return None # infeasible or reached step limit


def repeated_a_star(search, max_steps=1000):

    cost = lambda state, op, child: child.num_successes / child.num_attempts
    heuristic = lambda state: 0

    for _ in range(max_steps):
        goal_state = try_a_star(search, cost, heuristic)
        if goal_state is None:
            print("Could not find feasable action plan!")
            return None
        path = goal_state.get_path()
        stream_plan = extract_stream_plan_from_path(path)
        stream_ordering = extract_stream_ordering(stream_plan)
        object_mapping = sample_depth_first_with_costs(stream_ordering)
        if object_mapping is not None:
            return goal_state.get_actions(), object_mapping, goal_state
        print("Could not find object_mapping, retrying with updated costs")


def try_lpa_star(search, cost, heuristic, max_step=100000):
    # TODO: (1) allow for multiple parents to search state; (2) implement remove() on ProrityQueue
    q = PriorityQueue([search.init])
    closed = []
    expand_count = 0
    evaluate_count = 0
    
    while q and expand_count < max_step:
        state = q.pop()
        expand_count += 1

        if search.test_goal(state):
            av_branching_f = evaluate_count / expand_count
            approx_depth = math.log(evaluate_count) / math.log(av_branching_f)
            print(f'Explored {expand_count}. Evaluated {evaluate_count}')
            print(f"Av. Branching Factor {av_branching_f:.2f}. Approx Depth {approx_depth:.2f}")
            return state
        
        if state.start_distance > state.rhs:
            state.start_distance = state.rhs


        state.children = []
        for op, child in search.successors(state):
            child.action = op
            child.parent = state
            state.children.append((op, child))
            if child.unsatisfiable or any(search.test_equal(child, node) for node in closed):
                continue 
            evaluate_count += 1
            child.start_distance = state.start_distance + cost(state, op, child)
            q.push(child, child.start_distance + heuristic(child))

        closed.append(state)



def replace_axioms(domain):
    
    axioms_dict = {axiom.name: axiom for axiom in domain.axioms}

    def replace_args(c, replacements):
        def _replace_args(c):
            if isinstance(c, Atom) or isinstance(c, NegatedAtom):
                return Atom(c.predicate, [replacements.get(arg) for arg in c.args])
            else:
                _c = copy.deepcopy(c)

                if isinstance(c, UniversalCondition) or isinstance(c, ExistentialCondition):
                    _c.parameters = [replacements.get(p) for p in c.parameters]

                _c.parts = [_replace_args(p) for p in c.parts]

                return _c
                
        return _replace_args(c)

    def _replace_axioms(c):
        if isinstance(c, Atom) or isinstance(c, NegatedAtom):
            if c.predicate in axioms_dict:
                axiom = axioms_dict[c.predicate]
                replacements = defaultdict(lambda a: f"{a}__{c.predicate}")
                for p, a in zip(axiom.parameters, c.args):
                    replacements[p.name] = a
                return replace_args(axiom.condition, replacements)
            return c
        else:
            _c = copy.deepcopy(c)
            _c.parts = [_replace_axioms(p) for p in c.parts]
            return _c

    return [Action(a.name, a.parameters, a.num_external_parameters, _replace_axioms(a.precondition), a.effects, a.cost) for a in domain.actions]


def modified_task_from_domain_problem(domain, problem, add_identical=True):
    task_name, task_domain_name, task_requirements, objects, init, goal, use_metric, problem_pddl = problem

    assert domain.name == task_domain_name
    requirements = pddl.Requirements(sorted(set(domain.requirements.requirements +
                                                task_requirements.requirements)))
    objects = domain.constants + objects
    check_for_duplicates([o.name for o in objects],
        errmsg="error: duplicate object %r",
        finalmsg="please check :constants and :objects definitions")
    init.extend(pddl.Atom(EQ, (obj.name, obj.name)) for obj in objects)
    if add_identical:
        init.extend(get_identical_atoms(objects))
    #print('{} objects and {} atoms'.format(len(objects), len(init)))

    task = pddl.Task(domain.name, task_name, requirements, domain.types, objects,
                     domain.predicates, domain.functions, init, goal,
                     domain.actions, domain.axioms, use_metric)

    return task


def normalize_domain_goal(domain, goal_exp):
    evaluations = []
    problem = get_problem(evaluations, goal_exp, domain, unit_costs=False)
    task = modified_task_from_domain_problem(domain, problem)
    return task



def modified_parse_problem(
    problem,
    stream_info={},
    constraints=None,
    unit_costs=False,
    unit_efforts=False,
    use_unique=False,
):
    # TODO: just return the problem if already written programmatically
    reset_globals() # Prevents use of satisfaction.py
    domain_pddl, constant_map, stream_pddl, stream_map, init, goal = problem

    domain = parse_domain(domain_pddl)  # TODO: normalize here
    # domain = domain_pddl
    if len(domain.types) != 1:
        raise NotImplementedError("Types are not currently supported")
    if unit_costs:
        set_unit_costs(domain)
    if not has_costs(domain):
        # TODO: set effort_weight to 1 if no costs
        print("Warning! All actions have no cost. Recommend setting unit_costs=True")
    obj_from_constant = parse_constants(
        domain, constant_map
    )  # Keep before parse_stream_pddl

    streams = parse_stream_pddl(
        stream_pddl,
        stream_map,
        stream_info=stream_info,
        unit_costs=unit_costs,
        unit_efforts=unit_efforts,
        use_unique=use_unique,
    )
    check_problem(domain, streams, obj_from_constant)

    evaluations = evaluations_from_init(init)
    goal_exp = obj_from_value_expression(goal)

    if isinstance(domain, SimplifiedDomain):
        # assert isinstance(domain, str) # raw PDDL is returned
        _ = {name: Object(value, name=name) for name, value in constant_map.items()}
        return evaluations, goal_exp, domain, streams

    goal_exp = add_plan_constraints(constraints, domain, evaluations, goal_exp)
    parse_goal(goal_exp, domain)  # Just to check that it parses
    normalize_domain_goal(domain, goal_exp)  # TODO: does not normalize goal_exp

    compile_to_exogenous(evaluations, domain, streams)
    return evaluations, goal_exp, domain, streams


if __name__ == '__main__':
    from experiments.blocks_world.run import *
    from pddlstream.algorithms.algorithm import parse_problem

    import argparse

    url = None
    
    # naming scheme: <num_blocks>_<num_blockers>_<maximum_goal_stack_height>_<index>
    # problem_file = 'experiments/blocks_world/data_generation/random/train/1_0_1_40.yaml'
    # problem_file = 'experiments/blocks_world/data_generation/random/train/1_1_1_52.yaml'
    problem_file = 'experiments/blocks_world/data_generation/non_monotonic/train/1_1_1_0.yaml'

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', help='Task description file', default=problem_file, type=str)
    args = parser.parse_args()
    problem_file = args.task

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
    # [pick, move, place, stack, unstack] = domain.actions

    # actions = domain.actions
    actions = replace_axioms(domain)

    search = ActionStreamSearch(init, goal, externals, actions)
    # goal_state = try_bfs(search)
    # path = goal_state.get_path()
    # print(f"Path: {path}\n")
    # stream_plan = extract_stream_plan_from_path(path)
    # stream_ordering = extract_stream_ordering(stream_plan)
    # print(f"Stream ordering: {stream_ordering}\n")
    # object_mapping = sample_depth_first(stream_ordering)
    # print(f"Object mapping: {object_mapping}\n")

    result = repeated_a_star(search)
    if result is not None:
        action_skeleton, object_mapping, _ = result
        actions_str = "\n".join([str(a) for a in action_skeleton])
        print(f"Action Skeleton:\n{actions_str}")
        print(f"\nObject mapping: {object_mapping}\n")