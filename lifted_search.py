
#%%
from functools import partial
from typing import Any
from pddlstream.language.conversion import fact_from_evaluation, objects_from_evaluations
import sys
from collections import defaultdict, namedtuple
import math
import itertools

from dataclasses import dataclass, field
from pddlstream.language.stream import Stream, StreamResult
from pddlstream.language.object import Object

sys.path.insert(0, '/home/mohammed/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/')
sys.path.insert(0, '/home/atharv/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/')
import pddl
from pddl.conditions import Atom, Conjunction, NegatedAtom
from pddl.actions import PropositionalAction, Action
from pddl.effects import Effect
import pddl.conditions as conditions

from pddl.pddl_types import TypedObject
import pddl.conditions as conditions
import time
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
                if arg not in {x.name for x in action.parameters}:
                    continue
                candidates.setdefault(arg, set()).add(candidate)
    # if not candidates:
    #     assert False, "why am i here"
    #     yield bind_action(action, {})
    #     return
    # find all the possible versions
    for assignment in combinations(candidates) if candidates else [{}]:
        feasible = True
        for par in action.parameters:
            if par.name not in assignment:
                assignment[par.name] = Identifiers.next()

        assert isinstance(action.precondition, Conjunction)
        precondition_parts = []
        for atom in action.precondition.parts:
            args = [assignment.get(arg, arg) for arg in atom.args]
            atom = Atom(atom.predicate, args) if not atom.negated else NegatedAtom(atom.predicate, args)
            precondition_parts.append(atom)
            # grounded positive precondition not in state
            if (not atom.negated and atom not in state):
                if(atom.predicate not in allow_missing):
                    feasible = False
                    break
                if all(arg in object_stream_map for arg in atom.args):
                    feasible = False
                    break


    

            if atom.negated and atom.negate() in state:
                feasible = False
                break
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


@dataclass
class DictionaryWithFallback:
    own_keys: dict
    fallback: Any # another DictionaryWithFallback or None

    def __getitem__(self, key):
        if key in self.own_keys:
            return self.own_keys[key]
        elif self.fallback is not None:
            return self.fallback[key]
        return None

class SearchState:
    def __init__(self, state, object_stream_map, unsatisfied, parent=None):
        self.state = state.copy()
        self.object_stream_map = object_stream_map.copy()
        self.unsatisfied = unsatisfied
        self.unsatisfiable = False
        self.children = []
        self.parent = parent
        self.action = None
        self.start_distance = 0
        self.rhs = 0
        self.num_attempts = 1
        self.num_successes = 1
        self.expanded = False

        self.__full_stream_map = None

    def __eq__(self, other):
        # compare based on fluent predicates and computation graph in corresponding objects
        return False
    
    def __repr__(self):
        return str((self.state, self.object_stream_map, self.unsatisfied))

    @property
    def full_stream_map(self):
        if self.__full_stream_map is None:
            self.__full_stream_map = DictionaryWithFallback({k:v for k,v in self.object_stream_map.items() if v is not None}, self.parent.full_stream_map if self.parent is not None else None)
        return self.__full_stream_map
            
    # def get_object_computation_graph_key(self, obj):
    #     counter = itertools.count()
    #     stack = [obj]
    #     edges = set()
    #     anon = {}
    #     while stack:
    #         obj = stack.pop(0)
    #         stream_action = self.full_stream_map[obj]   
    #         if stream_action is None:
    #             edges.add((None, None, None, obj))
    #             continue

    #         input_objs = stream_action.inputs  + tuple(sorted(list(objects_from_state(stream_action.fluent_facts))))
    #         # TODO add fluent objects also to this tuple
    #         for parent_obj in input_objs:
    #             stack.insert(0, parent_obj)
    #             if obj not in anon:
    #                 anon[obj] = f'?{next(counter)}'
    #             if parent_obj not in anon and self.full_stream_map[parent_obj] is not None:
    #                 anon[parent_obj] = f'?{next(counter)}'
    #             edges.add((anon.get(parent_obj, parent_obj), stream_action.stream.name, stream_action.outputs.index(obj), anon[obj]))
            
    #     return frozenset(edges)
    def get_constraint_graph_key(self, obj):
        stream_action = self.full_stream_map[obj]
        if stream_action is None:
            return obj
        if self.object_stream_map[obj] is None:
            return self.parent.get_constraint_graph_key(obj)
        
        

        return stream_action

    def get_object_computation_graph_key(self, obj):
        stream_action = self.full_stream_map[obj]
        if stream_action is None:
            return obj
        if self.object_stream_map[obj] is None:
            # TODO: this is wasteful. Try to identify the object somehow.
            # return self.parent.get_constraint_graph_key(obj)
            return obj
        return frozenset({
            (
                self.get_object_computation_graph_key(input_object),
                stream_action.inputs.index(input_object) if stream_action.stream.name != 'all' else None,
                stream_action.stream.name,
                stream_action.outputs.index(obj),
            ) for input_object in stream_action.inputs
        })

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


def check_cg_equivalence(cg1, cg2):
    if cg1 == cg2:
        return True
    if len(cg1) == 1 and len(cg2) == 1 and list(cg1)[0][:-1] == list(cg2)[0][:-1]:
        return True

    return False
    

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
                if o1 != o2:
                    if o1.startswith('?') and o2.startswith('?'):
                        cg1 = s1.get_object_computation_graph_key(o1)
                        cg2 = s2.get_object_computation_graph_key(o2)
                        if check_cg_equivalence(cg1, cg2):
                            continue
                    return False
                if sub.setdefault(o1, o2) != o2:
                    return False

        return True
    
    def test_goal(self, state):
        return self.goal <= state.state 

    def action_successor(self, state, action):
        new_state = apply(action, state.state)
        missing_positive = {atom for atom in action.precondition.parts if not atom.negated} - set(state.state)
        return SearchState(new_state, state.object_stream_map, missing_positive | state.unsatisfied)
    
    def successors(self, state):
        if state.expanded:
            for child in state.children:
                yield child
        else:
            state.children = []
            for action in self.actions:
                ops = find_applicable_brute_force(action, state.state | state.unsatisfied, self.streams_by_predicate, state.object_stream_map)
                for op in ops:
                    new_world_state = apply(op, state.state)
                    missing_positive = {atom for atom in op.precondition.parts if not atom.negated} - set(state.state)
                    # this does the partial order plan
                    try:
                        missing = missing_positive | state.unsatisfied
                        partial_plan = certify(state.state, state.object_stream_map, missing, self.streams_by_predicate)
                        # this extracts the important bits from it (i.e. description of the state)
                        new_world_state, object_stream_map, new_missing = extract_from_partial_plan(state, missing, new_world_state, partial_plan)
                        op_state = SearchState(new_world_state, object_stream_map, new_missing, parent=state)
                        state.children.append((op, op_state))
                        yield (op, op_state)
                    except Unsatisfiable:
                        continue
            state.expanded = True

                
    
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
    fluent_facts: tuple = field(default_factory=tuple)
    id: int = field(default_factory=lambda: next(id_provider))
    pre: set = field(default_factory=set)
    eff: set = field(default_factory=set)

    def __hash__(self):
        return self.id
    
    def __repr__(self):
        return f'{self.stream.name}({self.inputs})->({self.outputs}), fluents={self.fluent_facts}'

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

        # handle fluents
        if stream.is_fluent:
            fluent_facts = compute_fluent_facts(partial_plan, stream)
        else:
            fluent_facts = tuple()
        if stream.is_test:
            outputs = (Identifiers.next(),) # just to include it in object stream map.
        action = StreamAction(stream, inputs, outputs, fluent_facts, pre=pre, eff=eff)
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

            for missing_precond in fluent_facts:
                for existing_action in partial_plan.actions:
                    if missing_precond in existing_action.eff:
                        assert existing_action.id == -1 # has to be part of the state!
                        links.append((existing_action, missing_precond, action))

        yield Resolver(action, links=links, binding=binding)

def compute_fluent_facts(partial_plan, external):
    state = [action.eff for action in partial_plan.actions if action.id == -1][0]
    fluent = set()
    for f in state:
        if f.predicate in external.fluents:
            fluent.add(f)
    return tuple(fluent)

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

    init_action = StreamAction(id=-1, eff=state)
    goal_action = StreamAction(id=-2, pre=missing)
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

def extract_from_partial_plan(old_world_state, old_missing, new_world_state, partial_plan):
    """Extract the successor state from the old world state and the partial plan.
    That involves updating the object stream map, the set of facts that are yet to be certified,
    and the new logical state based on the stream actions in the partial plan.

    Edit: As of today (Dec 1) objects are not added to the object stream map until their CG is
    fully determined.
    """
    new_objects = set()
    for act in partial_plan.actions:
        if act.stream is not None:
            for out in act.outputs:
                new_objects.add(out)

    object_stream_map = {o:None for o in old_world_state.object_stream_map}
    missing = old_missing.copy()
    produced = set()
    used = set()
    for act in partial_plan.actions:
        if act.stream is None:
            continue
        # object's CG is determined if all inputs are produced now, or previously
        if any(parent_obj not in new_objects and parent_obj not in old_world_state.object_stream_map for parent_obj in act.inputs):
            continue
        if not act.stream.is_fluent:
            new_world_state |= act.eff
        
        for out in act.outputs:
            object_stream_map[out] = act
            produced.add(out)
        for out in act.inputs:
            used.add(out)
        for e in act.eff:
            if e in missing:
                missing.remove(e)
            else:
                # print('Not missing!', e)
                pass
    
    placeholder = Identifiers.next()
    object_stream_map[placeholder] = StreamAction(
        DummyStream('all'),
        inputs=tuple(produced - used), # i need this to be ordered in order for the cg key to work. But i have nothing with which to base the order on.
        outputs=(placeholder,)
    )

    return new_world_state, object_stream_map, missing    

def extract_stream_plan(state):
    """Given a search state, return the list of object stream maps needed by each action
    along the path to state. The list contains one dictionary per action."""

    stream_plan = []
    while state is not None:
        objects_created = {k:v for k,v in state.object_stream_map.items() if v is not None}
        stream_plan.insert(0, objects_created)
        state = state.parent
    return stream_plan

def get_stream_action_edges(stream_actions):
    input_obj_to_streams = defaultdict(set)
    for action in stream_actions:
        for obj in action.inputs:
            input_obj_to_streams.setdefault(obj, set()).add(action)
    
    edges = []
    for action in stream_actions:
        children = set()
        for obj in action.outputs:
            children = children | input_obj_to_streams[obj]
        for child in children:
            edges.append((action, child))

    return edges

def extract_stream_ordering(stream_plan):
    """Given a stream_plan, return a list of stream actions in order that they should be
    computed. The order is determined by the order in which the objects are needed for the
    action plan, modulo a topological sort.
    
    Edit: Assumes each object_map depends only on itself and predecessors."""
    computed_objects = set(stream_plan[0][0][2].object_stream_map)

    def topological_sort(stream_actions):
        incoming_edges = {}
        ready = set()
        for stream_action in stream_actions:
            missing = set(stream_action.inputs) - computed_objects
            if missing:
                incoming_edges[stream_action] = missing
            else:
                ready.add(stream_action)

        result = []
        while ready:
            stream_action = ready.pop()
            result.append(stream_action)
            for out in stream_action.outputs:
                computed_objects.add(out)
            for candidate in list(incoming_edges):
                missing = incoming_edges[candidate] - computed_objects
                if missing:
                    incoming_edges[candidate] = missing
                else:
                    del incoming_edges[candidate]
                    ready.add(candidate)

        assert not incoming_edges, "Something went wrong. Either the CG has a cycle, or depends on missing (or future) step."
        return result

    stream_ordering = []
    for edge, object_map in stream_plan:
        stream_actions = {action for action in object_map.values()}

        # if stream_actions:
        #     final_node = StreamAction(
        #         DummyStream('and'),
        #         inputs=tuple(),
        #         outputs=tuple()
        #     )
        #     used = set({out for s in stream_actions for out in s.inputs})
        #     inputs = []
        #     for action in stream_actions:
        #         unused = set(action.outputs) - used
        #         inputs.extend(list(unused))
        #     final_node.inputs = tuple(inputs)
        #     stream_actions.add(final_node)

        local_ordering = topological_sort(stream_actions)
        stream_ordering.extend(local_ordering)
    return stream_ordering

@dataclass
class DummyStream:
    name: str
    outputs = tuple()
    enumerated = False
    is_test = True
    
    @property
    def external(self):
        return self

    def get_instance(self, inputs, fluent_facts=tuple()): 
        return self

    def next_results(self, verbose=False): 
        return [StreamResult(self, tuple())], []

@dataclass
class Binding:
    index: int
    stream_plan: list
    mapping: dict

def sample_depth_first(stream_plan, max_steps=10000):
    """Demo sampling a stream plan using a backtracking depth first approach.
    Returns a mapping if one exists, or None if infeasible or timeout. """
    queue = [Binding(0, stream_plan, {})]
    steps = 0
    while queue and steps < max_steps:
        binding = queue.pop(0)
        steps += 1

        stream_action = binding.stream_plan[binding.index]

        input_objects = [binding.mapping.get(var_name) or Object.from_name(var_name) for var_name in stream_action.inputs]
        fluent_facts = [(f.predicate, ) + tuple(binding.mapping.get(var_name) or Object.from_name(var_name) for var_name in f.args) for f in stream_action.fluent_facts]
        stream_instance = stream_action.stream.get_instance(input_objects, fluent_facts=fluent_facts)
        if stream_instance.enumerated:
            continue
        results, new_facts = stream_instance.next_results(verbose=True)
        if not results:
            continue
        [new_stream_result] = results
        output_objects = new_stream_result.output_objects

        new_mapping = binding.mapping.copy()
        new_mapping.update(dict(zip(stream_action.outputs, output_objects)))
        new_binding = Binding(binding.index + 1, binding.stream_plan, new_mapping)

        if len(new_binding.stream_plan) == new_binding.index:
            return new_binding.mapping

        queue.append(new_binding)
        queue.append(binding)
    return None # infeasible or reached step limit

# DummyStream = namedtuple('DummyStream', ['name'])
def ancestral_sampling(stream_ordering, objects_from_name=None):
    if objects_from_name is None:
        objects_from_name = Object._obj_from_name
    nodes = stream_ordering
    edges = get_stream_action_edges(stream_ordering)
    final_node = StreamAction(
        DummyStream('FINAL'),
        inputs=tuple(obj for stream_action in nodes for obj in stream_action.outputs),
        outputs=tuple()
    )
    start_node = StreamAction(
        DummyStream('START'),
        inputs=tuple(),
        outputs=tuple()
    )
    children = {
    }
    for parent, child in edges:
        children.setdefault(parent, set()).add(child)
    for node in nodes:
        children.setdefault(node, set()).add(final_node)
        children.setdefault(start_node, set()).add(node)
    stats = {
        node: 0
        for node in nodes
    }
    produced = dict()
    queue = [Binding(0, [start_node], {})]
    while queue:
        binding = queue.pop(0)
        stream_action = binding.stream_plan[0]
        if stream_action not in [start_node, final_node]:   
            input_objects = [produced.get(var_name) or objects_from_name[var_name] for var_name in stream_action.inputs]
            fluent_facts = [(f.predicate, ) + tuple(produced.get(var_name) or objects_from_name[var_name] for var_name in f.args) for f in stream_action.fluent_facts]
            stream_instance = stream_action.stream.get_instance(input_objects, fluent_facts=fluent_facts)
            if stream_instance.enumerated:
                continue
            results, new_facts = stream_instance.next_results(verbose=False)
            if not results:
                continue
            [new_stream_result] = results
            output_objects = new_stream_result.output_objects
            if stream_action.stream.is_test:
                output_objects = (True,)
            newly_produced = dict(zip(stream_action.outputs, output_objects))
            for obj in newly_produced:
                produced[obj] = newly_produced[obj]
            # new_mapping = binding.mapping.copy()
            # new_mapping.update(newly_produced)
    
        else:
            # new_mapping = binding.mapping
            pass

        stats[stream_action] = stats.get(stream_action, 0) + 1
        for child in children.get(stream_action, []):
            input_objects = list(child.inputs) + [var_name for f in child.fluent_facts for var_name in f.args]
            if all(obj in produced or obj in objects_from_name for obj in input_objects):
                new_binding = Binding(binding.index + 1, [child], {})
                queue.append(new_binding)
    return produced, stats.get(final_node, 0)

def ancestral_sample_with_costs(stream_ordering, final_state, stats={}, max_steps=30, verbose=False):
    to_produce = set({out for s in stream_ordering for out in s.outputs})
    for i in range(max_steps):
        produced, done = ancestral_sampling(stream_ordering)

        for obj in to_produce:
            cg_key = final_state.get_object_computation_graph_key(obj)
            cg_stats = stats.setdefault(cg_key, {'num_attempts': 0., 'num_successes': 0.})
            cg_stats['num_attempts'] += 1
            if obj in produced:
                cg_stats['num_successes'] += 1

        if done:
            return produced
    return None

def ancestral_sampling_by_edge(stream_plan, final_state, stats, max_steps=30):
    (_,_,initial_state), _ = stream_plan[0]
    objects = {k:v for k,v in Object._obj_from_name.items() if k in initial_state.object_stream_map}
    i = 0
    particles = [
        [objects]
    ]
    while i < len(stream_plan):
        (_, _, state), step = stream_plan[i]

        if step:
            to_produce = set({out for s in step for out in s.outputs})
            step_particles = []
            particles.append(step_particles)
            for j in range(max_steps):
                prev_particle = particles[i][j % len(particles[i])]

                new_objects, success = ancestral_sampling(step, prev_particle)

                for obj in to_produce:
                    cg_key = state.get_object_computation_graph_key(obj)
                    cg_stats = stats.setdefault(cg_key, {'num_attempts': 0., 'num_successes': 0.})
                    cg_stats['num_attempts'] += 1
                    if obj in new_objects:
                        cg_stats['num_successes'] += 1


                if success:
                    step_particles.append(dict(**prev_particle, **new_objects))
            if len(step_particles) == 0:
                break
        else:
            particles.append(particles[-1])
        i += 1

    
    print([len(p) for p in particles])
    return particles[-1][0] if len(stream_plan) + 1 == len(particles) and particles[-1] else None

def sample_depth_first_with_costs(stream_ordering, final_state, stats={}, max_steps=None, verbose=False):
    """Demo sampling a stream plan using a backtracking depth first approach.
    Returns a mapping if one exists, or None if infeasible or timeout. """
    if max_steps is None:
        max_steps = len(stream_ordering)*3
    queue = PriorityQueue([Binding(0, stream_ordering, {})])
    steps = 0
    while queue and steps < max_steps:
        binding = queue.pop()
        steps += 1

        stream_action = binding.stream_plan[binding.index]
  
        input_objects = [binding.mapping.get(var_name) or Object.from_name(var_name) for var_name in stream_action.inputs]
        fluent_facts = [(f.predicate, ) + tuple(binding.mapping.get(var_name) or Object.from_name(var_name) for var_name in f.args) for f in stream_action.fluent_facts]
        stream_instance = stream_action.stream.get_instance(input_objects, fluent_facts=fluent_facts)

        if stream_instance.enumerated:
            continue

        result = stream_instance.next_results(verbose=verbose)

        output_cg_keys = [final_state.get_object_computation_graph_key(obj) for obj in stream_action.outputs]
        for cg_key in output_cg_keys:
            cg_stats = stats.setdefault(cg_key, {'num_attempts': 0., 'num_successes': 0.})
            cg_stats['num_attempts'] += 1

        if len(result[0]) == 0:
            print(f"Invalid result for {stream_action}: {result}")
            # queue.push(binding, (stream_instance.num_calls, len(stream_ordering) - binding.index))
            continue

        for cg_key in output_cg_keys:
            cg_stats = stats[cg_key]
            cg_stats['num_successes'] += 1        
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


def extract_stream_plan_from_path(path):
    stream_plan = []
    for edge in path:
        stream_map = {k:v for k,v in edge[2].object_stream_map.items() if v is not None}
        stream_plan.append((edge, stream_map))
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

def try_a_star(search, cost, heuristic, max_step=10000):
    start_time = time.time()
    q = PriorityQueue([search.init])
    closed = []
    expand_count = 0
    evaluate_count = 0
    found = False
    
    while q and expand_count < max_step:
        state = q.pop()
        expand_count += 1

        if search.test_goal(state):
            found = True
            break

        for op, child in search.successors(state):
            child.action = op
            child.parent = state
            if child.unsatisfiable: #or any(search.test_equal(child, node) for node in closed):
                continue
            evaluate_count += 1
            child.start_distance = state.start_distance + cost(state, op, child)
            q.push(child, child.start_distance + heuristic(child, search.goal))

        closed.append(state)

    av_branching_f = evaluate_count / expand_count
    approx_depth = math.log(1e-6 + evaluate_count) / math.log(1e-6 + av_branching_f)
    print(f'Explored {expand_count}. Evaluated {evaluate_count}')
    print(f"Av. Branching Factor {av_branching_f:.2f}. Approx Depth {approx_depth:.2f}")
    print(f"Time taken: {(time.time() - start_time)} seconds")
    print(f"Solution cost: {state.start_distance}")

    if found:
        return state
    else:
        return None

def repeated_a_star(search, stats={}, heuristic=None, max_steps=1000):

    # cost = lambda state, op, child: 1 / (child.num_successes / child.num_attempts)
    def cost(state, op, child, verbose=False):
        included = set()
        c = 0
        for obj in child.object_stream_map:
            stream_action = child.object_stream_map[obj]
            if stream_action is not None:
                if stream_action in included:
                    continue
                cg_key = child.get_object_computation_graph_key(obj)
                if cg_key in stats:
                    s = stats[cg_key]
                    comp_cost = ((s['num_successes'] + 1) / (s['num_attempts'] + 1))**-1
                    c += comp_cost

                else:
                    comp_cost = 1
                    c += comp_cost
                if verbose:
                    print(stream_action, comp_cost)
                included.add(stream_action)
        return max(1, c)

    def _heuristic(state, goal):
        actions = state.get_actions()[1:]
        if len(actions) >= 2:
            if 'move' in actions[-1].name and 'move' in actions[-2].name:
                return np.inf
        return len(goal - state.state)*8
    if heuristic is None:
        heuristic = _heuristic

    for _ in range(max_steps):
        goal_state = try_a_star(search, cost, heuristic)
        if goal_state is None:
            print("Could not find feasable action plan!")
            break
        path = goal_state.get_path()
        c = 0
        for idx, i in enumerate(path):
            print(idx, i[1])
            a = cost(*i, verbose=True)
            print('action cost:', a)
            print('cum cost:', a+c)
            c += a

        action_skeleton = goal_state.get_actions()
        actions_str = "\n".join([str(a) for a in action_skeleton])
        print(f"Action Skeleton:\n{actions_str}")
        
        stream_plan = extract_stream_plan_from_path(path)
        stream_plan = [(edge, list({action for action in object_map.values()})) for (edge, object_map) in stream_plan]
        # stream_ordering = extract_stream_ordering(stream_plan)
        # if not stream_ordering:
        #     break
        object_mapping = ancestral_sampling_by_edge(stream_plan, goal_state, stats, max_steps=100)
        if object_mapping is not None:
            break
        c = 0
        for idx, i in enumerate(path):
            print(idx, i[1])
            a = cost(*i, verbose=True)
            print('action cost:', a)
            print('cum cost:', a+c)
            c += a
        print("Could not find object_mapping, retrying with updated costs")
    if goal_state is not None:
        return goal_state.get_actions(), object_mapping, goal_state

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


if __name__ == '__main__':
    from experiments.blocks_world_noaxioms.run import *
    from pddlstream.algorithms.algorithm import parse_problem
    import argparse

    url = None
    # url = 'tcp://127.0.0.1:6000'

    # naming scheme: <num_blocks>_<num_blockers>_<maximum_goal_stack_height>_<index>
    # problem_file = 'experiments/blocks_world/data_generation/random/train/1_0_1_40.yaml'
    # problem_file = 'experiments/blocks_world/data_generation/random/train/1_1_1_52.yaml'
    problem_file = 'experiments/blocks_world/data_generation/non_monotonic/train/1_1_1_0.yaml'
    problem_file = 'experiments/blocks_world/data_generation/non_monotonic/train/2_2_1_33.yaml'
    # problem_file = 'experiments/blocks_world/data_generation/non_monotonic/train/1_1_1_0.yaml'

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', help='Task description file', default=problem_file, type=str)
    parser.add_argument('-s', '--save-cgstats-path', help='Path to save a json of CG stats.', default=None, type=str)
    parser.add_argument('-p', '--profile', help='Path to save a profile.', default=None, type=str)

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
    [pick, move, place, stack, unstack] = domain.actions

    search = ActionStreamSearch(init, goal, externals, domain.actions)

    # goal_state = try_bfs(search)
    # action_skeleton = goal_state.get_actions()
    # stream_plan = extract_stream_plan_from_path(goal_state.get_path())
    # stream_ordering = extract_stream_ordering(stream_plan)            
    # object_mapping = sample_depth_first_with_costs(stream_ordering)
    # actions_str = "\n".join([str(a) for a in action_skeleton])
    # print(f"Action Skeleton:\n{actions_str}")
    # print(f"\nObject mapping: {object_mapping}\n") 

    if args.profile:
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
    start_time = time.time()
    try:
        stats = {}
        result = repeated_a_star(search, stats=stats)
        if result is not None:
            action_skeleton, object_mapping, _ = result
            actions_str = "\n".join([str(a) for a in action_skeleton])
            print(f"Action Skeleton:\n{actions_str}")
            print(f"\nObject mapping: {object_mapping}\n") 
        if args.save_cgstats_path:
            with open(args.save_cgstats_path, 'w') as f:
                json.dump(list((tuple(cg), v) for (cg,v) in stats.items()), f)
    except KeyboardInterrupt:
        print('KeyboardInterrupt while searching.')
    print(f'Total time: {(time.time() - start_time):.4f}')
    if args.profile:
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.print_stats()
            ps.dump_stats(args.profile)   

