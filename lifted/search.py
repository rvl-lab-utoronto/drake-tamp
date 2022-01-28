from typing import Any
import itertools
from dataclasses import dataclass

import numpy as np

from pddl.conditions import Atom, Conjunction, NegatedAtom
from pddl.actions import PropositionalAction
import pddl.conditions as conditions

from lifted.utils import Identifiers, Unsatisfiable
from lifted.partial import certify, extract_from_partial_plan
REUSE_INITIAL_CERTIFIABLE_OBJECTS = False

def combinations(candidates):
    """Given a dictionary from key (k^j) to a set of possible values D_k^j, yield all the
    tuples [(k^j, v_i^j)] where v_i^j is an element of D_k^j"""
    keys, domains = zip(*candidates.items())
    for combo in itertools.product(*domains):
        yield dict(zip(keys, combo))


def find_applicable_brute_force(
    action, state, allow_missing, object_stream_map={}, filter_precond=True
):
    """Given an action schema and a state, return the list of partially grounded
    operators possible in the given state"""

    # Find all possible assignments of the state.objects to action.parameters
    # such that [action.preconditions - {atom | atom in certified}] <= state
    candidates = {}
    for atom in action.precondition.parts:

        for ground_atom in state:
            if ground_atom.predicate != atom.predicate:
                continue

            if ground_atom.predicate in allow_missing:
                if (not REUSE_INITIAL_CERTIFIABLE_OBJECTS) or any([arg[0] == "?" for arg in ground_atom.args]):
                    continue

            for arg, candidate in zip(atom.args, ground_atom.args):
                if arg not in {x.name for x in action.parameters}:
                    continue

                
                if ground_atom.predicate in allow_missing:
                    assert REUSE_INITIAL_CERTIFIABLE_OBJECTS
                    candidates.setdefault(arg, set()).add('?')

                candidates.setdefault(arg, set()).add(candidate)

    # if not candidates:
    #     assert False, "why am i here"

    # find all the possible versions
    for assignment in combinations(candidates) if candidates else [{}]:
        feasible = True
        for par in action.parameters:
            if (par.name not in assignment) or (assignment[par.name] == '?'):
                assignment[par.name] = Identifiers.next()

        assert isinstance(action.precondition, Conjunction)
        precondition_parts = []
        for atom in action.precondition.parts:
            args = [assignment.get(arg, arg) for arg in atom.args]
            atom = (
                Atom(atom.predicate, args)
                if not atom.negated
                else NegatedAtom(atom.predicate, args)
            )
            precondition_parts.append(atom)
            # grounded positive precondition not in state
            if not atom.negated and atom not in state:
                if atom.predicate not in allow_missing:
                    feasible = False
                    break
                # THIS SAYS THAT A FACT WILL NOT BE ACHIEVABLE
                # IF ALL OF ITS ARGUMENTS ARE NON OPTIMISTIC
                # BUT THIS IS NOT TRUE FOR TEST STREAMS.
                # HAVE TO REVISIT THIS ASSUMPTION ELSEWHERE
                # if all(arg in object_stream_map for arg in atom.args):
                #     feasible = False
                #     break

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

            condition = effect.condition  # TODO: assign the parameters in this
            if effect.parameters or not isinstance(effect.condition, conditions.Truth):
                raise NotImplementedError
            effects.append((condition, atom, effect, assignment.copy()))

        arg_list = [
            assignment.get(par.name, par.name)
            for par in action.parameters[: action.num_external_parameters]
        ]
        name = "(%s %s)" % (action.name, " ".join(arg_list))
        cost = 1

        partial_op = PropositionalAction(
            name, Conjunction(precondition_parts), effects, cost, action, assignment
        )

        # are any of the preconditions missing?
        # if any of the preconditions correspond to non-certifiable 
        # AAAND they include missing args
        # then there's no way this will work

        if filter_precond:
            missing_positive = {
                atom for atom in partial_op.precondition.parts if not atom.negated
            } - set(state)

            # grounded positive precondition not in state
            if any(
                atom.predicate not in allow_missing
                 # THIS CONDITION IS NOT VALID FOR TEST STREAMS
                 # or all(arg in object_stream_map for arg in atom.args)
                for atom in missing_positive
            ):
                assert False

            # negative precondition is positive in state
            negative = {atom for atom in partial_op.precondition.parts if atom.negated}
            if any(atom.negate() in state for atom in negative):
                assert False

        yield partial_op


def apply(action, state):
    """Given an action, and a state, compute the successor state"""
    return (state | {eff[1] for eff in action.add_effects}) - {
        eff[1] for eff in action.del_effects
    }


def objects_from_state(state):
    return {arg for atom in state for arg in atom.args}


@dataclass
class DictionaryWithFallbacks:
    own_keys: dict
    fallbacks: Any  # list of DictionaryWithFallback or None

    def __getitem__(self, key):
        if key in self.own_keys:
            return self.own_keys[key]
        elif self.fallbacks is not None:
            for fb in self.fallbacks:
                ret = fb[key]
                if ret is not None:
                    return ret
        return None


class SearchEdge:
    def __init__(self, source, destination, action):
        self.source = source
        self.destination = self.destination
        self.action = action


class SearchState:
    def __init__(self, state, object_stream_map, unsatisfied, parents=set()):
        self.state = frozenset(state.copy())
        self.object_stream_map = object_stream_map.copy()
        self.unsatisfied = unsatisfied
        self.unsatisfiable = False
        self.children = set()
        self.parents = parents
        self.start_distance = 0
        self.rhs = 0
        self.num_attempts = 1
        self.num_successes = 1
        self.expanded = False
        self.object_computation_graph_keys = {}

        self.__full_stream_map = None

    def __eq__(self, other):
        return (
            self.state == other.state
            and self.object_stream_map == other.object_stream_map
        )
        # compare based on fluent predicates and 
        # computation graph in corresponding objects
        return False

    def __repr__(self):
        return str((self.state, self.object_stream_map, self.unsatisfied))

    def __hash__(self):
        return hash((self.state,))

    @property
    def full_stream_map(self):
        if self.__full_stream_map is None:
            self.__full_stream_map = DictionaryWithFallbacks(
                {k: v for k, v in self.object_stream_map.items() if v is not None},
                [parent.full_stream_map for _, parent in self.parents]
                if len(self.parents) > 0
                else None,
            )
        return self.__full_stream_map

    def get_object_computation_graph_key(self, obj):
        if obj in self.object_computation_graph_keys:
            return self.object_computation_graph_keys[obj]

        stream_action = self.full_stream_map[obj]
        if stream_action is None:
            return obj
        if self.object_stream_map[obj] is None:
            # TODO: this is wasteful. Try to identify the object somehow.
            # return self.parent.get_constraint_graph_key(obj)
            return obj
        edges = frozenset({
            (
                self.get_object_computation_graph_key(input_object),
                stream_action.inputs.index(input_object) if stream_action.stream.name != 'all' else None,
                stream_action.stream.name,
                stream_action.outputs.index(obj),
            ) for input_object in stream_action.inputs
        })

        self.object_computation_graph_keys[obj] = edges
        return edges

    # def get_object_computation_graph_key(self, obj):
    #     if obj in self.object_computation_graph_keys:
    #         return self.object_computation_graph_keys[obj]

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

    #         input_objs = stream_action.inputs + tuple(
    #             sorted(list(objects_from_state(stream_action.fluent_facts)))
    #         )
    #         # TODO add fluent objects also to this tuple
    #         for parent_obj in input_objs:
    #             stack.insert(0, parent_obj)
    #             if obj not in anon:
    #                 anon[obj] = f"?{next(counter)}"
    #             if (
    #                 parent_obj not in anon
    #                 and self.full_stream_map[parent_obj] is not None
    #             ):
    #                 anon[parent_obj] = f"?{next(counter)}"
    #             edges.add(
    #                 (
    #                     anon.get(parent_obj, parent_obj),
    #                     stream_action.stream.name,
    #                     stream_action.outputs.index(obj),
    #                     anon[obj],
    #                 )
    #             )

    #     self.object_computation_graph_keys[obj] = frozenset(edges)
    #     return frozenset(edges)

    def get_shortest_path_to_start(self):
        path = []
        node = self
        while node is not None:
            msd = np.inf
            mp = None
            a = None
            for action, parent in node.parents:
                if parent.start_distance < msd:
                    msd = parent.start_distance
                    mp = parent
                    a = action
            path.insert(0, (mp, a, node))
            node = mp
        return path[1:]


def check_cg_equivalence(cg1, cg2):
    if cg1 == cg2:
        return True
    if len(cg1) == 1 and len(cg2) == 1 and list(cg1)[0][:-1] == list(cg2)[0][:-1]:
        return True

    return False


class ActionStreamSearch:
    def __init__(self, init, goal, externals, actions):
        self.init_objects = {o for o in objects_from_state(init)}
        self.init = SearchState(init, {o: None for o in self.init_objects}, set())
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
        sub = {o: o for o in self.init_objects}
        for a, b in zip(f1, f2):
            if a.predicate != b.predicate:
                return False
            if a == b:
                continue
            for o1, o2 in zip(a.args, b.args):
                if o1 != o2:
                    if o1.startswith("?") and o2.startswith("?"):
                        cg1 = s1.get_object_computation_graph_key(o1)
                        cg2 = s2.get_object_computation_graph_key(o2)
                        if check_cg_equivalence(cg1, cg2):
                            # if s1.full_stream_map[o1] == s2.full_stream_map[o2]:
                            continue
                    return False
                assert o1 == o2
                if sub.setdefault(o1, o2) != o2:
                    return False

        return True

    def test_goal(self, state):
        return self.goal <= state.state


    def successors(self, state):
        if state.expanded:
            return list(state.children)
        else:
            successors = []
            for action in self.actions:
                ops = find_applicable_brute_force(
                    action,
                    state.state | state.unsatisfied,
                    self.streams_by_predicate,
                    state.object_stream_map,
                )
                for op in ops:
                    new_world_state = apply(op, state.state)
                    missing_positive = {
                        atom for atom in op.precondition.parts if not atom.negated
                    } - set(state.state)
                    # this does the partial order plan
                    try:
                        missing = missing_positive | state.unsatisfied
                        partial_plan = certify(
                            state.state,
                            state.object_stream_map,
                            missing,
                            self.streams_by_predicate,
                        )
                        # this extracts the important bits from it 
                        # (i.e. description of the state)
                        (
                            new_world_state,
                            object_stream_map,
                            new_missing,
                        ) = extract_from_partial_plan(
                            state, missing, new_world_state, partial_plan
                        )
                        op_state = SearchState(
                            new_world_state,
                            object_stream_map,
                            new_missing,
                            parents={(op, state)},
                        )
                        # state.children.add((op, op_state))
                        successors.append((op, op_state))
                    except Unsatisfiable:
                        continue
            state.expanded = True
            
            return successors
