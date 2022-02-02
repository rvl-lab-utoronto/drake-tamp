from typing import Any
import itertools
from dataclasses import dataclass
import copy

import numpy as np
from lifted.utils import replace_objects_in_condition

from pddl.conditions import Atom, Conjunction, NegatedAtom
import pddl.conditions as conditions

from lifted.utils import (
    Identifiers,
    Unsatisfiable,
    PropositionalAction,
    replace_objects_in_action,
    OPT_PREFIX,
    PredicateObject
)
from lifted.partial import certify, extract_from_partial_plan
REUSE_INITIAL_CERTIFIABLE_OBJECTS = True

def combinations(candidates):
    """Given a dictionary from key (k^j) to a set of possible values D_k^j, yield all the
    tuples [(k^j, v_i^j)] where v_i^j is an element of D_k^j"""
    keys, domains = zip(*candidates.items())
    for combo in itertools.product(*domains):
        yield dict(zip(keys, combo))

def find_assignments_brute_force(action, state, allow_missing):
    candidates = {}
    params = {x.name for x in action.parameters}
    for atom in action.precondition.parts:

        for ground_atom in state:
            if ground_atom.predicate != atom.predicate:
                continue

            if ground_atom.predicate in allow_missing:
                if (not REUSE_INITIAL_CERTIFIABLE_OBJECTS) or any([arg[0] == OPT_PREFIX for arg in ground_atom.args]):
                    continue
            if any(p not in params and p != arg for p,arg in zip(atom.args, ground_atom.args)):
                continue
            for arg, candidate in zip(atom.args, ground_atom.args):
                if arg not in params:
                    continue
                
                if ground_atom.predicate in allow_missing:
                    assert REUSE_INITIAL_CERTIFIABLE_OBJECTS
                    candidates.setdefault(arg, set()).add("?")

                candidates.setdefault(arg, set()).add(candidate)
    return list(combinations(candidates)) if candidates else [{}]
        
def find_applicable_brute_force(
    action, state, allow_missing, object_stream_map={}, filter_precond=True
):
    """Given an action schema and a state, return the list of partially grounded
    operators possible in the given state"""
    # Find all possible assignments of the state.objects to action.parameters
    # such that [action.preconditions - {atom | atom in certified}] <= state
    params = {x.name for x in action.parameters}
    for atom in action.precondition.parts:
        if not any(arg in params for arg in atom.args) and (
            (not atom.negated and atom not in state) or
            (atom.negated and atom in state)
        ):
            return

    # find all the possible versions
    for assignment in find_assignments_brute_force(action, state, allow_missing):
    
        assignment = {
            k: PredicateObject(copy.deepcopy(v.data)) if isinstance(v, PredicateObject) else v
            for k, v in assignment.items()
        }

        feasible = True
        for par in action.parameters:
            if (par.name not in assignment) or (assignment[par.name] == "?"):
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
        cost = 1

        partial_op = PropositionalAction(
            Conjunction(precondition_parts), effects, cost, action, assignment
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

class SearchState:
    def __init__(self, state, object_stream_map, unsatisfied, id_key, parents=set()):
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
        self.id_key = id_key

        self.__full_stream_map = None

    def __repr__(self):
        return str((self.state, self.object_stream_map, self.unsatisfied))

    def __eq__(self, other):
        return self.id_key == other.id_key

    def __hash__(self):
        return hash(self.id_key)

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


class ActionStreamSearch:
    def __init__(self, init, goal, externals, actions):
        self.init_objects = {o for o in objects_from_state(init)}
        self.goal = goal
        self.externals = externals
        self.actions = actions

        self.id_cg_map = {}
        self.cg_id_map = {}

        self.streams_by_predicate = {}
        for stream in externals:
            for fact in stream.certified:
                self.streams_by_predicate.setdefault(fact[0], set()).add(stream)

        self.fluent_predicates = set()
        for action in actions:
            for effect in action.effects:
                self.fluent_predicates.add(effect.literal.predicate)

        id_key = tuple(sorted(f for f in init if f.predicate in self.fluent_predicates))
        self.init = SearchState(
            init, {o: None for o in self.init_objects}, set(), id_key
        )

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
                            state,
                            missing,
                            new_world_state,
                            partial_plan,
                            self.cg_id_map,
                        )

                        temp_object_mapping = {
                            x: "?"
                            for f in new_world_state
                            for x in f.args
                            if x not in object_stream_map
                        }
                        temp_world_state = set(
                            replace_objects_in_condition(f, temp_object_mapping)
                            for f in new_world_state
                        )
                        state_id_key = tuple(
                            sorted(
                                f
                                for f in temp_world_state
                                if f.predicate in self.fluent_predicates
                            )
                        )

                        op_state = SearchState(
                            new_world_state,
                            object_stream_map,
                            new_missing,
                            state_id_key,
                            parents={(op, state)},
                        )

                        successors.append((op, op_state))
                    except Unsatisfiable:
                        continue
            state.expanded = True
            
            return successors
