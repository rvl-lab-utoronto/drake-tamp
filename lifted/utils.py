import heapq
import itertools
import collections

from pddl.conditions import Atom, NegatedAtom, Conjunction

OPT_PREFIX = "#"


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


class Identifiers:
    idx = 0

    @classmethod
    def next(cls):
        cls.idx += 1
        # return PredicateObject(
        #     f"{OPT_PREFIX}x{cls.idx}",
        #     generated=True,
        # )
        return f"{OPT_PREFIX}x{cls.idx}"


class Unsatisfiable(Exception):
    pass


def topological_sort(stream_actions, computed_objects):
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

    return result, incoming_edges


def replace_objects_in_condition(condition, object_mapping):
    if isinstance(condition, Atom):
        return Atom(
            condition.predicate,
            tuple(object_mapping.get(o, o) for o in condition.args),
        )
    elif isinstance(condition, NegatedAtom):
        return NegatedAtom(
            condition.predicate,
            tuple(object_mapping.get(o, o) for o in condition.args),
        )
    elif isinstance(condition, Conjunction):
        return Conjunction(
            [replace_objects_in_condition(c, object_mapping) for c in condition.parts]
        )
    else:
        raise NotImplementedError(
            "Replacements for condition type apart from Atom not implemented"
        )


def replace_objects_in_action(action, object_mapping):
    new_var_mapping = {
        arg: object_mapping.get(var, var) for arg, var in action.var_mapping.items()
    }

    new_precondition = replace_objects_in_condition(action.precondition, object_mapping)

    new_effects = []
    for condition, atom, effect, assignment in action.effect_mappings:
        new_atom = replace_objects_in_condition(atom, object_mapping)
        new_assignment = {
            new_var_mapping[k]: object_mapping.get(v, v) for k, v in assignment.items()
        }
        new_effects.append((condition, new_atom, effect, new_assignment))

    return PropositionalAction(
        new_precondition, new_effects, action.cost, action.action, new_var_mapping
    )


class PredicateObject(collections.UserString):
    def __init__(self, name, generated=False):
        super().__init__(name)
        self.generated = generated


class PropositionalAction:
    """
    Adapted from pddlstream/FastDownward/src/translate/pddl/actions.py
    with the addition of method to generate `name` using var_mapping
    instead of a hardcoded name to support mutable objects
    """

    def __init__(self, precondition, effects, cost, action=None, var_mapping=None):
        self.precondition = precondition
        self.add_effects = []
        self.del_effects = []
        self.effect_mappings = effects
        for condition, effect, _, _ in effects:
            if not effect.negated:
                self.add_effects.append((condition, effect))
        # Warning: This is O(N^2), could be turned into O(N).
        # But that might actually harm performance, since there are
        # usually few effects.
        # TODO: Measure this in critical domains, then use sets if acceptable.
        for condition, effect, _, _ in effects:
            if effect.negated and (condition, effect.negate()) not in self.add_effects:
                self.del_effects.append((condition, effect.negate()))
        self.cost = cost
        self.action = action
        self.var_mapping = var_mapping

    @property
    def name(self):
        return "%s %s" % (
            self.action.name,
            " ".join(
                str(self.var_mapping[param.name]) for param in self.action.parameters
            ),
        )

    def __repr__(self):
        return "<PropositionalAction %r at %#x>" % (self.name, id(self))

    def dump(self):
        print(self.name)
        for fact in self.precondition:
            print("PRE: %s" % fact)
        for cond, fact in self.add_effects:
            print("ADD: %s -> %s" % (", ".join(map(str, cond)), fact))
        for cond, fact in self.del_effects:
            print("DEL: %s -> %s" % (", ".join(map(str, cond)), fact))
        print("cost:", self.cost)


def anonymise(orig_obj, orig_obj_cg, id_cg_map):
    counter = itertools.count()
    stack = [orig_obj]
    edges = set()
    anon = {orig_obj: f"x{next(counter)}"}

    while stack:
        obj = stack.pop(0)

        if obj == orig_obj:
            idx, stream_name, inputs, fluents = orig_obj_cg
        elif obj in id_cg_map and id_cg_map[obj] != obj:
            idx, stream_name, inputs, fluents = id_cg_map[obj]
        else:
            continue

        input_objs = inputs + tuple(arg for fluent in fluents for arg in fluent.args)

        for parent_obj in input_objs:
            stack.insert(0, parent_obj)

            if parent_obj not in anon and parent_obj in id_cg_map:
                anon[parent_obj] = f"x{next(counter)}"
    
            edges.add(
                (
                    anon.get(parent_obj, parent_obj),
                    (stream_name, idx),
                    anon[obj],
                )
            )

    return tuple(sorted(edges))
