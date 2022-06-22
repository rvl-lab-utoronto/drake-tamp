from dataclasses import dataclass, field
import itertools
import copy

from pddl.conditions import Atom
from pddlstream.language.stream import Stream, StreamResult

from lifted.utils import (
    Identifiers,
    Unsatisfiable,
    topological_sort,
    replace_objects_in_condition,
    anonymise
)


def get_assignment(group, stream):
    assignment = {}
    for certified in stream.certified:
        for fact in group:
            if certified[0] == fact.predicate:
                partial = {var: val for (var, val) in zip(certified[1:], fact.args)}
                if any(
                    assignment.get(var, partial[var]) != partial[var] for var in partial
                ):
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
    domain = {
        Atom(dom[0], [assignment.get(arg) for arg in dom[1:]]) for dom in stream.domain
    }
    certified = {
        Atom(cert[0], [assignment.get(arg) for arg in cert[1:]])
        for cert in stream.certified
    }

    return (inputs, outputs, domain, certified)


# I want to think about partial order planning
@dataclass
class PartialPlan:
    agenda: set
    actions: set
    bindings: dict
    order: list
    links: list

    def copy(self):
        return PartialPlan(
            self.agenda.copy(),
            self.actions.copy(),
            self.bindings.copy(),
            self.order.copy(),
            self.links.copy(),
        )


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
        return f"{self.stream.name}({self.inputs})->({self.outputs}), fluents={self.fluent_facts}"

    def serialize(self):
        return dict(
            stream=self.stream.name,
            inputs=self.inputs,
            outputs=self.outputs,
            fluent_facts=self.fluent_facts,
        )
    def get_cg_key(self):
        return (self.stream.name, self.inputs, self.fluent_facts)


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
class Resolver:
    action: StreamAction = None
    links: list = field(default_factory=[])
    binding: dict = None


def get_resolvers(partial_plan, agenda_item, streams_by_predicate):
    (incomplete_action, missing_precond) = agenda_item
    for action in partial_plan.actions:
        if missing_precond in action.eff:
            assert (
                False
            ), "Didnt expect to get here, considering im doing the same work below"
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
        assignment = get_assignment((missing_precond,), stream)

        (inputs, outputs, pre, eff) = instantiate_stream_from_assignment(
            stream, assignment, new_vars=True
        )

        binding = {o: o for o in outputs}

        # handle fluents
        if stream.is_fluent:
            fluent_facts = compute_fluent_facts(partial_plan, stream)
        else:
            fluent_facts = tuple()
        if stream.is_test:
            outputs = (Identifiers.next(),)  # just to include it in object stream map.
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
                        assert existing_action.id == -1  # has to be part of the state!
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
    p0 = PartialPlan(
        agenda={(goal_action, sub) for sub in missing},
        actions={init_action, goal_action},
        bindings={o: o for o in object_stream_map},
        order=[],
        links=[],
    )

    while p0.agenda:
        for agenda_item in p0.agenda:
            resolvers = list(get_resolvers(p0, agenda_item, streams_by_predicate))
            if not resolvers:
                raise Unsatisfiable("Deadend")
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


def extract_from_partial_plan_old(
    old_world_state, old_missing, new_world_state, partial_plan, cg_id_map
):
    """Extract the successor state from the old world state and the partial plan.
    That involves updating the object stream map, the set of facts that are yet to be certified,
    and the new logical state based on the stream actions in the partial plan.

    Edit: As of today (Dec 1) objects are not added to the object stream map until their CG is
    fully determined.
    """
    object_stream_map = {o: None for o in old_world_state.object_stream_map}
    missing = old_missing.copy()
    object_mapping = {}
    ordered_actions, _ = topological_sort(partial_plan.actions, set(object_stream_map))
    produced = set()
    used = set()
    for act in ordered_actions:
        if act.stream is None or any(
            parent_obj not in produced
            and parent_obj not in old_world_state.object_stream_map
            for parent_obj in act.inputs
        ):
            continue

        # get inputs to stream action that determing CG
        cg_key = act.get_cg_key()

        # if CG already in map, replace objects
        if cg_key in cg_id_map:
            original_outputs, original_effs = cg_id_map[cg_key]

            if original_outputs != act.outputs or original_effs != act.eff:

                object_mapping.update(
                    {
                        str(old): str(new)
                        for old, new in zip(act.outputs, original_outputs)
                    }
                )

                for old, new in zip(act.outputs, original_outputs):
                    old.data = new.data

        else:
            cg_id_map[cg_key] = act.outputs, act.eff

        if not act.stream.is_fluent:
            new_world_state |= act.eff

        for out in act.outputs:
            object_stream_map[out] = act
            produced.add(out)

        for out in act.inputs:
            used.add(out)

        for e in act.eff:
            for f in list(missing):
                if e.predicate != f.predicate:
                    continue
                for e_args, f_arg in zip(e.args, f.args):
                    if e_args != f_arg:
                        break
                else:
                    missing.remove(f)

    new_world_state = set(
        [replace_objects_in_condition(fact, object_mapping) for fact in new_world_state]
    )

    placeholder = Identifiers.next()
    object_stream_map[placeholder] = StreamAction(
        DummyStream("all"),
        inputs=tuple(
            produced - used
        ),  # i need this to be ordered in order for the cg key to work. But i have nothing with which to base the order on.
        outputs=(placeholder,),
    )

    return new_world_state, object_stream_map, missing


def extract_from_partial_plan(
    old_world_state, old_missing, new_world_state, partial_plan
):

    object_stream_map = {o: None for o in old_world_state.object_stream_map}
    missing = old_missing.copy()
    ordered_actions, _ = topological_sort(partial_plan.actions, set(object_stream_map))
    produced = set()
    used = set()
    
    id_cg_map = old_world_state.id_cg_map.copy()
    id_anon_cg_map = old_world_state.id_anon_cg_map.copy()

    for act in ordered_actions:
        if act.stream is None or any(
            parent_obj not in produced
            and parent_obj not in old_world_state.object_stream_map
            for parent_obj in act.inputs
        ):
            continue

        for ob_idx, ob in enumerate(act.outputs):
            ob_cg = (
                ob_idx,
                act.stream.name,
                act.inputs,
                act.fluent_facts,
            )
            ob_anon_cg = anonymise(ob, ob_cg, id_cg_map)

            if ob in id_cg_map and id_cg_map[ob] != ob_cg:
                raise Exception("The same ID shouldn't be used with a differnt CG!")

            if ob_anon_cg in id_anon_cg_map.values():
                ob_cg = str(ob)
                ob_anon_cg = str(ob)
            
            if ob not in id_cg_map:
                id_cg_map[str(ob)], id_anon_cg_map[str(ob)] = ob_cg, ob_anon_cg


        if not act.stream.is_fluent:
            new_world_state |= act.eff

        for out in act.outputs:
            object_stream_map[out] = act
            produced.add(out)

        for out in act.inputs:
            used.add(out)

        for e in act.eff:
            for f in list(missing):
                if e.predicate != f.predicate:
                    continue
                for e_args, f_arg in zip(e.args, f.args):
                    if e_args != f_arg:
                        break
                else:
                    missing.remove(f)

    ph_obj = Identifiers.next()
    ph_in = tuple(produced - used) # need to define ordering
    object_stream_map[ph_obj] = StreamAction(DummyStream("all"), ph_in, (ph_obj,))
    ph_obj_cg = (0, "all", ph_in, ())
    id_cg_map[str(ph_obj)] = ph_obj_cg
    id_anon_cg_map[str(ph_obj)] = anonymise(ph_obj, ph_obj_cg, id_cg_map)

    return new_world_state, object_stream_map, missing, id_cg_map, id_anon_cg_map
