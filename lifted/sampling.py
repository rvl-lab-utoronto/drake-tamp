from collections import defaultdict
from dataclasses import dataclass
from lifted.partial import StreamAction, DummyStream

from pddlstream.language.object import Object

from lifted.utils import PriorityQueue


def extract_stream_plan(state):
    """Given a search state, return the list of object stream maps needed by each action
    along the path to state. The list contains one dictionary per action."""

    stream_plan = []
    while state is not None:
        objects_created = {
            k: v for k, v in state.object_stream_map.items() if v is not None
        }
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
    computed_objects = {x for x,s in stream_plan[0][0][2].object_stream_map.items() if s is None}

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

        assert (
            not incoming_edges
        ), "Something went wrong. Either the CG has a cycle, or depends on missing (or future) step."
        return result

    stream_ordering = []
    for edge, object_map in stream_plan:
        stream_actions = {action for action in object_map.values()}
        local_ordering = topological_sort(stream_actions)
        stream_ordering.extend(local_ordering)
    return stream_ordering


@dataclass
class Binding:
    index: int
    stream_plan: list
    mapping: dict


def sample_depth_first(stream_plan, max_steps=10000):
    """Demo sampling a stream plan using a backtracking depth first approach.
    Returns a mapping if one exists, or None if infeasible or timeout."""
    queue = [Binding(0, stream_plan, {})]
    steps = 0
    while queue and steps < max_steps:
        binding = queue.pop(0)
        steps += 1

        stream_action = binding.stream_plan[binding.index]

        input_objects = [
            binding.mapping.get(var_name) or Object.from_name(var_name)
            for var_name in stream_action.inputs
        ]
        fluent_facts = [
            (f.predicate,)
            + tuple(
                binding.mapping.get(var_name) or Object.from_name(var_name)
                for var_name in f.args
            )
            for f in stream_action.fluent_facts
        ]
        stream_instance = stream_action.stream.get_instance(
            input_objects, fluent_facts=fluent_facts
        )
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
    return None  # infeasible or reached step limit


def sample_depth_first_with_costs(
    stream_ordering, final_state, stats={}, max_steps=None, verbose=False
):
    """Demo sampling a stream plan using a backtracking depth first approach.
    Returns a mapping if one exists, or None if infeasible or timeout."""
    if max_steps is None:
        max_steps = len(stream_ordering) * 3
    queue = PriorityQueue([Binding(0, stream_ordering, {})])
    steps = 0
    while queue and steps < max_steps:
        binding = queue.pop()
        steps += 1

        stream_action = binding.stream_plan[binding.index]

        input_objects = [
            binding.mapping.get(var_name) or Object.from_name(var_name)
            for var_name in stream_action.inputs
        ]
        fluent_facts = [
            (f.predicate,)
            + tuple(
                binding.mapping.get(var_name) or Object.from_name(var_name)
                for var_name in f.args
            )
            for f in stream_action.fluent_facts
        ]
        stream_instance = stream_action.stream.get_instance(
            input_objects, fluent_facts=fluent_facts
        )

        if stream_instance.enumerated:
            continue

        result = stream_instance.next_results()

        output_cg_keys = [
            final_state.get_object_computation_graph_key(obj)
            for obj in stream_action.outputs
        ]
        for cg_key in output_cg_keys:
            cg_stats = stats.setdefault(
                cg_key, {"num_attempts": 0.0, "num_successes": 0.0}
            )
            cg_stats["num_attempts"] += 1

        if len(result[0]) == 0:
            if verbose:
                print(f"Invalid result for {stream_action}: {result}")
            # queue.push(binding, (stream_instance.num_calls, len(stream_ordering) - binding.index))
            continue

        for cg_key in output_cg_keys:
            cg_stats = stats[cg_key]
            cg_stats["num_successes"] += 1
        [new_stream_result], new_facts = result
        output_objects = new_stream_result.output_objects

        new_mapping = binding.mapping.copy()
        new_mapping.update(dict(zip(stream_action.outputs, output_objects)))
        new_binding = Binding(binding.index + 1, binding.stream_plan, new_mapping)

        if len(new_binding.stream_plan) == new_binding.index:
            return new_binding.mapping

        queue.push(new_binding, (0, len(stream_ordering) - new_binding.index))
        queue.push(
            binding, (stream_instance.num_calls, len(stream_ordering) - binding.index)
        )
    return None  # infeasible or reached step limit


def extract_stream_plan_from_path(path):
    stream_plan = []
    for edge in path:
        stream_map = {
            k: v for k, v in edge[2].object_stream_map.items() if v is not None
        }
        stream_plan.append((edge, stream_map))
    return stream_plan

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

    
    return particles[-1][0] if len(stream_plan) + 1 == len(particles) and particles[-1] else None