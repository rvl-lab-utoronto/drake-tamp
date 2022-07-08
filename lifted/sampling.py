from collections import defaultdict
from dataclasses import dataclass
from lifted.partial import StreamAction, DummyStream

from pddlstream.language.object import Object

from lifted.utils import PriorityQueue, topological_sort
import random

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

    computed_objects = set(stream_plan[0][0][2].object_stream_map)

    stream_ordering = []
    for edge, object_map in stream_plan:
        stream_actions = {action for action in object_map.values()}
        local_ordering, missing = topological_sort(stream_actions, computed_objects)
        assert (
            not missing
        ), "Something went wrong. Either the CG has a cycle, or depends on missing (or future) step."
        stream_ordering.extend(local_ordering)
    return stream_ordering


@dataclass
class Binding:
    index: int
    stream_plan: list
    mapping: dict


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
            final_state.id_anon_cg_map[obj]
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
            k: v for k, v in edge[1].object_stream_map_delta.items() if v is not None
        }
        stream_plan.append((edge, stream_map))
    return stream_plan


def ancestral_sampling(stream_ordering, objects_from_name=None):
    if objects_from_name is None:
        objects_from_name = Object._obj_from_name
    nodes = stream_ordering
    edges = get_stream_action_edges(stream_ordering)
    final_node = StreamAction(
        DummyStream("FINAL"),
        inputs=tuple(obj for stream_action in nodes for obj in stream_action.outputs),
        outputs=tuple(),
    )
    start_node = StreamAction(DummyStream("START"), inputs=tuple(), outputs=tuple())
    children = {}
    for parent, child in edges:
        children.setdefault(parent, set()).add(child)
    for node in nodes:
        children.setdefault(node, set()).add(final_node)
        children.setdefault(start_node, set()).add(node)
    stats = {node: 0 for node in nodes}
    produced = dict()
    queue = [Binding(0, [start_node], {})]
    levels = {start_node: 0}
    attempts = 0
    while queue:
        binding = queue.pop(0)
        stream_action = binding.stream_plan[0]
        if stream_action not in [start_node, final_node]:
            input_objects = [
                produced.get(var_name) or objects_from_name[var_name]
                for var_name in stream_action.inputs
            ]
            fluent_facts = [
                (f.predicate,)
                + tuple(
                    produced.get(var_name) or objects_from_name[var_name]
                    for var_name in f.args
                )
                for f in stream_action.fluent_facts
            ]
            stream_instance = stream_action.stream.get_instance(
                input_objects, fluent_facts=fluent_facts
            )
            if stream_instance.enumerated:
                if len(stream_instance.results_history) > 0:
                    results = random.choice(stream_instance.results_history)
                else:
                    results = None
            else:
                results, _ = stream_instance.next_results(verbose=False)
                attempts += 1
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
            input_objects = list(child.inputs) + [
                var_name for f in child.fluent_facts for var_name in f.args
            ]
            if all(
                obj in produced or obj in objects_from_name for obj in input_objects
            ):
                levels[child] = levels[stream_action] + 1
                new_binding = Binding(binding.index + 1, [child], {})
                queue.append(new_binding)
    return produced, stats.get(final_node, 0), levels, attempts


def ancestral_sample_with_costs(
    stream_ordering, final_state, stats={}, max_steps=30, verbose=False
):
    to_produce = set({out for s in stream_ordering for out in s.outputs})
    for i in range(max_steps):
        produced, done = ancestral_sampling(stream_ordering)

        for obj in to_produce:
            cg_key = final_state.id_anon_cg_map[obj]
            cg_stats = stats.setdefault(
                cg_key, {"num_attempts": 0.0, "num_successes": 0.0}
            )
            cg_stats["num_attempts"] += 1
            if obj in produced:
                cg_stats["num_successes"] += 1

        if done:
            return produced
    return None


def ancestral_sampling_by_edge(stream_plan, final_state, stats, max_steps=30):
    (initial_state, _, _), _ = stream_plan[0]
    objects = {
        k: v
        for k, v in Object._obj_from_name.items()
        if k in initial_state.object_stream_map
    }
    i = 0
    particles = [[objects]]
    while i < len(stream_plan):
        (_, op, state), step = stream_plan[i]

        if step:
            to_produce = set({out for s in step for out in s.outputs})
            step_particles = []
            particles.append(step_particles)
            for j in range(max_steps):
                prev_particle = particles[i][j % len(particles[i])]

                new_objects, success, _ = ancestral_sampling(step, prev_particle)

                for obj in to_produce:
                    cg_key = state.id_anon_cg_map[obj]
                    cg_stats = stats.setdefault(
                        cg_key, {"num_attempts": 0.0, "num_successes": 0.0}
                    )
                    stats[cg_key]["num_attempts"] += 1
                    if obj in new_objects:
                        stats[cg_key]["num_successes"] += 1

                if success:
                    _temp_dict = prev_particle.copy()
                    _temp_dict.update(new_objects)
                    step_particles.append(_temp_dict)
            if len(step_particles) == 0:
                break
        else:
            particles.append(particles[-1])
        i += 1

    
    return particles[-1][0] if len(stream_plan) + 1 == len(particles) and particles[-1] else None

def ancestral_sampling_by_edge_seq(stream_plan, final_state, stats, max_steps=30, result=None):
    (_,_,initial_state), _ = stream_plan[0]
    objects = {k:v for k,v in Object._obj_from_name.items() if k in initial_state.object_stream_map}
    i = 0
    particles = [
        [objects]
    ] + [[] for _ in range(len(stream_plan))]
    z = {}
    for k in range(max_steps):
        for i in range(len(stream_plan)):
            (_, op, state), step = stream_plan[i]
            assert len(particles[i]) >= 1, (k, i)

            if step:
                edge_stats = stats.setdefault(op, {'num_attempts': 0., 'num_successes': 0.}) 
                to_produce = set({out for s in step for out in s.outputs})
                prev_particle = particles[i][k % len(particles[i])]
                # prev_particle = random.choice(particles[i])

                edge_stats["num_attempts"] += 1
                new_objects, success, levels, attempts = ancestral_sampling(step, prev_particle)
                result.num_samples += attempts
                for obj in to_produce:
                    cg_key = state.id_anon_cg_map[obj]
                    cg_stats = stats.setdefault(cg_key, {'num_attempts': 0., 'num_successes': 0.})
                    cg_stats['num_attempts'] += 1
                    if obj in new_objects:
                        cg_stats['num_successes'] += 1
                        #stats.setdefault(obj, []).append(new_objects[obj])
                for index_i, stream_action_i in enumerate(step):
                    if stream_action_i not in levels:
                        continue
                    success_i = any(o in new_objects for o in stream_action_i.outputs)
                    for _, stream_action_j in enumerate(step[index_i + 1:]):
                        if stream_action_j not in levels:
                            continue
                        success_j = any(o in new_objects for o in stream_action_j.outputs)
                        
                        if levels[stream_action_i] == levels[stream_action_j]:

                            if (success_i or success_j):
                                pair_key = frozenset([
                                    state.id_anon_cg_map[stream_action_i.outputs[0]],
                                    state.id_anon_cg_map[stream_action_j.outputs[0]]
                                ])
                                kstats = z.setdefault(pair_key, {"num_attempts": 0, "num_successes": 0, "i": 0, "j": 0})
                                kstats["i"] += success_i
                                kstats["j"] += success_j
                                kstats["num_attempts"]+= 1
                                if success_i and success_j:
                                    kstats["num_successes"]+= 1

                if success:
                    edge_stats['num_successes'] += 1
                    particles[i + 1].append(dict(**prev_particle, **new_objects))
                if len(particles[i + 1]) == 0:
                    break
            else:
                particles[i + 1] = particles[i]
        else:
            return particles[-1][0]

    stats.setdefault('pairs', {}).update({k:v for k,v in z.items() if v['num_successes'] < min(v['i'], v['j'])})
    return None
