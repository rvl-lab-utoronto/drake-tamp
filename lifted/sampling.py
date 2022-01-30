from dataclasses import dataclass

from pddlstream.language.object import Object

from utils import PriorityQueue, topological_sort


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
    input_obj_to_streams = {}
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
