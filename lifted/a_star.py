from functools import partial
import itertools
import math
import copy
import time

from pddlstream.language.conversion import fact_from_evaluation


import sys

sys.path.insert(
    0,
    "/home/mohammed/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/",
)
sys.path.insert(
    0, "/home/atharv/drake-tamp/pddlstream/FastDownward/builds/release64/bin/translate/"
)
from pddl.conditions import Atom

from lifted.utils import PriorityQueue
from lifted.search import ActionStreamSearch
from lifted.sampling import ancestral_sampling_by_edge_seq, extract_stream_plan_from_path, ancestral_sampling_by_edge

ENABLE_EQUALITY_CHECK = False

def try_a_star(search, cost, heuristic, result, max_steps=10001, max_time=None, **kwargs):
    start_time = time.time()
    q = PriorityQueue([search.init])
    closed = set()
    expand_count = 0
    evaluate_count = 0
    found = False

    while q and expand_count < max_steps and (time.time() - start_time) < max_time:
        state = q.pop()

        if state in closed:
            continue

        if search.test_goal(state):
            found = True
            break

        expand_count += 1

        for op, child in search.successors(state):
            is_unique = True
            if ENABLE_EQUALITY_CHECK:
                if child in closed:
                    is_unique = False

            if not is_unique:
                continue

            child.parents = {(op, state)}
            child.ancestors = state.ancestors | {state}
            state.children.add((op, child))
            evaluate_count += 1

        for op, child in sorted(state.children, key=lambda x: x[0].name):
            child.start_distance = state.start_distance + cost(state, op, child)
            child.cost_to_go = heuristic(child, search.goal)
            q.push(child, child.start_distance + heuristic(child, search.goal))

        closed.add(state)

    result.expand_count += expand_count
    result.evaluate_count += evaluate_count
    if found:
        return state
    else:
        return None

def try_a_star_modified(search, cost, heuristic, max_step=10000):
    start_time = time.time()
    q = PriorityQueue([search.init])
    closed = {}
    generated = {hash(search.init): search.init}
    expand_count = 0
    evaluate_count = 0
    found = False

    while q and expand_count < max_step:
        state = q.pop()

        if hash(state) in closed:
            continue

        expand_count += 1

        if search.test_goal(state):
            found = True
            break

        successors = search.successors(state)
        for op, child in successors:

            if hash(child) in generated:
                node = generated[hash(child)]
                node.parents.add((op, state))
                node.start_distance = min(
                    node.start_distance, state.start_distance + cost(state, op, node)
                )
                state.children.add((op, node))
                continue

            generated[hash(child)] = child
            child.parents = {(op, state)}
            child.start_distance = state.start_distance + cost(state, op, child)
            state.children.add((op, child))
            evaluate_count += 1
            q.push(child, child.start_distance + heuristic(child, search.goal))

        closed[hash(state)] = state

    av_branching_f = evaluate_count / expand_count
    approx_depth = math.log(1e-6 + evaluate_count) / math.log(1e-6 + av_branching_f)
    print(f"Explored {expand_count}. Evaluated {evaluate_count}")
    print(f"Av. Branching Factor {av_branching_f:.2f}. Approx Depth {approx_depth:.2f}")
    print(f"Time taken: {(time.time() - start_time)} seconds")
    print(f"Solution cost: {state.start_distance}")

    return state if found else None


def try_a_star_tree(search, cost, heuristic, max_steps=10000, closed_exclusion=None):
    start_time = time.time()
    q = PriorityQueue([search.init])
    closed = {search.init}
    closed_exclusion = closed_exclusion if closed_exclusion is not None else set()

    expand_count = 0
    evaluate_count = 0
    found = False

    while q and expand_count < max_steps:
        state = q.pop()
        expand_count += 1

        if search.test_goal(state):
            found = True
            break

        for op, child in search.successors(state):
            evaluate_count += 1
            child.parents = {(op, state)}
            child.ancestors = state.ancestors | {state}
            child.start_distance = state.start_distance + cost(state, op, child)
            state.children.add((op, child))

            if child in closed_exclusion or child not in closed or child in state.ancestors:
                q.push(child, child.start_distance + heuristic(child, search.goal))
                if child in closed_exclusion:
                    closed_exclusion.remove(child)
                else:
                    closed.add(child)

    time_taken = time.time() - start_time

    av_branching_f = evaluate_count / expand_count
    approx_depth = math.log(1e-6 + evaluate_count) / math.log(1e-6 + av_branching_f)
    print(f"Explored {expand_count}. Evaluated {evaluate_count}")
    print(f"Av. Branching Factor {av_branching_f:.2f}. Approx Depth {approx_depth:.2f}")
    print(f"Time taken: {time_taken} seconds")
    print(f"Solution cost: {state.start_distance}")

    result = {
        "expanded": expand_count,
        "evaluated": evaluate_count,
        "search_time": time_taken,
        "solved": found,
        "out_of_budget": expand_count >= max_steps,
    }

    return state, result


class Result:
    solution = None
    stats = None
    skeletons = None
    action_skeleton = None
    object_mapping = None
    def __init__(self, stats):
        self.stats = stats
        self.skeletons = []
        self.start_time = time.time()
        self.expand_count = 0
        self.evaluate_count = 0
        self.num_samples = 0
    def end(self, goal_state, object_mapping):
        if object_mapping is not None:
            action_skeleton = [a for _, a, _ in goal_state.get_shortest_path_to_start()]
            self.action_skeleton = action_skeleton
            self.object_mapping = object_mapping
            self.solution = goal_state
        self.planning_time = time.time() - self.start_time

def default_action_cost(state, op, child, stats={}, verbose=False):
    s = stats.get(op, None)
    if s is None:
        return 1
    return (
        (s["num_successes"] + 1) / (s["num_attempts"] + 1)
    ) ** -1

def stream_cost(state, op, child, verbose=False,given_stats=dict()):
    if op in given_stats:
        return given_stats[op]
    c = 1
    res = {"num_successes": 0,"num_attempts": 0}
    included = set()
    for stream_action in {v for v in child.object_stream_map.values() if v is not None}:
        obj = stream_action.outputs[0]
        cg_key = child.id_anon_cg_map[obj]
        if cg_key in given_stats:
            included.add(cg_key)
            s = given_stats[cg_key]
            comp_cost = (
                (s["num_successes"] + 1) / (s["num_attempts"] + 1)
            )
            if comp_cost < c:
                c = comp_cost
                res = s
    for i, j in itertools.combinations(included, 2):
        cg_key = frozenset([i, j])
        if cg_key in given_stats.get("pairs", {}):
            s = given_stats["pairs"][cg_key]
            comp_cost = (
                (s["num_successes"] + 1) / (s["num_attempts"] + 1)
            )
            if comp_cost < c:
                c = comp_cost
                res = s
    return res

def goalcount_heuristic(s, g):
    return len(g - s.state)

def repeated_a_star(search, max_steps=1000, stats={}, heuristic=goalcount_heuristic, cost=default_action_cost, debug=False, edge_eval_steps=30, max_time=None, policy_ts=None, search_func=try_a_star):
    def lprint(*args):
        if debug:
            print(*args)
    result = Result(stats)
    cost = partial(cost, stats=stats)

    closed_exclusion = set()

    max_time = max_time if max_time is not None else math.inf
    for _ in range(max_steps):
        goal_state = search_func(search, cost=cost, policy_ts=policy_ts, max_time=max_time, heuristic=heuristic, result=result, closed_exclusion=closed_exclusion)
        if goal_state is None:
            lprint("Could not find feasable action plan!")
            object_mapping = None
            break

        closed_exclusion |= goal_state.ancestors | {goal_state}

        lprint("Getting path ...")
        path = goal_state.get_shortest_path_to_start()
        c = 0
        for idx, i in enumerate(path):
            lprint(idx, i[1])
            a = cost(*i, verbose=True)
            lprint("action cost:", a)
            lprint("cum cost:", a + c)
            c += a

        action_skeleton = [a for _, a, _ in path]
        actions_str = "\n".join([str(a) for a in action_skeleton])
        lprint(f"Action Skeleton:\n{actions_str}")

        stream_plan = extract_stream_plan_from_path(path)
        stream_plan = [(edge, list({action for action in object_map.values()})) for (edge, object_map) in stream_plan]
        object_mapping = ancestral_sampling_by_edge_seq(stream_plan, goal_state, stats, max_steps=edge_eval_steps, result=result)

        result.skeletons.append(path)
        c = 0
        for idx, i in enumerate(path):
            lprint(idx, i[1])
            a = cost(*i, verbose=True)
            lprint("action cost:", a)
            lprint("cum cost:", a + c)
            c += a


        if object_mapping is not None:
            break
        lprint("Could not find object_mapping, retrying with updated costs")

    result.end(goal_state, object_mapping)
    return result


