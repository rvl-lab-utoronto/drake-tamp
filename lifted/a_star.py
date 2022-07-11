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


def beam(search, priority, result, beam_size, max_steps=math.inf, max_time=None, **kwargs):
    start_time = time.time()
    beam = [search.init]
    closed = set()
    expand_count = 0
    evaluate_count = 0
    found = False

    while beam and expand_count < max_steps and (time.time() - start_time) < max_time:
        q = PriorityQueue()
        for state in beam:
            if state in closed:
                continue

            expand_count += 1

            for op, child in search.successors(state):
                child.parents = {(op, state)}
                state.children.add((op, child))
                evaluate_count += 1

            for op, child in sorted(state.children, key=lambda x: x[0].name):
   
                if search.test_goal(child):
                    state = child
                    found = True
                    break

                is_unique = True
                if ENABLE_EQUALITY_CHECK:
                    if child in closed:
                        is_unique = False

                if not is_unique:
                    continue
                
                q.push(child, priority(state, op, child))

            if found:
                break

            closed.add(state)

        if found:
            break

        beam = []
        while len(beam) < beam_size and len(q) > 0:
            beam.append(q.pop())
    
    result.expand_count += expand_count
    result.evaluate_count += evaluate_count

    if found:
        return state
    else:
        return None


def bfs(search, priority, result, max_steps=math.inf, max_time=None, **kwargs):
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
            child.parents = {(op, state)}
            state.children.add((op, child))
            evaluate_count += 1

        for op, child in sorted(state.children, key=lambda x: x[0].name):
            is_unique = True
            if ENABLE_EQUALITY_CHECK:
                if child in closed:
                    is_unique = False

            if not is_unique:
                continue

            q.push(child, priority(state, op, child))

        closed.add(state)

    result.expand_count += expand_count
    result.evaluate_count += evaluate_count
    if found:
        return state
    else:
        return None

def bfs_exclusion(search, priority, result, max_steps=10001, max_time=None, closed_exclusion=None, **kwargs):
    start_time = time.time()
    q = PriorityQueue([search.init])
    closed = {search.init}
    closed_exclusion = closed_exclusion if closed_exclusion is not None else set()

    expand_count = 0
    evaluate_count = 0
    found = False

    while q and expand_count < max_steps and (time.time() - start_time) < max_time:
        state = q.pop()
        expand_count += 1

        if search.test_goal(state):
            found = True
            break

        for op, child in search.successors(state):
            evaluate_count += 1
            child.parents = {(op, state)}
            state.children.add((op, child))

            if child in closed_exclusion or child not in closed:
                q.push(child, priority(state, op, child))
                if child in closed_exclusion:
                    closed_exclusion.remove(child)
                else:
                    closed.add(child)

    time_taken = time.time() - start_time

    result.expand_count += expand_count
    result.evaluate_count += evaluate_count

    if found:
        return state
    return None

def try_a_star(search, cost, heuristic, result, **kwargs):
    def priority(state, op, child):
        child.start_distance = state.start_distance + cost(state, op, child)
        child.cost_to_go = heuristic(child, search.goal)
        return child.start_distance + child.cost_to_go

    return bfs(search, priority, result, **kwargs)
    
def try_policy_guided(search, policy_ts, result, **kwargs):
    return bfs(search, policy_ts, result, **kwargs)

def try_a_star_exclusion(search, cost, heuristic, result, **kwargs):
    def priority(state, op, child):
        child.start_distance = state.start_distance + cost(state, op, child)
        child.cost_to_go = heuristic(child, search.goal)
        return child.start_distance + child.cost_to_go

    return bfs_exclusion(search, priority, result, **kwargs)
    
def try_policy_guided_exclusion(search, policy_ts, result, **kwargs):
    return bfs_exclusion(search, policy_ts, result, **kwargs)

def try_beam(search, cost, heuristic, result, beam_size, **kwargs):
    def priority(state, op, child):
        child.start_distance = state.start_distance + cost(state, op, child)
        child.cost_to_go = heuristic(child, search.goal)
        return child.start_distance + child.cost_to_go

    return beam(search, priority, result, beam_size, **kwargs)

def try_policy_guided_beam(search, policy_ts, result, beam_size, **kwargs):
    return beam(search, policy_ts, result, beam_size, **kwargs)


def default_action_cost(state, op, child, stats={}, verbose=False):
    s = stats.get(op, None)
    if s is None:
        return 1
    return (
        (s["num_successes"] + 1) / (s["num_attempts"] + 1)
    ) ** -1

def _stream_cost(state, op, child, verbose=False,stats=dict()):
    given_stats = stats
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

def stream_cost(state, op, child, verbose=False,stats=dict()):
    s = _stream_cost(state, op, child, verbose, stats)
    return (
        (s["num_successes"] + 1) / (s["num_attempts"] + 1)
    ) ** -1

def goalcount_heuristic(s, g):
    return len(g - s.state)

def repeated_a_star(search, max_steps=1000, stats={}, heuristic=goalcount_heuristic, cost=stream_cost, debug=False, edge_eval_steps=30, max_time=None, policy_ts=None, beam_size=20, search_func=try_a_star):
    def lprint(*args):
        if debug:
            print(*args)
    result = Result(stats)
    cost = partial(cost, stats=stats)

    closed_exclusion = set()
    start_time = time.time()

    max_time = max_time if max_time is not None else math.inf
    for _ in range(max_steps):
        goal_state = search_func(
            search,
            cost=cost,
            policy_ts=policy_ts,
            max_time=max_time - (time.time() - start_time),
            heuristic=heuristic,
            result=result,
            closed_exclusion=closed_exclusion,
            beam_size=beam_size
        )
        if goal_state is None:
            lprint("Could not find feasable action plan!")
            object_mapping = None
            break

        lprint("Getting path ...")
        path = goal_state.get_shortest_path_to_start()
        closed_exclusion |= {s for s, _, _ in path} | {goal_state}

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


