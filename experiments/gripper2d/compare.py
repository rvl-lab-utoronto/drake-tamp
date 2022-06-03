import time

import pandas as pd
from lifted.a_star import repeated_a_star
from lifted.search import ActionStreamSearch
from experiments.gripper2d.problem import generate_scene
from experiments.gripper2d.run import create_problem, solve, StreamInfo
from experiments.gripper2d.lifted_problem import create_problem as create_problem_lifted
import sys, os, time
from contextlib import redirect_stdout
import multiprocessing as mp
import tempfile
from tqdm import tqdm
import json

def run_adaptive(scene, goal, logpath):
    problem = create_problem(scene, goal)
    start = time.time()
    solution = solve(
        problem,
        algorithm='adaptive',
        # use_unique=True,
        max_time=30,
        search_sample_ratio=0.01,
        max_planner_time = 30,
        logpath=logpath,
        stream_info={
            "grasp": StreamInfo(use_unique=True),   
            "ik": StreamInfo(use_unique=True),  
            "placement": StreamInfo(use_unique=True),   
            "safe": StreamInfo(use_unique=True),    
            "safe-block": StreamInfo(use_unique=True),  
        },
    verbose=False
    )
    end = time.time()

    plan, _, evaluations = solution

    with open(logpath + "stats.json") as f:
        stats = json.load(f)
    stats = stats["fd_stats"]
    solved_stats_list = [s for s in stats if s.get("solved", False)]
    solved_stats = solved_stats_list[-1] if solved_stats_list else None

    return {
        "algo": "adaptive",
        "success": plan is not None,
        "duration": end - start,
        "expanded": solved_stats["expanded"] if solved_stats else None,
        "evaluated": solved_stats["evaluated"] if solved_stats else None,
        "search_time": solved_stats["search_time"] if solved_stats else None,
        "total_expanded": sum([s["expanded"] for s in stats if "expanded" in s]),
        "total_evaluated": sum([s["evaluated"] for s in stats if "evaluated" in s]),
        "total_search_time": sum([s["search_time"] for s in stats if "search_time" in s]),
        "attempts": len([s for s in stats if "expanded" in s]),
        "skeleton_length": len(plan) if plan else None,
    }

def heuristic(state, goal):
    return len(goal - state.state)*10

def run_lifted(scene, goal):
    initial_state, goal, externals, actions, _ = create_problem_lifted(scene, goal)
    start = time.time()
    search = ActionStreamSearch(initial_state, goal, externals, actions)
    stats = {}
    result = repeated_a_star(search, stats=stats, max_steps=20, heuristic=heuristic)
    end = time.time()
    return {
        "algo": "lifted",
        "success": result["success"],
        "duration": end - start,
        "expanded": result["stats"][-1]["expanded"],
        "evaluated": result["stats"][-1]["evaluated"],
        "search_time": result["stats"][-1]["search_time"],
        "total_expanded": sum([s["expanded"] for s in result["stats"]]),
        "total_evaluated": sum([s["evaluated"] for s in result["stats"]]),
        "total_search_time": sum([s["search_time"] for s in result["stats"]]),
        "attempts": len(result["stats"]),
        "skeleton_length": len(result["action_skeleton"]) if result["success"] else None,
    }

    return result["success"], end - start,

def run_adaptive_process(scene, goal, scene_idx, rep_idx, res):
    with open(f"/tmp/adaptive_{scene_idx}_{rep_idx}_out.log", 'w') as f:
        with redirect_stdout(f):
            with tempfile.TemporaryDirectory() as tmpdirname:
                os.chdir(tmpdirname)
                res[(scene_idx, rep_idx, 'adaptive')] = run_adaptive(scene, goal, logpath=f"/tmp/adaptive_{scene_idx}_{rep_idx}_")

def run_lifted_process(scene, goal, scene_idx, rep_idx, res):
    with open(f"/tmp/lifted_{scene_idx}_{rep_idx}.log", 'w') as f:
        with redirect_stdout(f):
            res[(scene_idx, rep_idx, 'lifted')] = run_lifted(scene, goal)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", "-p", type=str, default="all", choices=["all", "lifted", "adaptive"], help="Which planner to use")
    parser.add_argument("--num_scenes", "-s", type=int, default=50, help="Number of scenes to run")
    parser.add_argument("--num_reps", "-r", type=int, default=4, help="Number of repetitions per scene")
    args = parser.parse_args()
    run_adaptive_flag = args.planner == "adaptive" or args.planner == "all"
    run_lifted_flag = args.planner == "lifted" or args.planner == "all"

    import numpy as np
    np.random.seed(10)
    num_scenes = args.num_scenes
    num_reps = args.num_reps
    scenes = []
    data = []
    goal = ('and', 
        ('on', 'b0', 'r2'),
        ('on', 'b1', 'r1'),
        ('on', 'b2', 'r1'),
        ('on', 'b3', 'r1')
    )
    block_heights = [[1, 2, 3, 4]]
    region_widths = [5, 1, 4]
    for i in range(num_scenes):
        scene = generate_scene(block_heights, region_widths)
        scenes.append(scene)

    manager = mp.Manager()
    return_dict = manager.dict()
    pool = mp.Pool(processes=mp.cpu_count() - 1)
    results = []
    for scene_idx in range(num_scenes):
        for rep in range(num_reps):
            if run_adaptive_flag:
                results.append(
                    pool.apply_async(
                        run_adaptive_process, 
                        (scenes[scene_idx], goal, scene_idx, rep, return_dict)
                    )
                )
            if run_lifted_flag:
                results.append(
                    pool.apply_async(
                        run_lifted_process,
                        (scenes[scene_idx], goal, scene_idx, rep, return_dict)  
                    )
                )

    pool.close()
    num_jobs = len(results)

    pbar = tqdm(total=num_jobs)
    old_done_count = 0
    while True:
        time.sleep(0.1)
        done_count = sum([r.ready() for r in results])

        if done_count > old_done_count:
            pbar.update(done_count - old_done_count)
            old_done_count = done_count

        if done_count == num_jobs:
            break
    pbar.close()
    pool.join()

    data = []
    missing = []
    for scene_idx in range(num_scenes):
        for rep in range(num_reps):
            if run_lifted_flag: 
                key = (scene_idx, rep, 'lifted')
                if key in return_dict:
                    res_dict = return_dict[key]
                    res_dict['scene_idx'] = scene_idx
                    res_dict['rep'] = rep
                    data.append(res_dict)
                else:
                    missing.append(("lifted", scene_idx, rep))
            if run_adaptive_flag:
                key = (scene_idx, rep, 'adaptive')
                if key in return_dict:
                    res_dict = return_dict[key]
                    res_dict['scene_idx'] = scene_idx
                    res_dict['rep'] = rep
                    data.append(res_dict)
                else:
                    missing.append(("adaptive", scene_idx, rep))

    df = pd.DataFrame(data)
    df.to_csv('temp/comparison_results.csv')

    print("Missing: ", missing)

    if run_lifted_flag: 
        print("Lifted Summary -")
        print(df[df.algo == 'lifted'].mean())
    if run_adaptive_flag:
        print("Adaptive Summary -")
        print(df[df.algo == 'adaptive'].mean())
