import time

import pandas as pd
from lifted.a_star import repeated_a_star
from lifted.search import ActionStreamSearch
from experiments.gripper2d.problem import generate_scene
from experiments.gripper2d.run import create_problem, solve, StreamInfo
from experiments.gripper2d.lifted_problem import create_problem as create_problem_lifted
import sys, os, time
import multiprocessing as mp
import tempfile
from tqdm import tqdm

def run_adaptive(scene, goal):
    problem = create_problem(scene, goal)
    start = time.time()
    solution = solve(
        problem,
        algorithm='adaptive',
        # use_unique=True,
        max_time=30,
        search_sample_ratio=0.01,
        max_planner_time = 30,
        logpath="/tmp/",
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
    solved, duration = plan is not None, end - start
    return solved, duration

def heuristic(state, goal):
    return len(goal - state.state)*10

def run_lifted(scene, goal):
    initial_state, goal, externals, actions, _ = create_problem_lifted(scene, goal)
    start = time.time()
    search = ActionStreamSearch(initial_state, goal, externals, actions)
    stats = {}
    result = repeated_a_star(search, stats=stats, max_steps=10, heuristic=heuristic)
    end = time.time()
    return result is not None, end - start

def run_adaptive_process(scene, goal, scene_idx, rep_idx, res):
    with open(os.devnull, 'w') as f:
        sys.stdout = f
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            res[(scene_idx, rep_idx, 'adaptive')] = run_adaptive(scene, goal)

def run_lifted_process(scene, goal, scene_idx, rep_idx, res):
    with open(os.devnull, 'w') as f:
        sys.stdout = f
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
    for i in range(num_scenes):
        scene = generate_scene([1, 2, 3, 4])
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
    for scene_idx in range(num_scenes):
        for rep in range(num_reps):
            if run_lifted_flag: 
                lsolved, lduration = return_dict[(scene_idx, rep, 'lifted')]
                data.append(dict(alg='lifted', solved=lsolved, duration=lduration, scene_idx=scene_idx, rep_idx=rep))
            if run_adaptive_flag:
                asolved, aduration = return_dict[(scene_idx, rep, 'adaptive')]
                data.append(dict(alg='adaptive', solved=asolved, duration=aduration, scene_idx=scene_idx, rep_idx=rep))

    df = pd.DataFrame(data)
    df.to_csv('temp/comparison_results.csv')

    if run_lifted_flag: 
        lifted_df = df[df.alg == 'lifted']
        print("\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}lifted".format(
            lifted_df["solved"].mean(), lifted_df["solved"].std(),
            lifted_df["duration"].mean(), lifted_df["duration"].std(),
        ))
    if run_adaptive_flag:
        adaptive_df = df[df.alg == 'adaptive']
        print("\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}adaptive".format(
            adaptive_df["solved"].mean(), adaptive_df["solved"].std(),
            adaptive_df["duration"].mean(), adaptive_df["duration"].std(),
        ))
