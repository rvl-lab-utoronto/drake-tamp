import time

import pandas as pd
from integrate_planner import generate_scene, instantiate_planning_problem, externals
from lifted_search import ActionStreamSearch, repeated_a_star, Atom
from experiments.gripper2d.run import create_problem, solve, StreamInfo
import multiprocessing, sys, os
import tempfile

def run_adaptive(scene):
    goal = ('and', 
        ('on', 'b0', 'r2'),
        ('on', 'b1', 'r1'),
        ('on', 'b2', 'r1'),
        ('on', 'b3', 'r1')
    )
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

def run_lifted(scene):
    objects, actions, initial_state = instantiate_planning_problem(scene)
    goal = set()
    goal.add(Atom('on', (objects['b0'].pddl, objects['r2'].pddl)))
    goal.add(Atom('on', (objects['b1'].pddl, objects['r1'].pddl)))
    goal.add(Atom('on', (objects['b2'].pddl, objects['r1'].pddl)))
    goal.add(Atom('on', (objects['b3'].pddl, objects['r1'].pddl)))

    start = time.time()
    search = ActionStreamSearch(initial_state, goal, externals, actions)
    stats = {}
    result = repeated_a_star(search, stats=stats, max_steps=10, heuristic=heuristic)
    end = time.time()
    return result is not None, end - start

def run_adaptive_process(scene, scene_idx, rep_idx, res):
    with open(os.devnull, 'w') as f:
        sys.stdout = f
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            res[(scene_idx, rep_idx, 'adaptive')] = run_adaptive(scene)

def run_lifted_process(scene, scene_idx, rep_idx, res):
    with open(os.devnull, 'w') as f:
        sys.stdout = f
        res[(scene_idx, rep_idx, 'lifted')] = run_lifted(scene)


if __name__ == '__main__':
    num_scenes = 50
    num_reps = 6
    scenes = []
    data = []
    for i in range(num_scenes):
        scene = generate_scene([1, 2, 3, 4])
        scenes.append(scene)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for scene_idx in range(num_scenes):
        jobs = []
        for rep in range(num_reps):
            p = multiprocessing.Process(target=run_adaptive_process, args=(scenes[scene_idx], scene_idx, rep, return_dict,))
            p.start()
            jobs.append(p)
            
        for p in jobs:
            p.join()


    for scene_idx in range(num_scenes):
        jobs = []
        for rep in range(num_reps):
            p = multiprocessing.Process(target=run_lifted_process, args=(scenes[scene_idx], scene_idx, rep, return_dict,))
            p.start()
            jobs.append(p)
            
        for p in jobs:
            p.join()


    data = []
    for scene_idx in range(num_scenes):
        for rep in range(num_reps):
            lsolved, lduration = return_dict[(scene_idx, rep, 'lifted')]
            asolved, aduration = return_dict[(scene_idx, rep, 'adaptive')]
            data.append(dict(alg='lifted', solved=lsolved, duration=lduration, scene_idx=scene_idx, rep_idx=rep))
            data.append(dict(alg='adaptive', solved=asolved, duration=aduration, scene_idx=scene_idx, rep_idx=rep))



    df = pd.DataFrame(data)
    df.to_csv('comparison_results.csv')