import matplotlib
import numpy as np
import yaml
from tqdm import tqdm

matplotlib.use("Agg")
import yaml
from experiments.blocks_world.data_generation import make_problem
import itertools
import os
FILEPATH, _ = os.path.split(os.path.realpath(__file__))

if __name__ == '__main__':

    # random

    glob_index = 0

    # 100 x:
    # num_blocks: [1,6]
    # num_blocks: [0,6]
    # max_start_stack: 1
    # max_goal_stack: [1,6]

    for _ in tqdm(range(100)):
        num_blocks = np.random.randint(1, 6 + 1)
        num_blockers = np.random.randint(0, 6 + 1)
        max_stack = np.random.randint(1, min(6, num_blocks) + 1)

        yaml_data = make_problem.make_random_problem(
            num_blocks=num_blocks,
            num_blockers=num_blockers,
            buffer_radius=0,
            max_start_stack = 1,
            max_goal_stack =max_stack,
            colorize = True,
        )

        outpath = f"{FILEPATH}/random/train/{num_blocks}_{num_blockers}_{max_stack}_{glob_index}.yaml"
        glob_index += 1
        with open(outpath, "w") as stream:
            yaml.dump(yaml_data, stream, default_flow_style=False)
        print('Written', outpath)

    glob_index = 0

    # 100 x:
    # num_blocks: [2,5]
    # num_blocks: 2*num_blocks
    # max_start_stack: 1
    # max_goal_stack: [1,5]

    for _ in tqdm(range(100)):
        num_blocks = np.random.randint(2, 5 + 1)
        num_blockers = 2*num_blocks
        max_stack = np.random.randint(1, min(6, num_blocks) + 1)

        yaml_data = make_problem.make_clutter_problem(
            num_blocks=num_blocks,
            num_blockers=num_blockers,
            buffer_radius=0,
            max_start_stack =1,
            max_goal_stack=max_stack,
            colorize = True,
        )

        outpath = f"{FILEPATH}/clutter/train/{num_blocks}_{num_blockers}_{max_stack}_{glob_index}.yaml"
        glob_index += 1
        with open(outpath, "w") as stream:
            yaml.dump(yaml_data, stream, default_flow_style=False)
        print('Written', outpath)

    # grid
    # num_blocks: [1,3]
    # repeat each 33 times
    glob_index = 0

    for num_blocks in tqdm(range(1, 3 + 1)):

        for i in range(33):
            yaml_data = make_problem.make_non_monotonic_problem(
                num_blocks=num_blocks,
                buffer_radius=0,
                colorize = True,
            )
            outpath = f"{FILEPATH}/non_monotonic/train/{num_blocks}_{num_blocks}_1_{glob_index}.yaml"
            glob_index += 1
            with open(outpath, "w") as stream:
                yaml.dump(yaml_data, stream, default_flow_style=False)
            print('Written', outpath)

    # grid
    # num_blocks: [2, 7] 
    # repeat each 17 times
    # total of 102
    # max_goal_stack = 1
    glob_index = 0
    
    for num_blocks in tqdm(range(2, 7 + 1)):
        for i in range(17):
            yaml_data = make_problem.make_sorting_problem(
                num_blocks=num_blocks,
                buffer_radius=0,
            )
            outpath = f"{FILEPATH}/sorting/train/{num_blocks}_{0}_1_{glob_index}.yaml"
            glob_index += 1
            with open(outpath, "w") as stream:
                yaml.dump(yaml_data, stream, default_flow_style=False)
            print('Written', outpath)

    # random x 50
    # num_blocks: [2, 4] 
    # max_start_stack: [1, 6] 
    # max_goal_stack: [1, 6] 

    # random x 50
    # num_blocks: [2, 4] 
    # max_start_stack: 1
    # max_goal_stack: [1, 6] 
    glob_index = 0
    
    for _ in tqdm(range(50)):
        num_blocks = np.random.randint(2, 4 + 1)
        num_blockers = 0
        max_start_stack = min(6, num_blocks)
        max_goal_stack = min(6, num_blocks)

        for i in range(1):
            yaml_data = make_problem.make_random_problem(
                num_blocks=num_blocks,
                num_blockers=num_blockers,
                buffer_radius=0,
                max_start_stack = max_start_stack,
                max_goal_stack=max_goal_stack,
                colorize = True,
            )

            outpath = f"{FILEPATH}/stacking/train/{num_blocks}_{num_blockers}_{max_start_stack}_{glob_index}.yaml"
            glob_index += 1
            with open(outpath, "w") as stream:
                yaml.dump(yaml_data, stream, default_flow_style=False)
            print('Written', outpath)

    for _ in tqdm(range(50)):
        num_blocks = np.random.randint(2, 4 + 1)
        num_blockers = 0
        max_start_stack = 1
        max_goal_stack = min(6, num_blocks)

        for i in range(1):
            yaml_data = make_problem.make_random_problem(
                num_blocks=num_blocks,
                num_blockers=num_blockers,
                buffer_radius=0,
                max_start_stack = max_start_stack,
                max_goal_stack=max_goal_stack,
                colorize = True,
            )

            outpath = f"{FILEPATH}/stacking/train/{num_blocks}_{num_blockers}_{max_start_stack}_{glob_index}.yaml"
            glob_index += 1
            with open(outpath, "w") as stream:
                yaml.dump(yaml_data, stream, default_flow_style=False)
            print('Written', outpath)

    # grid
    # num_blocks: [1,3]
    # repeat each 33 times
    glob_index = 0

    for num_blocks in tqdm(range(1, 3 + 1)):

        for i in range(33):
            yaml_data = make_problem.make_non_monotonic_problem_v2(
                num_blocks=num_blocks,
                buffer_radius=0,
                colorize = True,
            )
            outpath = f"{FILEPATH}/non_monotonic_v2/train/{num_blocks}_{num_blocks}_1_{glob_index}.yaml"
            glob_index += 1
            with open(outpath, "w") as stream:
                yaml.dump(yaml_data, stream, default_flow_style=False)
            print('Written', outpath)
