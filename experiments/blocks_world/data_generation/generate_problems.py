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

    min_num_blocks = 3
    min_num_blockers = 0
    max_num_blocks = 12
    max_num_blockers = 12
    buffer_radius = 0
    max_max_stack = 6
    num_same = 1

    for _ in tqdm(range(100)):
        num_blocks = np.random.randint(min_num_blocks, max_num_blocks + 1)
        num_blockers = np.random.randint(min_num_blockers, max_num_blockers + 1)
        max_stack = np.random.randint(1, min(max_max_stack, num_blocks) + 1)

        for i in range(num_same):
            yaml_data = make_problem.make_random_problem(
                num_blocks=num_blocks,
                num_blockers=num_blockers,
                colorize=True,
                buffer_radius=buffer_radius,
                max_stack_num=max_stack
            )

            outpath = f"{FILEPATH}/random/train/{num_blocks}_{num_blockers}_{max_stack}_{i}_{glob_index}.yaml"
            glob_index += 1
            with open(outpath, "w") as stream:
                yaml.dump(yaml_data, stream, default_flow_style=False)
            print('Written', outpath)


    for _ in tqdm(range(100)):
        num_blocks = np.random.randint(min_num_blocks, max_num_blocks + 1)
        num_blockers = 2*num_blocks
        max_stack = np.random.randint(1, min(max_max_stack, num_blocks) + 1)

        for i in range(num_same):
            yaml_data = make_problem.make_clutter_problem(
                num_blocks=num_blocks,
                num_blockers=num_blockers,
                colorize=True,
                buffer_radius=buffer_radius,
                max_stack_num=max_stack
            )

            outpath = f"{FILEPATH}/clutter/train/{num_blocks}_{num_blockers}_{max_stack}_{i}_{glob_index}.yaml"
            glob_index += 1
            with open(outpath, "w") as stream:
                yaml.dump(yaml_data, stream, default_flow_style=False)
            print('Written', outpath)

    min_num_blocks = 2
    num_same = 10

    for num_blocks in tqdm(range(min_num_blocks, max_num_blocks + 1)):
        for i in range(num_same):
            yaml_data = make_problem.make_non_monotonic_problem(
                num_blocks=num_blocks,
                colorize=True,
                buffer_radius=buffer_radius,
                max_stack_num=1
            )
            outpath = f"{FILEPATH}/non_monotonic/train/{num_blocks}_{num_blocks}_1_{i}_{glob_index}.yaml"
            glob_index += 1
            with open(outpath, "w") as stream:
                yaml.dump(yaml_data, stream, default_flow_style=False)
            print('Written', outpath)

    min_num_blocks = 4
    max_num_blocks = 14
    
    for num_blocks in tqdm(range(min_num_blocks, max_num_blocks + 1)):
        for i in range(num_same):
            yaml_data = make_problem.make_non_monotonic_problem(
                num_blocks=num_blocks,
                colorize=True,
                buffer_radius=buffer_radius,
                max_stack_num=1
            )
            outpath = f"{FILEPATH}/sorting/train/{num_blocks}_{num_blocks}_1_{i}_{glob_index}.yaml"
            glob_index += 1
            with open(outpath, "w") as stream:
                yaml.dump(yaml_data, stream, default_flow_style=False)
            print('Written', outpath)

    min_num_blocks = 4
    max_num_blocks = 14
    num_same = 1
    buffer_radius = 0.05
    
    for _ in tqdm(range(100)):
        num_blocks = np.random.randint(min_num_blocks, max_num_blocks + 1)
        num_blockers = 0
        max_stack = min(max_max_stack, num_blocks)

        for i in range(num_same):
            yaml_data = make_problem.make_random_problem(
                num_blocks=num_blocks,
                num_blockers=num_blockers,
                colorize=True,
                buffer_radius=buffer_radius,
                max_stack_num=max_stack
            )

            outpath = f"{FILEPATH}/stacking/train/{num_blocks}_{num_blockers}_{max_stack}_{i}_{glob_index}.yaml"
            glob_index += 1
            with open(outpath, "w") as stream:
                yaml.dump(yaml_data, stream, default_flow_style=False)
            print('Written', outpath)
