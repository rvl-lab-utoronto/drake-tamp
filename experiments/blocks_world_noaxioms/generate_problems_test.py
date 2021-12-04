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
FILEPATH = os.path.join(FILEPATH, 'data-test')
if __name__ == '__main__':

    # random

    glob_index = 0

    # 100 x:
    # num_blocks: [2,7]
    # num_blocks: [0,6]
    # max_start_stack: 1
    # max_goal_stack: [1,6]

    for _ in tqdm(range(10)):
        num_blocks = np.random.randint(1, 3 + 1)
        num_blockers = np.random.randint(0, 3 + 1)
        max_stack = np.random.randint(1, min(6, num_blocks) + 1)

        yaml_data = make_problem.make_random_problem(
            num_blocks=num_blocks,
            num_blockers=num_blockers,
            buffer_radius=0,
            max_start_stack = 1,
            max_goal_stack =max_stack,
            colorize = True,
        )

        outpath = f"{FILEPATH}/{num_blocks}_{num_blockers}_{max_stack}_random_{glob_index}.yaml"
        glob_index += 1
        with open(outpath, "w") as stream:
            yaml.dump(yaml_data, stream, default_flow_style=False)
        print('Written', outpath)


    # 100 x:
    # num_blocks: [2,6]
    # num_blocks: 2*num_blocks
    # max_start_stack: 1
    # max_goal_stack: [1,6]

    for _ in tqdm(range(10)):
        num_blocks = np.random.randint(1, 3 + 1)
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

        outpath = f"{FILEPATH}/{num_blocks}_{num_blockers}_{max_stack}_clutter_{glob_index}.yaml"
        glob_index += 1
        with open(outpath, "w") as stream:
            yaml.dump(yaml_data, stream, default_flow_style=False)
        print('Written', outpath)

    # grid
    # num_blocks: [2,6]
    # repeat each 15 times

    # for num_blocks in tqdm(range(2, 6 + 1)):

    #     for i in range(20):
    #         yaml_data = make_problem.make_non_monotonic_problem(
    #             num_blocks=num_blocks,
    #             buffer_radius=0,
    #             colorize = True,
    #         )
    #         outpath = f"{FILEPATH}/non_monotonic/{num_blocks}_{num_blocks}_1_{glob_index}.yaml"
    #         glob_index += 1
    #         with open(outpath, "w") as stream:
    #             yaml.dump(yaml_data, stream, default_flow_style=False)
    #         print('Written', outpath)

    # grid
    # num_blocks: [2, 10] 
    # repeat each 12 times
    # max_goal_stack = 1
    
    # for num_blocks in tqdm(range(2, 10 + 1)):
    #     for i in range(12):
    #         yaml_data = make_problem.make_sorting_problem(
    #             num_blocks=num_blocks,
    #             buffer_radius=0,
    #         )
    #         outpath = f"{FILEPATH}/{num_blocks}_0_1_sorting_{glob_index}.yaml"
    #         glob_index += 1
    #         with open(outpath, "w") as stream:
    #             yaml.dump(yaml_data, stream, default_flow_style=False)
    #         print('Written', outpath)

    # random x 50
    # num_blocks: [2, 7] 
    # max_start_stack: [1, 6] 
    # max_goal_stack: [1, 6] 

    # random x 50
    # num_blocks: [2, 7] 
    # max_start_stack: 1
    # max_goal_stack: [1, 6] 
    
    for _ in tqdm(range(10)):
        num_blocks = np.random.randint(2, 3 + 1)
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

            outpath = f"{FILEPATH}/{num_blocks}_{num_blockers}_{max_start_stack}_stacking_{glob_index}.yaml"
            glob_index += 1
            with open(outpath, "w") as stream:
                yaml.dump(yaml_data, stream, default_flow_style=False)
            print('Written', outpath)

    for _ in tqdm(range(10)):
        num_blocks = np.random.randint(2, 3 + 1)
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

            outpath = f"{FILEPATH}/{num_blocks}_{num_blockers}_{max_start_stack}_stacking_{glob_index}.yaml"
            glob_index += 1
            with open(outpath, "w") as stream:
                yaml.dump(yaml_data, stream, default_flow_style=False)
            print('Written', outpath)