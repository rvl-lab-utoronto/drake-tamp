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

    base_output = f"{FILEPATH}/perception_random/train/" 
    try: 
        os.makedirs(base_output)
    except OSError: 
        print("failed to create directory %s" % base_output)

    for _ in tqdm(range(25000)):
        num_blocks = np.random.randint(1, 3 + 1)
        num_blockers = np.random.randint(0, 6 + 1)
        max_stack = 1#np.random.randint(1, min(6, num_blocks) + 1)

        yaml_data = make_problem.make_random_problem(
            num_blocks=num_blocks,
            num_blockers=num_blockers,
            buffer_radius=0,
            max_start_stack = 1,
            max_goal_stack =max_stack,
            colorize = True,
        )
        

        outpath = base_output+ f"{num_blocks}_{num_blockers}_{max_stack}_{glob_index}.yaml"
        
        glob_index += 1
        with open(outpath, "w+") as stream:
            yaml.dump(yaml_data, stream, default_flow_style=False)
        print('Written', outpath)
