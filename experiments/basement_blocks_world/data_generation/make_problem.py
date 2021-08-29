import os
import time
import itertools
from panda_station.models.blocks_world.sdf.make_blocks import TEMPLATE_NAME
from learning.poisson_disc_sampling import PoissonSampler
import numpy as np
import yaml
import xml.etree.ElementTree as ET

np.random.seed(seed=int(time.time()))

DIRECTIVE = os.path.expanduser(
    "~/drake-tamp/panda_station/directives/basement.yaml"
)

TABLE_HEIGHT = 0.74
BLOCK_DIMS = np.array([0.05, 0.05, 0.05])
R = max(BLOCK_DIMS[:2]) * np.sqrt(2) + 0.01
WOODEN_TABLE_DIMS = np.array([0.76, 1.215]) - np.ones(2) * R
BLOCKER_DIMS = np.array([0.05, 0.05, 0.1])

TABLE_CENTER = np.array([0.755, 0])
VALID_POS = [np.array([0.1*x,0.1*y]) + TABLE_CENTER for x in range(-1, 1) for y in range(-3,4)]

TEMPLATE_PATH = "models/blocks_world/sdf/red_block.sdf"
MODELS_PATH = "models/blocks_world/sdf/"

def make_block(block_name, color, size, buffer, ball_radius=1e-7):
    tree = ET.parse(
        os.path.expanduser(
            "~/drake-tamp/panda_station/models/blocks_world/sdf/template_block.sdf"
        )
    )
    root = tree.getroot()
    for model in root.iter("model"):
        model.set("name", block_name)
    for diffuse in root.iter("diffuse"):
        diffuse.text = color

    model = root[0]
    base = model[0]
    for elm in base:
        name = elm.attrib.get("name", "")
        if name.startswith("ball"):
            num = int(name[-1]) - 1
            x = (-1) ** (num & 0b01) * size[0]
            y = (-1) ** ((num & 0b10) >> 1) * size[1]
            z = ((num & 0b100) >> 2) * size[2]
            elm.find("pose").text = f"{x} {y} {z} 0 0 0"
            elm.find("geometry").find("sphere").find("radius").text = f"{ball_radius}"
        elif name == "visual":
            w = size[0]
            d = size[1]
            h = size[2]
            z = h / 2
            elm.find("pose").text = f"0 0 {z} 0 0 0"
            elm.find("geometry").find("box").find("size").text = f"{w} {d} {h}"
        elif name == "collision":
            w = size[0] - buffer * 2
            d = size[1] - buffer * 2
            h = size[2] - buffer * 2
            z = size[2] / 2
            elm.find("pose").text = f"0 0 {z} 0 0 0"
            elm.find("geometry").find("box").find("size").text = f"{w} {d} {h}"

    tree.write(
        os.path.expanduser(
            f"~/drake-tamp/panda_station/models/blocks_world/sdf/{block_name}.sdf"
        )
    )
    return tree

def make_random_stacking(blocks, max_stack_num = None, num_stacks = None):
    num_blocks = len(blocks)
    block_perm = blocks.copy()
    np.random.shuffle(block_perm)
    stacking = set()
    
    if max_stack_num is not None:
        assert (
            0 < max_stack_num <= num_blocks
        ), "Max stack height must be a integer greater than 0 and less than the number of blocks"
        num_max_stacks = num_blocks//max_stack_num
        num_stack = max_stack_num
        while len(block_perm) > 0:
            stacking.add(tuple(block_perm[:num_stack]))
            block_perm = block_perm[num_stack:]
            num_stack = min(np.random.randint(1, max_stack_num + 1), len(block_perm))
    else:
        lower_num = 0
        if num_blocks == 0:
            return stacking

        if num_blocks == 1:
            return set([tuple(block_perm)]) | stacking

        if num_stacks is None:
            num_splits = np.random.randint(lower_num, num_blocks)
        else:
            assert num_stacks >= 0 and num_stacks < num_blocks, "Invalid stack number"
            num_splits = num_stacks - 1
        split_locs = np.random.choice(
            list(range(1, num_blocks)), size=num_splits, replace=False
        )
        split_locs.sort()
        split_locs = np.append(split_locs, num_blocks)
        i = 0
        for split_loc in split_locs:
            stacking.add(tuple(block_perm[i:split_loc]))
            i = split_loc
    return stacking


def make_random_problem(num_blocks, colorize=True, buffer_radius=0, max_start_stack = None, max_goal_stack = None):
    """
    buffer_radius is an addition to the minimum distance (in the same units as the extent
    - for our purposes it is meters)
    between two objects (which is currently ~1cm).
    """
    valid_positions = VALID_POS.copy()
    np.random.shuffle(valid_positions)
    positions = {"wooden_table": valid_positions}

    yaml_data = {
        "directive": "directives/basement.yaml",
        "planning_directive": "directives/basement.yaml",
        "arms": {
            "panda": {
                "panda_name": "panda",
                "hand_name": "hand",
                "X_WB": [0.05, 0, 0.8, 0, 0, 0],
            }
        },
        "objects": {},
        "main_links": {
            "wooden_table": "base_link",
            "thor_table": "base_link"
        },
        "surfaces": {
            "wooden_table": ["base_link"]
        },
    }


    blocks = [f"block{i}" for i in range(num_blocks)]
    stacking = make_random_stacking(blocks, max_stack_num=max_start_stack)
    max_start_stack = max([len(s) for s in stacking])

    added_blocks = []
    for stack in stacking:
        table = "wooden_table"
        if len(positions) == 0:
            print("Warning: failed to add all desired blocks")
            break
        added_blocks += stack
        point = positions[table].pop(-1)
        point = np.append(point, TABLE_HEIGHT)
        point = np.concatenate((point, np.zeros(3)))
        yaw = 0
        block = stack[0]
        path = TEMPLATE_PATH
        if colorize:
            color = np.random.uniform(np.zeros(3), np.ones(3))
            color = f"{color[0]} {color[1]} {color[2]} 1"
            path = MODELS_PATH + block + ".sdf"
            make_block(block_name=block, color=color, size=BLOCK_DIMS, buffer=0.001)
        yaml_data["objects"][block] = {
            "path": path,
            "X_WO": point.tolist(),
            "main_link": "base_link",
            "on-table": [str(table), "base_link"],
        }
        prev_block = block
        for i, block in enumerate(stack[1:]):
            point[2] += BLOCK_DIMS[2] + 1e-3
            yaw = (np.pi/4) * ((i+1)%2)
            point[-1] = yaw 
            path = TEMPLATE_PATH
            if colorize:
                color = np.random.uniform(np.zeros(3), np.ones(3))
                color = f"{color[0]} {color[1]} {color[2]} 1"
                path = MODELS_PATH + block + ".sdf"
                make_block(block_name=block, color=color, size=BLOCK_DIMS, buffer=0.001)
            yaml_data["objects"][block] = {
                "path": path,
                "X_WO": point.tolist(),
                "main_link": "base_link",
                "on-block": prev_block,
            }
            prev_block = block

    stacking = make_random_stacking(added_blocks, max_stack_num=max_goal_stack)
    max_goal_stack = max([len(s) for s in stacking])
    goal = ["and"]
    for stack in stacking:
        table = "wooden_table"
        base_block = stack[0]
        goal.append(["on-table", base_block, [str(table), "base_link"]])
        prev_block = base_block
        for block in stack[1:]:
            goal.append(["on-block", block, prev_block])
            prev_block = block

    yaml_data["goal"] = goal

    yaml_data["run_attr"] = {
        "num_blocks": len(added_blocks),
        "max_start_stack": max_start_stack,
        "max_goal_stack": max_goal_stack,
        "buffer_radius": buffer_radius,
        "type": "basement_blocks_world"
    }

    return yaml_data


if __name__ == "__main__":

    yaml_data = make_random_problem(num_blocks=5, max_start_stack=1, max_goal_stack = 5, colorize=True)
    with open("test_problem.yaml", "w") as stream:
        yaml.dump(yaml_data, stream, default_flow_style=False)
