import os
import time
import itertools
from panda_station.models.blocks_world.sdf.make_blocks import TEMPLATE_NAME
from learning.poisson_disc_sampling import PoissonSampler
import numpy as np
import yaml
import xml.etree.ElementTree as ET
np.random.seed(seed = int(time.time()))

DIRECTIVE = os.path.expanduser(
    "~/drake-tamp/panda_station/directives/one_arm_blocks_world.yaml"
)

TABLE_HEIGHT = 0.325
BLOCK_DIMS = np.array([0.045, 0.045, 0.045])
R = max(BLOCK_DIMS[:2]) * np.sqrt(2)
X_TABLE_DIMS = np.array([0.4, 0.75]) - np.ones(2) * R
Y_TABLE_DIMS = np.array([0.75, 0.4]) - np.ones(2) * R
BLOCKER_DIMS = np.array([0.045, 0.045, 0.1])

# table_name: (center point, extent)
TABLES = {
    "red_table": [
        np.array([0.6, 0]),
        PoissonSampler(X_TABLE_DIMS, r=R, centered=True),
    ],
    "blue_table": [
        np.array([-0.6, 0]),
        PoissonSampler(X_TABLE_DIMS, r=R, centered=True),
    ],
    "green_table": [
        np.array([0, 0.6]),
        PoissonSampler(Y_TABLE_DIMS, r=R, centered=True),
    ],
    "purple_table": [
        np.array([0, -0.6]),
        PoissonSampler(Y_TABLE_DIMS, r=R, centered=True),
    ],
}

TEMPLATE_PATH = "models/blocks_world/sdf/red_block.sdf"
MODELS_PATH = "models/blocks_world/sdf/"


def pick_random_table():
    return np.random.choice(list(TABLES.keys()))


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


def make_random_problem(num_blocks, num_blockers, colorize=False, buffer_radius = 0):
    """
    buffer_radius is an addition to the minimum distance (in the same units as the extent
    - for our purposes it is meters)
    between two objects (which is currently ~1cm).
    """

    positions = {}
    for name, item in TABLES.items():
        sampler = PoissonSampler(
            np.clip(item[1].extent - buffer_radius, 0, np.inf),
            item[1].r + buffer_radius,
            centered=True,
        )
        points = sampler.make_samples(num = (num_blocks + num_blockers)*10)
        np.random.shuffle(points)
        positions[name] = points

    yaml_data = {
        "directive": "directives/one_arm_blocks_world.yaml",
        "planning_directive": "directives/one_arm_blocks_world.yaml",
        "arms": {
            "panda": {
                "panda_name": "panda",
                "hand_name": "hand",
                "X_WB": [0, 0, 0, 0, 0, 0],
            }
        },
        "objects": {},
        "main_links": {
            "red_table": "base_link",
            "blue_table": "base_link",
            "green_table": "base_link",
            "purple_table": "base_link",
        },
        "surfaces": {
            "red_table": ["base_link"],
            "green_table": ["base_link"],
            "blue_table": ["base_link"],
            "purple_table": ["base_link"],
        },
    }

    blocks = [f"block{i}" for i in range(num_blocks)]
    blockers = [f"blocker{i}" for i in range(num_blockers)]
    stacking = make_random_stacking(blocks)
    for stack in stacking:
        table = pick_random_table()
        if len(positions[table]) == 0:
            res = TABLES[table][1].sample()
            if res is None: 
                continue
        point = positions[table].pop(-1) + TABLES[table][0]
        point = np.append(point, TABLE_HEIGHT)
        point = np.concatenate((point, np.zeros(3)))
        point[-1] = np.random.uniform(0, 2*np.pi)
        block = stack[0]
        path = TEMPLATE_PATH
        if colorize:
            color = np.random.uniform(np.zeros(3), np.ones(3))
            color = f"{color[0]} {color[1]} {color[2]} 1"
            path = MODELS_PATH + block + ".sdf"
            make_block(
                block_name=block, color=color, size=BLOCK_DIMS, buffer=0.001
            )
        yaml_data["objects"][block] = {
            "path": path,
            "X_WO": point.tolist(),
            "main_link": "base_link",
            "on-table": [str(table), "base_link"],
        }
        prev_block = block
        for block in stack[1:]:
            point[2] += BLOCK_DIMS[2] + 0.01
            point[-1] = np.random.uniform(0, 2*np.pi)
            path = TEMPLATE_PATH
            if colorize:
                color = np.random.uniform(np.zeros(3), np.ones(3))
                color = f"{color[0]} {color[1]} {color[2]} 1"
                path = MODELS_PATH + block + ".sdf"
                make_block(
                    block_name=block, color=color, size=BLOCK_DIMS, buffer=0.001
                )
            yaml_data["objects"][block] = {
                "path": path,
                "X_WO": point.tolist(),
                "main_link": "base_link",
                "on-block": prev_block,
            }
            prev_block = block

    stacking = make_random_stacking(blocks)
    goal = ["and"]
    for stack in stacking:
        table = pick_random_table()
        base_block = stack[0]
        goal.append(["on-table", base_block, [str(table), "base_link"]])
        prev_block = base_block
        for block in stack[1:]:
            goal.append(["on-block", block, prev_block])
            prev_block = block

    yaml_data["goal"] = goal

    for blocker in blockers:
        table = pick_random_table()
        if len(positions[table]) == 0:
            res = TABLES[table][1].sample()
            if res is None:
                continue
            positions[table].append(res)
        point = positions[table].pop(-1) + TABLES[table][0]
        point = np.append(point, TABLE_HEIGHT)
        point = np.concatenate((point, np.zeros(3)))
        point[-1] = np.random.uniform(0, 2*np.pi)
        yaml_data["objects"][blocker] = {
            "path": "models/blocks_world/sdf/blocker_block.sdf",
            "X_WO": point.tolist(),
            "main_link": "base_link",
            "on-table": [str(table), "base_link"],
        }

    return yaml_data


def make_random_stacking(blocks, num_stacks=None):
    num_blocks = len(blocks)
    block_perm = blocks.copy()
    np.random.shuffle(block_perm)
    if num_stacks is None:
        num_splits = np.random.randint(0, num_blocks)
    else:
        assert num_stacks >= 0 and num_stacks < num_blocks, "Invalid stack number"
        num_splits = num_stacks - 1
    split_locs = np.random.choice(
        list(range(1, num_blocks)), size=num_splits, replace=False
    )
    split_locs.sort()
    split_locs = np.append(split_locs, num_blocks)
    stacking = set()
    i = 0
    for split_loc in split_locs:
        stacking.add(tuple(block_perm[i:split_loc]))
        i = split_loc
    return stacking


def make_stackings(blocks):
    num_blocks = len(blocks)
    stackings = set()
    for block_perm in itertools.permutations(blocks):
        for num_splits in range(num_blocks):
            # num_groups = num_splits + 1
            for split_locs in itertools.combinations(
                range(1, num_blocks), r=num_splits
            ):
                grouping = ()
                i = 0
                split_locs += (num_blocks,)
                for split_loc in split_locs:
                    grouping += (block_perm[i:split_loc],)
                    i = split_loc
                stackings.add(frozenset(grouping))
    return stackings


if __name__ == "__main__":
    # specify:
    # number of blocks
    # number of blockers
    # iterate through all possible initial/goal stackings of blocks
    # randomly place the initial stacks/blocks using poisson disc
    # randomly assign each stack of a goal table

    yaml_data = make_random_problem(num_blocks=5, num_blockers=5, colorize=True)
    with open("test_problem.yaml", "w") as stream:
        yaml.dump(yaml_data, stream, default_flow_style=False)

    pass
