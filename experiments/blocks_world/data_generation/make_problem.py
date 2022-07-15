import os
import time
import itertools
from learning.poisson_disc_sampling import PoissonSampler, GridSampler
import numpy as np
import yaml
import xml.etree.ElementTree as ET

np.random.seed(seed=int(time.time()))

DIRECTIVE = os.path.expanduser(
    "~/drake-tamp/panda_station/directives/one_arm_blocks_world.yaml"
)

TABLE_HEIGHT = 0.325
BLOCK_DIMS = np.array([0.045, 0.045, 0.045])
R = max(BLOCK_DIMS[:2]) * np.sqrt(2)
X_TABLE_DIMS = np.array([0.4, 0.75]) - np.ones(2) * R
Y_TABLE_DIMS = np.array([0.75, 0.4]) - np.ones(2) * R
BLOCKER_DIMS = np.array([0.045, 0.045, 0.1])
ARM_POS = np.array([0,0])
MAX_ARM_REACH = 0.7 # Note: the actual limit is 0.855, https://www.generationrobots.com/media/panda-franka-emika-datasheet.pdf


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
            x = (-1) ** (num & 0b01) * (size[0] / 2)
            y = (-1) ** ((num & 0b10) >> 1) * (size[1] / 2)
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

def make_random_problem(num_blocks, num_blockers, colorize=False, buffer_radius=0, max_start_stack = None, max_goal_stack = None, prioritize_grouping = False, clump = False, grid = False, rand_rotation = True, type = "random"):
    """
    buffer_radius is an addition to the minimum distance (in the same units as the extent
    - for our purposes it is meters)
    between two objects (which is currently ~1cm).
    """
    
    filter = lambda point: np.linalg.norm((point + item[0]) - ARM_POS) > MAX_ARM_REACH

    if grid:
        rand_rotation = False

    positions = {}
    samplers= {}
    for name, item in TABLES.items():
        if grid:
            sampler = GridSampler(
                np.clip(item[1].extent - buffer_radius, 0, np.inf),
                np.max(BLOCK_DIMS) + buffer_radius,
                centered=True,
            )
        else:
            sampler = PoissonSampler(
                np.clip(item[1].extent - buffer_radius, 0, np.inf),
                item[1].r + buffer_radius,
                centered=True,
                clump = clump
            )
        samplers[name] = sampler
        points = sampler.make_samples(num=(num_blocks + num_blockers) * 10, filter = filter)
        if not prioritize_grouping:
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
    stacking = make_random_stacking(blocks, max_stack_num=max_start_stack)
    max_start_stack = max([len(s) for s in stacking])


    for stack in stacking:
        table = pick_random_table()
        if len(positions[table]) == 0:
            res = samplers[table].make_samples(filter = filter)
            if len(res) == 0:
                continue
            positions[table] += res
        point = positions[table].pop(-1) + TABLES[table][0]
        point = np.append(point, TABLE_HEIGHT)
        point = np.concatenate((point, np.zeros(3)))
        if rand_rotation:
            yaw = np.random.uniform(0, 2 * np.pi)
        else:
            yaw = 0
        point[-1] = yaw
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
        for block in stack[1:]:
            point[2] += BLOCK_DIMS[2] + 1e-3
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

    stacking = make_random_stacking(blocks, max_stack_num=max_goal_stack)
    max_goal_stack = max([len(s) for s in stacking])
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
            res = samplers[table].make_samples(filter = filter)
            if len(res) == 0:
                continue
            positions[table] += res
        point = positions[table].pop(-1) + TABLES[table][0]
        point = np.append(point, TABLE_HEIGHT)
        point = np.concatenate((point, np.zeros(3)))
        if rand_rotation:
            point[-1] = np.random.uniform(0, 2 * np.pi)
        yaml_data["objects"][blocker] = {
            "path": "models/blocks_world/sdf/blocker_block.sdf",
            "X_WO": point.tolist(),
            "main_link": "base_link",
            "on-table": [str(table), "base_link"],
        }

    yaml_data["run_attr"] = {
        "num_blocks": num_blocks,
        "num_blockers": num_blockers,
        "max_start_stack": max_start_stack,
        "max_goal_stack": max_goal_stack,
        "buffer_radius": buffer_radius,
        "type": type
    }

    return yaml_data


def make_clutter_problem(num_blocks, num_blockers = None, tighten_clumping = False, max_start_stack = None, max_goal_stack = None, buffer_radius = 0, colorize = False, grid = False):
    if num_blockers is None:
        num_blockers = num_blocks*3
    return make_random_problem(num_blocks, num_blockers, colorize=colorize, buffer_radius=buffer_radius, max_goal_stack = max_goal_stack, max_start_stack = max_start_stack, prioritize_grouping = True, clump = tighten_clumping, grid = grid, type = "clutter")

def make_non_monotonic_problem(num_blocks, clump = False, buffer_radius = 0, prioritize_grouping = False, colorize = False, max_goal_stack = 1):

    num_blockers = num_blocks
    filter = lambda point: np.linalg.norm((point + item[0]) - ARM_POS) > MAX_ARM_REACH

    positions = {}
    samplers= {}
    for name, item in TABLES.items():
        sampler = PoissonSampler(
            np.clip(item[1].extent - buffer_radius, 0, np.inf),
            2*item[1].r + buffer_radius,
            centered=True,
            clump = clump
        )
        samplers[name] = sampler
        points = sampler.make_samples(num=num_blocks * 10, filter = filter)
        if not prioritize_grouping:
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
    blocker_points = {}

    start_tables = ["red_table", "green_table"]

    for i, (block, blocker) in enumerate(zip(blocks, blockers)):
        table = np.random.choice(start_tables) 
        if len(positions[table]) == 0:
            res = samplers[table].make_samples(filter = filter)
            if len(res) == 0:
                continue
            positions[table] += res
        point = positions[table].pop(-1) + TABLES[table][0]
        point = np.append(point, TABLE_HEIGHT)
        point = np.concatenate((point, np.zeros(3)))
        path = TEMPLATE_PATH
        if colorize:
            color = np.random.uniform(np.zeros(3), np.ones(3))
            color = f"{color[0]} {color[1]} {color[2]} 1"
            path = MODELS_PATH + block + ".sdf"
            make_block(block_name=block, color=color, size=BLOCK_DIMS, buffer=0.001)

        block_point = point.copy()
        blocker_point = point.copy()

        if table == start_tables[0]:
            block_point[0] += np.max(BLOCK_DIMS)/2 + 0.005
            blocker_point[0] -= np.max(BLOCK_DIMS)/2 
        elif table == start_tables[1]:
            block_point[1] += np.max(BLOCK_DIMS)/2 + 0.005
            blocker_point[1] -= np.max(BLOCK_DIMS)/2 

        block_point = block_point.tolist()
        blocker_point = blocker_point.tolist()
        blocker_points[i] = blocker_point

        yaml_data["objects"][block] = {
            "path": path,
            "X_WO": block_point,
            "main_link": "base_link",
            "on-table": [str(table), "base_link"],
        }
        yaml_data["objects"][blocker] = {
            "path": "models/blocks_world/sdf/blocker_block.sdf",
            "X_WO": blocker_point,
            "main_link": "base_link",
            "on-table": [str(table), "base_link"],
        }

    end_tables = ["blue_table", "purple_table"]

    goal = ["and"]

    stacking = make_random_stacking(blocks, max_stack_num=max_goal_stack)
    for stack in stacking:
        table = np.random.choice(end_tables)
        base_block = stack[0]
        goal.append(["on-table", base_block, [str(table), "base_link"]])
        prev_block = base_block
        for block in stack[1:]:
            goal.append(["on-block", block, prev_block])
            prev_block = block

    for i, pt in blocker_points.items():
        goal.append(["atworldpose", blockers[i], pt])

    yaml_data["goal"] = goal

    yaml_data["run_attr"] = {
        "num_blocks": num_blocks,
        "num_blockers": num_blockers,
        "max_start_stack": 1,
        "max_goal_stack": max_goal_stack,
        "buffer_radius": buffer_radius,
        "type": "non_monotonic"
    }

    return yaml_data

def make_sorting_problem(num_blocks, num_blockers = None, buffer_radius=0, max_start_stack = 1, max_goal_stack = 1, prioritize_grouping = False, clump = False, grid = False, rand_rotation = True):

    if num_blockers == None:
        num_blockers = num_blocks

    num_red = num_blocks//2
    num_green = num_blocks-num_red


    filter = lambda point: np.linalg.norm((point + item[0]) - ARM_POS) > MAX_ARM_REACH

    if grid:
        rand_rotation = False

    positions = {}
    samplers= {}
    for name, item in TABLES.items():
        if grid:
            sampler = GridSampler(
                np.clip(item[1].extent - buffer_radius, 0, np.inf),
                np.max(BLOCK_DIMS) + buffer_radius,
                centered=True,
            )
        else:
            sampler = PoissonSampler(
                np.clip(item[1].extent - buffer_radius, 0, np.inf),
                item[1].r + buffer_radius,
                centered=True,
                clump = clump
            )
        samplers[name] = sampler
        points = sampler.make_samples(num=(num_blocks + num_blockers) * 10, filter = filter)
        if not prioritize_grouping:
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

    red_blocks =[f"red_block{i}" for i in range(num_red)]
    green_blocks = [f"green_block{i}" for i in range(num_green)]
    blocks = red_blocks + green_blocks
    np.random.shuffle(blocks) 
    blockers = [f"blocker{i}" for i in range(num_blockers)]

    stacking = make_random_stacking(blocks, max_stack_num=max_start_stack)
    max_start_stack = max([len(s) for s in stacking])
    start_tables = ["purple_table", "blue_table"]

    for stack in stacking:
        table = np.random.choice(start_tables)
        if len(positions[table]) == 0:
            res = samplers[table].make_samples(filter = filter)
            if len(res) == 0:
                continue
            positions[table] += res
        point = positions[table].pop(-1) + TABLES[table][0]
        point = np.append(point, TABLE_HEIGHT)
        point = np.concatenate((point, np.zeros(3)))
        if rand_rotation:
            yaw = np.random.uniform(0, 2 * np.pi)
        else:
            yaw = 0
        point[-1] = yaw
        block = stack[0]
        name = block.split("_")[0]+ "_block.sdf"
        path = os.path.join(MODELS_PATH, name)
        yaml_data["objects"][block] = {
            "path": path,
            "X_WO": point.tolist(),
            "main_link": "base_link",
            "on-table": [str(table), "base_link"],
        }
        prev_block = block
        for block in stack[1:]:
            point[2] += BLOCK_DIMS[2] + 1e-3
            point[-1] = yaw
            name = block.split("_")[0]+ "_block.sdf"
            path = os.path.join(MODELS_PATH, name)
            yaml_data["objects"][block] = {
                "path": path,
                "X_WO": point.tolist(),
                "main_link": "base_link",
                "on-block": prev_block,
            }
            prev_block = block

    goal = ["and"]
    for blocks,table in [(red_blocks, "red_table"), (green_blocks, "green_table")]:
        stacking = make_random_stacking(blocks, max_stack_num=max_goal_stack)
        max_goal_stack = max([len(s) for s in stacking] + [max_goal_stack])
        for stack in stacking:
            base_block = stack[0]
            goal.append(["on-table", base_block, [str(table), "base_link"]])
            prev_block = base_block
            for block in stack[1:]:
                goal.append(["on-block", block, prev_block])
                prev_block = block


    for blocker in blockers:
        table = np.random.choice(start_tables)
        if len(positions[table]) == 0:
            res = samplers[table].make_samples(filter = filter)
            if len(res) == 0:
                continue
            positions[table] += res
        point = positions[table].pop(-1) + TABLES[table][0]
        point = np.append(point, TABLE_HEIGHT)
        point = np.concatenate((point, np.zeros(3)))
        if rand_rotation:
            point[-1] = np.random.uniform(0, 2 * np.pi)
        yaml_data["objects"][blocker] = {
            "path": "models/blocks_world/sdf/blocker_block.sdf",
            "X_WO": point.tolist(),
            "main_link": "base_link",
            "on-table": [str(table), "base_link"],
        }
        goal.append(["on-table", blocker, [str(table), "base_link"]])

    yaml_data["goal"] = goal

    yaml_data["run_attr"] = {
        "num_blocks": num_blocks,
        "num_blockers": num_blockers,
        "max_start_stack": max_start_stack,
        "max_goal_stack": max_goal_stack,
        "buffer_radius": buffer_radius,
        "type": "sorting"
    }

    return yaml_data


def make_distractor_problem(num_blocks, num_blockers, colorize=False, buffer_radius=0, max_start_stack = None, max_goal_stack = None, prioritize_grouping = False, clump = False, grid = False, rand_rotation = True, type = "random"):
    """
    buffer_radius is an addition to the minimum distance (in the same units as the extent
    - for our purposes it is meters)
    between two objects (which is currently ~1cm).
    """
    def pick_random_table_block():
        return np.random.choice(list(TABLES.keys())[:2])
    def pick_random_table_blocker():
        return np.random.choice(list(TABLES.keys())[2:])

    filter = lambda point: np.linalg.norm((point + item[0]) - ARM_POS) > MAX_ARM_REACH

    if grid:
        rand_rotation = False

    positions = {}
    samplers= {}
    for name, item in TABLES.items():
        if grid:
            sampler = GridSampler(
                np.clip(item[1].extent - buffer_radius, 0, np.inf),
                np.max(BLOCK_DIMS) + buffer_radius,
                centered=True,
            )
        else:
            sampler = PoissonSampler(
                np.clip(item[1].extent - buffer_radius, 0, np.inf),
                item[1].r + buffer_radius,
                centered=True,
                clump = clump
            )
        samplers[name] = sampler
        points = sampler.make_samples(num=(num_blocks + num_blockers) * 10, filter = filter)
        if not prioritize_grouping:
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
    stacking = make_random_stacking(blocks, max_stack_num=max_start_stack)
    max_start_stack = max([len(s) for s in stacking])


    for stack in stacking:
        table = pick_random_table_block()
        if len(positions[table]) == 0:
            res = samplers[table].make_samples(filter = filter)
            if len(res) == 0:
                continue
            positions[table] += res
        point = positions[table].pop(-1) + TABLES[table][0]
        point = np.append(point, TABLE_HEIGHT)
        point = np.concatenate((point, np.zeros(3)))
        if rand_rotation:
            yaw = np.random.uniform(0, 2 * np.pi)
        else:
            yaw = 0
        point[-1] = yaw
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
        for block in stack[1:]:
            point[2] += BLOCK_DIMS[2] + 1e-3
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

    stacking = make_random_stacking(blocks, max_stack_num=max_goal_stack)
    max_goal_stack = max([len(s) for s in stacking])
    goal = ["and"]
    for stack in stacking:
        table = pick_random_table_block()
        base_block = stack[0]
        goal.append(["on-table", base_block, [str(table), "base_link"]])
        prev_block = base_block
        for block in stack[1:]:
            goal.append(["on-block", block, prev_block])
            prev_block = block

    yaml_data["goal"] = goal

    for blocker in blockers:
        table = pick_random_table_blocker()
        if len(positions[table]) == 0:
            res = samplers[table].make_samples(filter = filter)
            if len(res) == 0:
                continue
            positions[table] += res
        point = positions[table].pop(-1) + TABLES[table][0]
        point = np.append(point, TABLE_HEIGHT)
        point = np.concatenate((point, np.zeros(3)))
        if rand_rotation:
            point[-1] = np.random.uniform(0, 2 * np.pi)
        yaml_data["objects"][blocker] = {
            "path": "models/blocks_world/sdf/blocker_block.sdf",
            "X_WO": point.tolist(),
            "main_link": "base_link",
            "on-table": [str(table), "base_link"],
        }

    yaml_data["run_attr"] = {
        "num_blocks": num_blocks,
        "num_blockers": num_blockers,
        "max_start_stack": max_start_stack,
        "max_goal_stack": max_goal_stack,
        "buffer_radius": buffer_radius,
        "type": type
    }

    return yaml_data

def make_non_monotonic_problem_v2(num_blocks, clump = False, buffer_radius = 0, prioritize_grouping = False, colorize = False, max_goal_stack = 1):
    
    num_blockers = num_blocks
    filter = lambda point: np.linalg.norm((point + item[0]) - ARM_POS) > MAX_ARM_REACH

    positions = {}
    samplers= {}
    for name, item in TABLES.items():
        sampler = PoissonSampler(
            np.clip(item[1].extent - BLOCK_DIMS.max()/2, 0, np.inf),
            2*item[1].r + buffer_radius,
            centered=True,
            clump = clump
        )
        samplers[name] = sampler
        points = sampler.make_samples(num=num_blocks * 10, filter = filter)
        if not prioritize_grouping:
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
    blocker_points = {}

    start_tables = ["blue_table", "purple_table","red_table","green_table"]
    goal = ["and"]
    for i, (block, blocker) in enumerate(zip(blocks, blockers)):
        table = np.random.choice(start_tables) 
        if len(positions[table]) == 0:
            res = samplers[table].make_samples(filter = filter)
            if len(res) == 0:
                continue
            positions[table] += res
        point = positions[table].pop(-1) + TABLES[table][0]
        point = np.append(point, TABLE_HEIGHT)
        point = np.concatenate((point, np.zeros(3)))
        path = TEMPLATE_PATH
        if colorize:
            color = np.random.uniform(np.zeros(3), np.ones(3))
            color = f"{color[0]} {color[1]} {color[2]} 1"
            path = MODELS_PATH + block + ".sdf"
            make_block(block_name=block, color=color, size=BLOCK_DIMS, buffer=0.001)

        block_point = point.copy()
        blocker_point = point.copy()

        base_angle = np.random.choice([0, 90, 180, 270])
        random_angle = 60 * (np.random.random() - 0.5)
        angle = np.deg2rad(random_angle + base_angle)

        min_distance = BLOCK_DIMS.max()/np.cos(np.deg2rad(random_angle))
        max_distance = (BLOCK_DIMS.max()+0.01)/np.cos(np.deg2rad(random_angle))
        distance = np.random.random()*(max_distance - min_distance) + min_distance
        displacement = np.array([distance*np.cos(angle), distance*np.sin(angle)])
        # print('angle', random_angle, 'distance', distance)

        # print(max(np.abs(displacement[0]) - BLOCK_DIMS.max(), np.abs(displacement[1]) - BLOCK_DIMS.max()))

        blocker_point[0] += displacement[0]/2
        blocker_point[1] += displacement[1]/2
        block_point[0] -= displacement[0]/2
        block_point[1] -= displacement[1]/2

        block_point = block_point.tolist()
        blocker_point = blocker_point.tolist()
        blocker_points[i] = blocker_point

        yaml_data["objects"][block] = {
            "path": path,
            "X_WO": block_point,
            "main_link": "base_link",
            "on-table": [str(table), "base_link"],
        }
        yaml_data["objects"][blocker] = {
            "path": "models/blocks_world/sdf/blocker_block.sdf",
            "X_WO": blocker_point,
            "main_link": "base_link",
            "on-table": [str(table), "base_link"],
        }
        goal.append(["on-table", blocker, [str(table), "base_link"]])

    end_tables = ["blue_table", "purple_table","red_table","green_table"]

    

    stacking = make_random_stacking(blocks, max_stack_num=max_goal_stack)
    for stack in stacking:
        table = np.random.choice(end_tables)
        base_block = stack[0]
        goal.append(["on-table", base_block, [str(table), "base_link"]])
        prev_block = base_block
        for block in stack[1:]:
            goal.append(["on-block", block, prev_block])
            prev_block = block

    yaml_data["goal"] = goal

    yaml_data["run_attr"] = {
        "num_blocks": num_blocks,
        "num_blockers": num_blockers,
        "max_start_stack": 1,
        "max_goal_stack": max_goal_stack,
        "buffer_radius": buffer_radius,
        "type": "non_monotonic"
    }

    return yaml_data


if __name__ == "__main__":

    yaml_data = make_non_monotonic_problem_v2(10, buffer_radius=0, colorize=True)
    with open("test_problem_3.yaml", "w") as stream:
        yaml.dump(yaml_data, stream, default_flow_style=False)