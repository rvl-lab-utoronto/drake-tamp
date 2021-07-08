import os
from panda_station.models.blocks_world.sdf.make_blocks import TEMPLATE_NAME
from learning.poisson_disc_sampling import PoissonSampler
import numpy as np
import xml.etree.ElementTree as ET
from pydrake.all import RigidTransform, RollPitchYaw

DIRECTIVE = os.path.expanduser("~/drake-tamp/panda_station/directives/kitchen.yaml")
PLANNING_DIRECTIVE = os.path.expanduser(
    "~/drake-tamp/panda_station/directives/kitchen_planning.yaml"
)
CABBAGE_DIMS = np.array([0.045, 0.045, 0.07])
RADDISH_DIMS = np.array([0.045, 0.045, 0.1])
GLASS_DIMS = np.array([0.06, 0.06, 0.07])  # a conservative bounding around the cylinder
R = 0.03 * 2 + 0.01
# name: {X_WR, Sampler, num_objects}


REGIONS = {
    "tray": {
        "name": "tray",
        "X_WR": RigidTransform(
            RollPitchYaw(0, 0, np.pi / 2), np.array([0.5, 0.1, 0.325])
        ),
        "sampler": PoissonSampler(np.array([0.4, 0.3]) - R, r=R, centered=True),
        "max_goal": np.inf,
        "main_link": "base_link",
    },
    "sink": {
        "name": "sink",
        "X_WR": RigidTransform(
            RollPitchYaw(0, 0, np.pi / 4), np.array([0.45, -0.4, 0.325])
        ),
        "sampler": PoissonSampler(np.array([0.34, 0.24]) - 2*R, r=R, centered=True),
        "max_goal": np.inf,
        "main_link": "base_link",
    },
    "rightplate": {
        "name": "rightplate",
        "X_WR": RigidTransform(RollPitchYaw(0, 0, 0), np.array([-0.15, 0.45, 0.34])),
        "valid_positions": [np.array([0, 0, 0.05])],
        "max_goal": 1,
        "main_link": "base_link",
    },
    "leftplate": {
        "name": "leftplate",
        "X_WR": RigidTransform(RollPitchYaw(0, 0, 0), np.array([-0.45, 0.15, 0.34])),
        "valid_positions": [np.array([0, 0, 0.05])],
        "max_goal": 1,
        "main_link": "base_link",
    },
    "leftplacemat_left": {
        "name": "leftplacemat",
        "X_WR": RigidTransform(
            RollPitchYaw(0, 0, np.pi / 4), np.array([-0.45, 0.15, 0.325])
        ),
        "valid_positions": [np.array([-0.15, 0.1, 0.05]), np.array([-0.15, -0.1, 0.05])],
        "max_goal": 1,
        "main_link": "leftside",
    },
    "leftplacemat_right": {
        "name": "leftplacemat",
        "X_WR": RigidTransform(
            RollPitchYaw(0, 0, np.pi / 4), np.array([-0.45, 0.15, 0.325])
        ),
        "valid_positions": [np.array([0.15, 0.1, 0.05]), np.array([0.15, -0.1, 0.05])],
        "max_goal": 1,
        "main_link": "rightside",
    },
    "rightplacemat_left": {
        "name": "rightplacemat",
        "X_WR": RigidTransform(
            RollPitchYaw(0, 0, np.pi / 4), np.array([-0.15, 0.45, 0.325])
        ),
        "valid_positions": [np.array([-0.15, 0.1, 0.05]), np.array([-0.15, -0.1, 0.05])],
        "max_goal": 2,
        "main_link": "leftside",
    },
    "rightplacemat_right": {
        "name": "rightplacemat",
        "X_WR": RigidTransform(
            RollPitchYaw(0, 0, np.pi / 4), np.array([-0.15, 0.45, 0.325])
        ),
        "valid_positions": [np.array([0.15, 0.1, 0.05]), np.array([0.15, -0.1, 0.05])],
        "max_goal": 2,
        "main_link": "rightside",
    },
}
MODELS_PATH = "models/kitchen/sdf/"


class RegionInfo:
    def __init__(
        self, name, main_link, X_WR, max_goal, sampler=None, valid_positions=None
    ):
        self.num_supporing = 0
        self.name = name
        self.main_link = main_link
        self.X_WR = X_WR
        assert (
            sampler is not None or valid_positions is not None
        ), "This region must have either a sampler or a list of valid positions"
        assert not (
            sampler is not None and valid_positions is not None
        ), "This region cannot have both a sampler and a list of valid positions"
        self.sampler = sampler
        self.valid_positions = valid_positions
        self.max_goal = max_goal
        self.num_goal = 0

    def has_position(self):
        if self.sampler:
            return True
        return len(self.valid_positions) > 0

    def get_X_WO(self):
        if self.sampler:
            p_RO_O = self.sampler.sample()
            if p_RO_O is None:
                return None
            p_RO_O = np.append(p_RO_O, 0.05)
        else:
            if len(self.valid_positions) == 0:
                return None
            rand_ind = np.random.randint(len(self.valid_positions))
            p_RO_O = self.valid_positions.pop(rand_ind)

        R_WR = self.X_WR.rotation()
        p_RO_W = R_WR.multiply(p_RO_O)

        return RigidTransform(
            RollPitchYaw(0, 0, np.random.uniform(0, 2 * np.pi)),
            self.X_WR.translation() + p_RO_W,
        )


def X_to_np(X):
    assert isinstance(X, RigidTransform), "Must supply a rigid transform"
    rpy = RollPitchYaw(X.rotation()).vector()
    xyz = X.translation()
    return np.concatenate((xyz, rpy))


class KitchenProblemMaker:
    def __init__(self, num_cabbages, num_raddishes, num_glasses):
        self.num_cabbages = num_cabbages
        self.num_glasses = num_glasses
        self.num_raddishes = num_raddishes
        self.num_objects = num_cabbages + num_raddishes + num_glasses
        self.regions = {}
        self.X_WOs = {}
        for region_name in REGIONS:
            info = REGIONS[region_name]
            if "sampler" in info:
                self.regions[region_name] = RegionInfo(
                    info["name"],
                    info["main_link"],
                    info["X_WR"],
                    info["max_goal"],
                    sampler=PoissonSampler(
                        info["sampler"].extent,
                        r=info["sampler"].r,
                        centered= True
                    ),
                )
                self.X_WOs[region_name] = []
                for i in range(self.num_objects):
                    X_WO = self.regions[region_name].get_X_WO()
                    if X_WO is None:
                        break
                    self.X_WOs[region_name].append(X_WO)
            else:
                self.regions[region_name] = RegionInfo(
                    info["name"],
                    info["main_link"],
                    info["X_WR"],
                    info["max_goal"],
                    valid_positions=info["valid_positions"].copy(),
                )

                self.X_WOs[region_name] = []
                while True:
                    X_WO = self.regions[region_name].get_X_WO()
                    if X_WO is None:
                        break
                    self.X_WOs[region_name].append(X_WO)

    @staticmethod
    def put_at_start(region_list, name):
        s = region_list[0]
        ind = region_list.index(name)
        region_list[0] = name
        region_list[ind] = s

    def get_random_region_and_point(self):

        region_list = list(self.X_WOs.keys())

        np.random.shuffle(region_list)
        tp = np.random.uniform(0,10)

        if tp < 4:
            self.put_at_start(region_list, "tray")
        if 4 < tp < 6:
            self.put_at_start(region_list, "sink")

        for region in region_list:
            region_name = self.regions[region].name
            main_link =  self.regions[region].main_link
            if len(self.X_WOs[region]) == 0:
                X_WO = self.regions[region].get_X_WO()
                if X_WO is None:
                    continue
                self.X_WOs[region].append(X_WO)
            return [str(region_name), str(main_link)], self.X_WOs[region].pop(-1)
        return None

    def get_random_region(self):
        region = np.random.choice(list(self.X_WOs.keys()))
        region_name = self.regions[region].name
        main_link =  self.regions[region].main_link
        return [str(region_name), str(main_link)]

    def get_random_goal_region(self):
        region_list = list(self.X_WOs.keys())
        np.random.shuffle(region_list)
        for region in region_list:
            if self.regions[region].num_goal >= self.regions[region].max_goal:
                continue
            self.regions[region].num_goal += 1
            region_name = self.regions[region].name
            main_link = self.regions[region].main_link
            return [str(region_name), str(main_link)]
        return None

    def make_problem(self):
        yaml_data = {
            "directive": "directives/kitchen.yaml",
            "planning_directive": "directives/kitchen_planning.yaml",
            "arms": {
                "panda": {
                    "panda_name": "panda",
                    "hand_name": "hand",
                    "X_WB": [0, 0, 0, 0, 0, 0],
                }
            },
            "objects": {},
            "main_links": {
                "table_long": "base_link",
                "table_square": "base_link",
                "table_serving": "base_link",
                "stove": "base_link",
                "sink": "base_link",
                "leftplacemat": "base_link",
                "rightplacemat": "base_link",
                "leftplate": "base_link",
                "rightplate": "base_link",
                "tray": "base_link",
            },
            "surfaces": {
                "stove": [
                    "infopad",
                    "burner1",
                    "burner2",
                    "burner3",
                    "burner4",
                    "burner5",
                ],
                "sink": ["base_link"],
                "leftplacemat": ["leftside", "rightside"],
                "leftplate": ["base_link"],
                "rightplacemat": ["leftside", "rightside"],
                "rightplate": ["base_link"],
                "tray": ["base_link"],
            },
        }

        goal = ["and"]
        cabbages = [f"cabbage{i}" for i in range(self.num_cabbages)]
        raddishes = [f"raddish{i}" for i in range(self.num_raddishes)]
        glasses = [f"glass{i}" for i in range(self.num_glasses)]

        for cabbage in cabbages:
            region, X_WO = self.get_random_region_and_point()
            if X_WO is None:
                print(f"failed to add {cabbage}")
                continue
            # add a goal for this cabbage
            yaml_data["objects"][cabbage] = {
                "path": MODELS_PATH + "cabbage.sdf",
                "X_WO": X_to_np(X_WO).tolist(),
                "main_link": "base_link",
                "contained": region,
            }
            goal_region = self.get_random_goal_region()
            if goal_region is None:
                continue
            goal.append(["in", cabbage, goal_region])
            if np.random.randint(0, 2):
                goal.append(["cooked", cabbage])

        for raddish in raddishes:
            region, X_WO = self.get_random_region_and_point()
            if X_WO is None:
                print(f"failed to add {raddish}")
                continue
            # add a goal for this cabbage
            yaml_data["objects"][raddish] = {
                "path": MODELS_PATH + "raddish.sdf",
                "X_WO": X_to_np(X_WO).tolist(),
                "main_link": "base_link",
                "contained": region,
            }
            # rarely: make a goal that cooks the raddish
            if np.random.randint(0, 10) == 1:
                goal.append(["cooked", raddish])
                goal_region = self.get_random_goal_region()
                if goal_region is None:
                    continue
                goal.append(["in", raddish, goal_region])

        for glass in glasses:
            region, X_WO = self.get_random_region_and_point()
            if X_WO is None:
                print(f"failed to add {glass}")
                continue
            yaml_data["objects"][glass] = {
                "path": MODELS_PATH + "glass.sdf",
                "X_WO": X_to_np(X_WO).tolist(),
                "main_link": "base_link",
                "contained": region,
            }
            goal_region = self.get_random_goal_region()
            if goal_region is None:
                continue
            goal.append(["in", glass, goal_region])
            if np.random.randint(0, 2):
                goal.append(["clean", glass])

        yaml_data["goal"] = goal
        return yaml_data


def make_random_problem(num_cabbages, num_raddishes, num_glasses):

    maker = KitchenProblemMaker(num_cabbages, num_raddishes, num_glasses)
    return maker.make_problem()

def pick_random_region():
    regions = list(REGIONS.keys())
    np.random.shuffle(regions)
    for region in regions:
        region_info = REGIONS[region]
        if "sampler" in region.keys():
            return region


if __name__ == "__main__":

    make_random_problem(1, 1, 1)