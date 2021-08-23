"""
A module with utilities allowing the PandaStation to interface
with PDDL and the streams (generators)

This module contains convenience classes ObjectInfo
BodyInfo and ShapeInfo. Objects are made of bodies which are made of shapes
"""
import yaml
import numpy as np
from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
)
from .panda_station import PandaStation
from .construction_utils import find_resource
from .utils import *
from .grasping_and_placing import (
    backup_on_hand_z, 
    backup_on_world_z,
    is_placeable, 
    is_graspable, 
    is_safe_to_place
)


class ProblemInfo:
    """
    Class for storing information about the intial setup of a problem
    """

    def __init__(self, problem_file):
        """
        Construct a ProblemInfo object

        Args:
            problem_file: the absolute path to the .yaml file with the
            information to setup the problem

            This yaml file will have the format:

            directive: "directive/directive_name"

            objects:
                object_name1:
                    path: "model_path"
                    X_WO: [x, y, z, roll, pitch, yaw]
                    main_link: "main_link_name"
                object_name2:
                    path: "model_path"
                    X_WO: [x, y, z, roll, pitch, yaw]
                    main_link: "main_link_name"
                ...
        """
        info = None
        with open(problem_file, "r") as stream:
            info = yaml.safe_load(stream)
        assert "directive" in info, "Problem specification missing directive"
        assert "objects" in info, "Problem specification missing objects"
        assert "surfaces" in info, "Problem specification missing surfaces"
        assert "main_links" in info, "Problem specification missing main_links"
        assert "arms" in info, "Problem specification missing arms"
        self.attr = info.get("run_attr", None)
        self.arms = info["arms"]
        self.directive = info["directive"]
        self.planning_directive = info["directive"]
        if "planning_directive" in info:
            self.planning_directive = info["planning_directive"]
        self.objects = info["objects"]
        self.main_links = info["main_links"]
        self.surfaces = info["surfaces"]
        for name in self.objects:
            transform = ProblemInfo.list_to_transform(
                self.objects[name]["X_WO"]
            )
            self.objects[name]["X_WO"] = transform 

        if "goal" in info:
            self.goal = info["goal"]
            for i, item in enumerate(info["goal"]):
                if isinstance(item, list):
                    for j, val in enumerate(item):
                        if isinstance(val, list):
                            if len(val) == 2:
                                self.goal[i][j] = tuple(val)
                    self.goal[i] = tuple(item)
        self.names_and_links = [
            (name, main_link) for name, main_link in self.main_links.items()
        ]

    def make_station(
        self,
        weld_to_world,
        weld_to_hand=None,
        weld_fingers=False,
        arm_name = None,
        name="panda_station",
        X_PO = None,
        planning = False,
        dummy = False,
        time_step = 1e-3,
        blocked = False,
    ):
        """
        Makes a PandaStation based on this problem instance.
        see `make_all_stations`

        Args:
            weld_to_world: the names of objects to weld to the world
            weld_to_hand: the name of the object to weld to hand
            X_PO: override the rigid transform of this object wrt it's parent
        Returns:
            the newly created PandaStation
        """
        station = PandaStation(name=name, dummy = dummy, time_step = time_step)
        directive = self.directive
        if planning:
            directive = self.planning_directive
        station.setup_from_file(directive, names_and_links=self.names_and_links)

        #arms = []
        #if arm_name is None:
            #arms = self.arms.values()
        #else:
            #arms = [self.arms[arm_name]]

        if arm_name is None:
            arm_name = list(self.arms.values())[0]["panda_name"]

        panda_info = None
        for arm_info in self.arms.values():
            info = station.add_panda_with_hand(
                hand_name = arm_info["hand_name"],
                panda_name = arm_info["panda_name"],
                X_WB = ProblemInfo.list_to_transform(arm_info["X_WB"]),
                weld_fingers=weld_fingers,
                blocked = blocked
            )
            if info.panda_name == arm_name:
                panda_info = info

        plant = station.get_multibody_plant()

        for name in self.objects:
            welded = False
            P = None
            X = self.objects[name]["X_WO"]
            if name in weld_to_world:
                welded = True
            if name == weld_to_hand:
                P = plant.GetFrameByName("panda_hand", panda_info.hand)
                X = RigidTransform()
                X.set_translation([0, 0, 0.2])
                welded = True
            if X_PO is not None:
                X = X_PO
            object_info = station.add_model_from_file(
                find_resource(self.objects[name]["path"]),
                X,
                main_body_name=self.objects[name]["main_link"],
                welded=welded,
                P=P,
                name=name,
            )

        station.finalize()
        return station

    def make_main_station(self, time_step = 1e-4):
        """
        Make the main station for TAMP: a station with no objects welded
        """
        print(f"{Colors.BLUE}Building main station{Colors.RESET}")
        return self.make_station([], name="main", time_step = 1e-4)

    def make_move_free_station(self, arm_name = None):
        """
        Make the move_free station for TAMP: a station with all objects welded
        to the world
        """
        print(f"{Colors.BLUE}Building move_free station{Colors.RESET}")
        return self.make_station(
            list(self.objects.keys()),
            weld_fingers = True,
            name="move_free",
            planning = True,
            arm_name = arm_name,
            dummy = True
        )

    def make_blocked_free_station(self, arm_name = None):
        """
        Make the move_free station for TAMP: a station with all objects welded
        to the world
        """
        print(f"{Colors.BLUE}Building move_free station{Colors.RESET}")
        return self.make_station(
            list(self.objects.keys()),
            blocked = True,
            name="move_free",
            planning = True,
            arm_name = arm_name,
            dummy = True
        )

    def make_holding_station(self, name, X_HO = None, arm_name = None):
        """
        Make a station with object named `name` welded to the panda
        hand. X_HO is an optional argument specifying the relative
        transform between the panda hand and the object it is holding
        """
        print(f"{Colors.BLUE}Building {name} station{Colors.RESET}")
        weld_to_world = list(self.objects.keys())
        weld_to_world.remove(name)
        return self.make_station(
            weld_to_world,
            weld_to_hand=name,
            weld_fingers=True,
            arm_name = arm_name,
            name=name,
            X_PO = X_HO,
            planning = True,
            dummy = True
        )

    def make_all_stations(self):
        """
        Makes all PandaStations needed for TAMP. This includes:
        1. a station with no objects welded
        2. a station with all objects welded to the world frame
        3. for each object, one station with that object welded to the
        panda hand and every other object welded to the world

        Returns:
            A dictionary in the form:
            {main: station_type_1,
            move_free: station_type_2,
            object_name: station_type3,
            ...}
        """
        res = {}
        res["main"] = self.make_main_station()
        res["move_free"] = self.make_move_free_station()
        if len(self.arms) == 1:
            for name in self.objects:
                res[name] = self.make_holding_station(name)
        else:
            for arm_name in self.arms:
                res[arm_name] = {}
                res[arm_name]["move_free"] = self.make_move_free_station()
                for name in self.objects:
                    res[arm_name][name] = self.make_holding_station(
                        name,
                        arm_name = arm_name
                    )
        return res

    @staticmethod
    def list_to_transform(xyzrpy):
        """
        Input a list in the form:
        [x, y, z, roll, pitch, yaw]
        and return a pydrake.math.RigidTransform
        """
        return RigidTransform(
            RotationMatrix(RollPitchYaw(np.array(xyzrpy[3:]))), xyzrpy[:3]
        )

    def get_objects(self):
        """
        Get the objects in the problem
        """
        return self.objects

    def __getitem__(self, key):
        return self.objects[key]

    def get_directive(self):
        """
        Get the directive in this problem
        """
        return self.directive

    def __str__(self):
        res = f"directive: {self.directive}\n"
        for name in self.objects:
            res += f"object: {name}\n"
            X_WO = self.objects[name]["X_WO"]
            res += f"X_WO: {X_WO}\n"
        return res


def parse_start_poses(station, station_context):
    """
    Parses the information in PandaStation `station` to obtain the start
    poses of all FREE objects (manipulands)

    Args:
        station: PandaStation instance
        station_context: the station context to parse
    Returns:
        A diction dictionary in the form:
        start_poses = {
            object_name: X_WO1
            ...
        }
        where object_info1 is the ObjectInfo instance for that object,
        and X_WO1 is apydrake.math.RigidTransform representing the
        tranformation between the world frame and the main_body_index
        of that object (see `drake-tamp/panda_station/panda_station.py`)
    """
    start_poses = {}
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    for object_info, X in station.object_infos.values():
        if X is None:
            continue
        X_WO = object_info.get_main_body().get_body().EvalPoseInWorld(plant_context)
        start_poses[object_info.get_name()] = X_WO
    return start_poses


def parse_config(station, station_context):
    """
    Parses the information in PandaStation `station` to obtain the name
    and configuration of the panda arm(s)

    Args:
        station: PandaStation instance
        station_context: the station context to parse
    Returns:
        arms = {
            arm_name: q0
            ...
        }
        arm_name is the name of the panda (string) and qi is the initial
        config of the ith panda arm
    """
    # TODO(agro): current, panda station only supports one arm, make this more
    arms = {}
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)

    for panda_info in station.panda_infos.values():
        panda = panda_info.panda
        q = plant.GetPositions(plant_context, panda)
        arms[panda_info.panda_name] = q

    return arms


def parse_tables(directive):
    """
    DEPRECIATED
    Parser all of the names of tables in the provided PandaStation instance

    Args:
        directive: the full path the the yaml file used to build the
        plant
    Returns:
        A list of the names of the tables in the station
    """
    res = []
    in_model = False
    with open(directive, "r") as file:
        words = file.read().split()
        for i, word in enumerate(words):
            if (in_model) and (word == "name:"):
                if "table" in words[i + 1]:
                    res.append(words[i + 1])
                in_model = False
            if word == "add_model:":
                in_model = True
    return res


def update_arm(station, station_context, panda_name, q):
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    panda = station.panda_infos[panda_name].panda
    plant.SetPositions(plant_context, panda, q)

def update_station(station, station_context, pose_fluents, set_others_to_inf=False):
    """
    Update the poses of the welded objects in the
    PandaStation based on the poses in pose_fluents.

    Args:
        station: the PandaStation will all manipulands welded in place
        (output from `create_welded_station`)

        station_context: the context for the panda station

        pose_fluents: A list of tuples of the form
        [('atpose', object_info_name, X_WO), ..., ('atgraspose', object_info_name, X_WH)]
        X* can be either Drake's RigidTransform or a RigidTransformWrapper

        set_others_to_inf: if True, the poses of any unspecified objects will be set to
        infinity (far away) so they are not considerd in planning
    Returns:
        None, but updates the welded station provided in welded_station
    """
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    set_pose = []
    for _, name, X_PO in pose_fluents:
        if isinstance(X_PO, RigidTransformWrapper):
            X_PO = X_PO.get_rt()
        set_pose.append(name)
        object_info = station.object_infos[name][0]
        offset_frame = object_info.get_frame()
        assert (
            offset_frame is not None
        ), "you are trying to set the pose of a free object"
        offset_frame.SetPoseInBodyFrame(plant_context, X_PO)

    if set_others_to_inf:
        X_WO = RigidTransform(RotationMatrix(), [10, 0, 0])
        for object_info, Xinit_WO in list(station.object_infos.values()):
            if object_info.get_name() in set_pose:  # its pose has been set
                continue
            if Xinit_WO is None:  # it is not a manipuland
                continue
            offset_frame = object_info.get_frame()
            offset_frame.SetPoseInBodyFrame(plant_context, X_WO)
            # spacing of 5 m should be far enough
            X_WO.set_translation(X_WO.translation() + np.array([5, 0, 0]))


def update_graspable_shapes(object_info):
    """
    Updates and returns the internal list graspable_shapes within
    the object_info instance
    """
    if len(object_info.graspable_shapes) > 0:
        return object_info.graspable_shapes
    shapes = object_info.query_shape_infos(is_graspable)
    object_info.graspable_shapes = shapes
    return shapes

def update_placeable_shapes(object_info):
    """
    Updates and returns the internal list graspable_shapes within
    the object_info instance
    """
    if len(object_info.placeable_shapes) > 0:
        return object_info.placeable_shapes
    shapes = object_info.query_shape_infos(is_placeable)
    object_info.placeable_shapes = shapes
    return shapes

def update_surfaces(object_info, link_name, station, station_context):
    """
    Updates and return the internal list of surfaces suitable for
    placement within the object_info instance, considering only 
    link_name as the safe surface for placement
    """
    if link_name in object_info.surfaces:
        return object_info.surfaces[link_name]
    for body_info in object_info.get_body_infos().values():
        if link_name != body_info.get_name():
            continue
        object_info.surfaces[link_name] = []
        for shape_info in body_info.get_shape_infos():
            is_safe, surface = is_safe_to_place(shape_info, station, station_context)
            if not is_safe:
                continue
            object_info.surfaces[link_name].append(surface)

    return object_info.surfaces[link_name]

def random_q(station):
    """
    Given a panda station, find a random joint configuration
    """
    lower = station.get_panda_lower_limits()
    upper = station.get_panda_upper_limits()
    return np.random.uniform(lower, upper)

def random_normal_q(station, q_nominal):
    """
    Given a panda station, find a random joint configuration
    as a gaussian centered around q_nominal
    """
    lower = station.get_panda_lower_limits()
    upper = station.get_panda_upper_limits()
    stddev = np.maximum(q_nominal - lower, upper - q_nominal)/2
    rand_q = np.random.normal(
        q_nominal, 
        scale = stddev
    )
    return np.clip(rand_q, lower, upper)

# TODO(agro): this currently only supports one panda
def pre_and_post_grasps(station, station_context, grasp_q, dist = 0.07, panda_info = None):
    """
    Return the pre and post grasp joint configurations for the 
    PandaStation station given the grasping configuration 
    grasp_q. `dist` is the optimal distance between the pre/post grasp
    end effector poses and the grasp end effector pose
    """
    pregrasp_q, postgrasp_q = grasp_q.copy(), grasp_q.copy()
    set_pregrasp = False
    set_postgrasp = False
    while (dist > 0) and not (set_pregrasp and set_postgrasp):
        q, cost = backup_on_hand_z(
            grasp_q, 
            station, 
            station_context, 
            d = dist,
            panda_info = panda_info
        )
        if np.isfinite(cost) and not set_pregrasp:
            pregrasp_q = q
            set_pregrasp = True
        q, cost = backup_on_world_z(
            grasp_q, 
            station, 
            station_context, 
            d = dist,
            panda_info = panda_info
        )
        if np.isfinite(cost) and not set_postgrasp:
            postgrasp_q = q
            set_postgrasp = True
        dist -= 0.01
    return pregrasp_q, postgrasp_q

def find_pregrasp(station, station_context, grasp_q, dist = 0.07, panda_info = None):
    """
    Return the pre grasp joint configurations for the 
    PandaStation station given the grasping configuration 
    grasp_q. `dist` is the optimal distance between the pre/post grasp
    end effector poses and the grasp end effector pose
    """
    while (dist > 0):
        q, cost = backup_on_hand_z(
            grasp_q, 
            station, 
            station_context, 
            d = dist,
            panda_info = panda_info
        )
        if np.isfinite(cost):
            return q
        dist -= 0.01
    return q