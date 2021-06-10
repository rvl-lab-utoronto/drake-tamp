"""
A module with utilities allowing the PandaStation to interface
with PDDL and the streams (generators)

This module contains convenience classes ObjectInfo
BodyInfo and ShapeInfo. Objects are made of bodies which are made of shapes
"""
import yaml
import numpy as np
from pydrake.all import (
    Cylinder,
    Sphere,
    Box,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
)
from .panda_station import PandaStation, find_resource


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
            try:
                info = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        assert "directive" in info, "Problem specification missing directive"
        assert "objects" in info, "Problem specification missing objects"
        self.directive = info["directive"]
        self.objects = info["objects"]
        for name in self.objects:
            self.objects[name]["X_WO"] = ProblemInfo.list_to_transform(
                self.objects[name]["X_WO"]
            )
        # TODO(agro): generalize this
        self.names_and_links = [
            (name, "base_link") for name in parse_tables(find_resource(self.directive))
        ]

    def make_station(self, weld_to_world, weld_to_hand=None):
        """
        Makes a PandaStation based on this problem instance.
        see `make_all_stations`

        Args:
            weld_to_world: the names of objects to weld to the world
            weld_to_hand: the name of the object to weld to hand

        Returns:
            the newly created PandaStation
        """
        station = PandaStation()
        station.setup_from_file(self.directive, names_and_links=self.names_and_links)
        station.add_panda_with_hand()
        plant = station.get_multibody_plant()

        for name in self.objects:
            if (weld_to_hand is not None) and (name == weld_to_hand):
                # it has already been added
                continue
            welded = False
            P = plant.world_frame()
            X = self.objects[name]["X_WO"]
            if name in weld_to_world:
                welded = True
            if name == weld_to_hand:
                P = plant.GetFrameByName("panda_hand", station.get_hand())
                X = RigidTransform()
            station.add_model_from_file(
                find_resource(self.objects[name]["path"]),
                X,
                main_body_name=self.objects[weld_to_hand]["main_link"],
                welded=welded,
                P=P,
                name=name,
            )

        station.finalize()
        return station

    def make_all_stations(self):
        """
        Makes all PandaStations needed for TAMP. This includes:
        1. a station with not objects welded
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
        res["main"] = self.make_station([])
        res["move_free"] = self.make_station(list(self.objects.keys()))
        for name in self.objects:
            weld_to_world = list(self.objects.keys())
            weld_to_world.remove(name)
            res[name] = self.make_station(weld_to_world, weld_to_hand = name)
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


class ObjectInfo:
    """
    Class for storing all bodies associated with an object
    """

    def __init__(self, name, welded_to_frame=None, path=None):
        """
        Construct an ObjectInfo object

        Args:
            main_body_info: the BodyInfo of the main body of this object
            welded_to_frame: the FixedOffsetFrame that the main body is welded
            to (optional)
            path: the absolute filepath used to find the model (optional)
        """
        self.path = path
        self.main_body_info = None
        self.welded_to_frame = welded_to_frame
        self.body_infos = {}
        self.name = name
        # self.body_infos[main_body_info.get_index()] = self.main_body_info

    def set_main_body_info(self, main_body_info):
        """
        Set the main body info of this ObjectInfo
        """
        self.main_body_info = main_body_info
        if not main_body_info.get_index() in self.body_infos.keys():
            self.body_infos[main_body_info.get_index()] = self.main_body_info

    def add_body_info(self, body_info):
        """
        Given a BodyInfo, add it to this object
        """
        self.body_infos[body_info.get_index()] = body_info

    def get_main_body(self):
        """
        Return the BodyInfo of the main body of this object
        """
        return self.main_body_info

    def get_frame(self):
        """
        Get the welded_to_frame
        """
        return self.welded_to_frame

    def get_path(self):
        """
        returns the path used to create this object
        """
        return self.path

    def get_name(self):
        """
        return the name of this ObjectInfo
        """
        return self.name

    def get_body_infos(self):
        """
        Return the body infos associated with this object
        """
        return self.body_infos

    def str_info(self):
        """
        get string info about this ObjectInfo
        """
        frame_name = None
        if self.welded_to_frame is not None:
            frame_name = self.welded_to_frame.name()
        res = f"""
        name: {self.name}
        path: {self.path}
        main body name: {self.main_body_info.get_name()}
        welded to frame: {frame_name}
        BodyInfos: 
        """
        for info in self.body_infos.values():
            str_info = str(info).split("\n")
            for string in str_info:
                res += "\t" + string + "\n"

        return res

    def __str__(self):
        res = f"Object name: {self.name}"
        return res


class BodyInfo:
    """
    Class for storing all geometries associated with a body
    """

    def __init__(self, body, body_index):
        """
        Construct a body info instance by providing its
        pydrake.multibody.tree.BodyIndex

        Args:
            body_index: the BodyIndex of the main body
            welded to (if it is welded to one)
            body: the actual Body associated with this body
            index
        """
        self.body_index = body_index
        self.body = body
        self.shape_infos = []

    def add_shape_info(self, shape_info):
        """
        Add a ShapeInfo instance to be associated with this body
        """
        self.shape_infos.append(shape_info)

    def get_name(self):
        """
        Get the name of this body
        """
        return self.body.name()

    def get_index(self):
        """
        Return the index of this body
        """
        return self.body.index()

    def get_body_frame(self):
        """
        Return the body frame of this body
        """
        return self.body.body_frame()

    def get_body(self):
        """
        Returns this body
        """
        return self.body

    def get_shape_infos(self):
        """
        Return the shape infos associated with this body
        """
        return self.shape_infos

    def __str__(self):
        res = f"""body index: {self.body_index}, body_name: {self.body.name()}
        ShapeInfos: 
        """

        for info in self.shape_infos:
            str_info = str(info).split("\n")
            for string in str_info:
                res += "\t" + string + "\n\t"

        return res


class ShapeInfo:
    """
    Class for storing the information about a shape
    """

    def __init__(self, shape, offset_frame):

        """
        Construct a ShapeInfo instance given a
        pydrake.geometry.GeometryInstance.shape and its
        associated pydrake.multibody.tree.FixedOffsetFrame
        """
        self.shape = shape
        self.offset_frame = offset_frame
        self.type = type(shape)

    def __str__(self):
        """
        Used when printing out the shape info
        """
        res = f"offset_frame: {self.offset_frame.name()}, type:"
        if self.type == Box:
            res += "box"
        if self.type == Cylinder:
            res += "cylinder"
        if self.type == Sphere:
            res += "sphere"
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
    for object_info, _ in station.object_infos.values():
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
    panda = station.get_panda()
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    q = plant.GetPositions(plant_context, panda)
    arms = {}
    arms["panda"] = q
    return arms


def parse_tables(directive):
    """
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


def update_station(station, station_context, pose_fluents):
    """
    Update the poses of the welded objects in the
    PandaStation based on the poses in pose_fluents.

    Args:
        station: the PandaStation will all manipulands welded in place
        (output from `create_welded_station`)

        station_context: the context for the panda station

        pose_fluents: A list of tuples of the form
        [('atpose', object_info_name, X_WO), ..., ('atgraspose', object_info_name, X_WH)]
    Returns:
        None, but updates the welded station provided in welded_station
    """
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    for _, name, X_PO in pose_fluents:
        object_info = station.object_infos[name][0]
        offset_frame = object_info.get_frame()
        assert (
            offset_frame is not None
        ), "you are trying to set the pose of a free object"
        offset_frame.SetPoseInBodyFrame(plant_context, X_PO)
