"""
A module with utilities allowing the PandaStation to interface
with PDDL and the streams (generators)
"""
from pydrake.all import (
    Cylinder,
    Sphere,
    Box,
    Parser,
    FixedOffsetFrame,
    RigidTransform,
    ProcessModelDirectives,
    LoadModelDirectives,
)
from . import panda_station
from .construction_utils import add_package_paths


class BodyInfo:
    """
    Class for storing all geometries associated with a body
    """

    def __init__(self, body_index, offset_frame=None):
        """
        Construct a body info instance by providing its
        pydrake.multibody.tree.BodyIndex

        Args:
            body_index: the BodyIndex of the main body
            offest_frame: the FixedOffsetFrame this object is
            welded to (if it is welded to one)
        """
        self.body_index = body_index
        self.offset_frame = offset_frame
        self.shape_infos = []

    def add_shape_info(self, shape_info):
        """
        Add a ShapeInfo instance to be associated with this body
        """
        self.shape_infos.append(shape_info)

    def __str__(self):
        res = f"""
        main body index: {self.body_index}
        offset_frame: {self.offset_frame} 
        """

        for info in self.shape_infos:
            res += str(info)
            res += "\n"
        
        return res


class ShapeInfo:
    """
    Class for storing the information about a shape
    """

    def __init__(self, shape, frame):
        """
        Construct a ShapeInfo instance given a
        pydrake.geometry.GeometryInstance.shape and its
        associated pydrake.multibody.tree.FixedOffsetFrame
        """
        self.shape = shape
        self.frame = frame
        self.type = type(shape)

    def __str__(self):
        """
        Used when printing out the shape info
        """
        res = None
        if self.type == Box:
            res = "box"
        if self.type == Cylinder:
            res = "cylinder"
        if self.type == Sphere:
            res = "sphere"
        return res + " " + str(self.frame)


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
            item1_name: X_WO1
            ...
        }
        where item1_name is the string name of the item, and X_WO1 is a
        pydrake.math.RigidTransform representing the tranformation between
        the world frame and the main_body_index of that object (see
        `drake-tamp/panda_station/panda_station.py`)
    """
    start_poses = {}
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    for object_info in station.object_infos.values():
        body = plant.get_body(object_info.main_body_index)
        X_WO = body.EvalPoseInWorld(plant_context)
        start_poses[object_info.name] = X_WO
    return start_poses


def parse_initial_config(station, station_context):
    """
    Parses the information in PandaStation `station` to obtain the name
    and start configuration of the panda arm(s)

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


def parse_tables(station):
    """
    Parser all of the names of tables in the provided PandaStation instance

    Args:
        station: the PandaStation
        station_context: the context of the PandaStation
    Returns:
        A list of the names of the tables in the station
    """
    directive = station.get_directive()
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


def create_welded_station(station, station_context, holding_index=None):
    """
    Create new PandaStation with all free floating objects in
    PandaStation station welded to fixedoffset frames with the
    same pose in world as in the provided context station_context.
    Also adds fixed offset frames for every geometry for each one of
    those objects.

    Args:
        station: PandaStation instance
        station_context: the station context to parse
        holding_index: supply the BodyIndex of the object that should
        be welded to the panda hand for motion trajectory or placement
        planning
    Returns:
        The welded PandaStation as described above with a dictionary
        of the BodyInfo objects for each object (if they are suitable
        for manipulation or a valid placement surface) in the world of
        the form {"object_name": BodyInfo(), ...}
    """
    # get names of objects (tables) suitable for object placement
    placement_model_names = parse_tables(station)
    directive = station.get_directive()
    plant, scene_graph = station.get_plant_and_scene_graph()
    plant_context = station.GetSubsystemContext(plant, station_context)
    # get inspector for geometry queries
    scene_graph_context = station.GetSubsystemContext(scene_graph, station_context)
    query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
    inspector = query_object.inspector()
    # the new welded station
    welded_station = panda_station.PandaStation()
    welded_plant = welded_station.get_multibody_plant()
    parser = Parser(welded_plant)
    add_package_paths(parser)
    ProcessModelDirectives(LoadModelDirectives(directive), welded_plant, parser)
    # setup hand and arm
    welded_station.add_panda_with_hand(weld_fingers=True)
    welded_hand = welded_station.get_hand()

    welded_body_infos = {}
    for obj in station.object_infos.values():
        base_body = plant.get_body(obj.main_body_index)
        X_WB = base_body.EvalPoseInWorld(plant_context)
        welded_model = parser.AddModelFromFile(obj.path, obj.name)
        offset_frame = None
        if (holding_index is not None) and (obj.main_body_index == holding_index):
            base_body_frame = base_body.body_frame()
            X_HB = base_body_frame.CalcPose(
                plant_context, plant.GetFrameByName("panda_hand")
            )
            welded_plant.WeldFrames(
                welded_plant.GetFrameByName("panda_hand", welded_hand),
                welded_plant.GetFrameByName(base_body.name(), welded_model),
                X_HB,
            )
        else:
            # add a FixedOffsetFrame first so we can modify it later if need be
            offset_frame = FixedOffsetFrame(
                "frame_" + obj.name, welded_plant.world_frame(), X_WB
            )
            welded_plant.AddFrame(offset_frame)
            welded_plant.WeldFrames(
                offset_frame,
                welded_plant.GetFrameByName(base_body.name(), welded_model),
                RigidTransform(),
            )
        welded_indices = welded_plant.GetBodyIndices(welded_model)
        for i in welded_indices:
            welded_body = welded_plant.get_body(i)
            if welded_body.name() == base_body.name():
                welded_body_infos[obj.name] = BodyInfo(i, offset_frame=offset_frame)

        indices = plant.GetBodyIndices(plant.GetModelInstanceByName(obj.name))
        for i in indices:
            body = plant.get_body(i)
            welded_body = welded_plant.GetBodyByName(body.name(), welded_model)
            for j, geom_id in enumerate(plant.GetCollisionGeometriesForBody(body)):
                shape = inspector.GetShape(geom_id)
                X_BG = inspector.GetPoseInFrame(geom_id)
                frame_name = "frame_" + obj.name + "_" + body.name() + "_" + str(j)
                frame = welded_plant.AddFrame(
                    FixedOffsetFrame(frame_name, welded_body.body_frame(), X_BG)
                )
                welded_body_infos[obj.name].add_shape_info(ShapeInfo(shape, frame))

    for model_name in placement_model_names:
        model = plant.GetModelInstanceByName(model_name)
        indices = plant.GetBodyIndices(model)
        # TODO(agro): generalize this to table with more than one body
        assert (
            len(indices) == 1
        ), "Table has more than one body, this has yet to be implemented"
        body = plant.get_body(indices[0])
        X_WB = body.EvalPoseInWorld(plant_context)
        welded_model = plant.GetModelInstanceByName(model_name)
        indices = welded_plant.GetBodyIndices(welded_model)
        welded_body_info = BodyInfo(indices[0])
        welded_body = welded_plant.get_body(indices[0])
        for i, geom_id in enumerate(plant.GetCollisionGeometriesForBody(body)):
            shape = inspector.GetShape(geom_id)
            X_BG = inspector.GetPoseInFrame(geom_id)
            frame_name = "frame_" + model_name + "_" + welded_body.name() + "_" + str(i)
            frame = welded_plant.AddFrame(
                FixedOffsetFrame(frame_name, welded_body.body_frame(), X_BG)
            )
            welded_body_info.add_shape_info(ShapeInfo(shape, frame))
        welded_body_infos[model_name] = welded_body_info

    welded_station.finalize()

    return welded_station, welded_body_infos


def update_welded_station(welded_station, pose_fluents):
    """
    Update the poses of the welded objects in the welded
    PandaStation based on the poses in pose_fluents.

    Args:
        welded_station: the PandaStation will all manipulands welded in place
        (output from `create_welded_station`)
        pose_fluents: A list of tuples of the form
        [('atpose', <object_name>, <X_WO>), ...]
        where <object_name> is a string with the object name and <X_WO> is a
        pydrake.math.RigidTransform with the relative pose of the object.
        The poses of any objects not found in the list pose_fluents will be
        set to RigidTransform(RotationMatrix(), [1000, 0, 0]) so as not to
        interfere with planning
    Returns:
        None, but updates the welded station provided in welded_station
    """
    pass
