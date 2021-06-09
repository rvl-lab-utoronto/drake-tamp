"""
A module with utilities allowing the PandaStation to interface 
with PDDL and the streams (generators)
"""
import pydrake.all
from . import panda_station


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
    plant_context = plant.GetMyContextFromRoot(station_context)
    for object_info in station.object_infos:
        body = plant.get_body(object_info.main_body_index)
        X_WO = body.EvalPoseInWorld(plant_context)
        start_poses[object_info.name] = X_WO
    return start_poses


def parse_inital_config(station, station_context):
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
    #TODO(agro): current, panda station only supports one arm, make this more
    panda = station.get_panda()
    plant = station.get_multibody_plant()
    plant_context = plant.GetMyContextFromRoot(station_context)
    q = plant.GetPositions(plant_context, panda)
    arms = {}
    arms["panda"] = q
    return arms

def parse_tables(station, station_context):
    """
    TODO(agro)
    """
    pass


def create_welded_station(station, station_context):
    """
    Create new PandaStation with all free floating objects in
    PandaStation station welded to fixedoffset frames with the
    same pose in world as in the provided context station_context.
    Also adds fixed offset frames for every geometry for each one of
    those objects.

    Args:
        station: PandaStation instance
        station_context: the station context to parse
    Returns:
        The welded PandaStation as described above with a list of 
        BodyInfo objects for each object in the world
    """
    pass

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
