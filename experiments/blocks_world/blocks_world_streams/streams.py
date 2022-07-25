"""
This module contains functions that assist with grasping and placing
in the kitchen environment
"""
from panda_station.grasping_and_placing import DROP_HEIGHT
import random
import numpy as np
from numpy import random
from pydrake.all import (
    Solve,
    InverseKinematics,
    Box,
    Cylinder,
    Sphere,
    RotationMatrix,
    RigidTransform,
)

NUM_Q = 7  # DOF of panda arm
GRASP_WIDTH = 0.08  # distance between fingers at max extension
GRASP_HEIGHT = 0.0535  # distance from hand to tips of fingers
FINGER_WIDTH = 0.017
# distance along z axis from hand frame origin to fingers
HAND_HEIGHT = 0.1
COL_MARGIN = 0.0  # acceptable margin of error for collisions
CONSIDER_MARGIN = 0.1
GRASP_MARGIN = 0.006  # margin for grasp planning
Q_NOMINAL = np.array([0.0, 0.55, 0.0, -1.45, 0.0, 1.58, 0.0])
HAND_FRAME_NAME = "panda_hand"
THETA_TOL = np.pi * 0.01
DROP_HEIGHT = 0.02

def find_grasp(shape_info):
    """
    Returns a handpose X_HI that is a grasp pose for
    `shape_info`
    """
    length = None
    z_rot = None
    if isinstance(shape_info.shape, Box):
        length = shape_info.shape.height()
        z_rot = (np.pi/2)*np.random.randint(0, 4)
    elif isinstance(shape_info.shape, Cylinder):
        length = shape_info.shape.length()
        z_rot = np.random.uniform(0, 2*np.pi)
    else:
        return None, np.inf
    h = HAND_HEIGHT + length - FINGER_WIDTH / 2
    R = RotationMatrix.MakeXRotation(np.pi).multiply(RotationMatrix.MakeZRotation(z_rot))
    return RigidTransform(R, [0, 0, h])

def find_table_place(station, station_context, shape_info, surface):
    """
    Returns a worldpose X_WI that is a stable placement pose for
    `shape_info` on `surface`
    """
    border = None
    length = None
    if isinstance(shape_info.shape, Box):
        border = max(shape_info.shape.width(), shape_info.shape.depth())/2
    if isinstance(shape_info.shape, Cylinder):
        border = shape_info.shape.radius()
    border = np.ones(3)*border
    lower = surface.bb_min + border
    upper = surface.bb_max - border
    plant, plant_context = get_plant_and_context(station, station_context)
    S = surface.shape_info.offset_frame
    #TODO(agro): make this random choice smarter
    p_SI_S = np.random.uniform(lower,upper)
    p_SI_S[2] = surface.bb_min[2] + 5e-3
    X_WS = plant.CalcRelativeTransform(plant_context, plant.world_frame(), S)
    p_WS_W = X_WS.translation()
    R_WS = X_WS.rotation()
    # R_WS*p_SI_S = p_WI_W
    p_WI_W = p_WS_W + R_WS.multiply(p_SI_S)
    R_WI = RotationMatrix.MakeZRotation(np.random.uniform(0, 2*np.pi))
    return RigidTransform(R_WI, p_WI_W)


def find_block_place(station, station_context, shape_info, surface):
    plant, plant_context = get_plant_and_context(station, station_context)
    S = surface.shape_info.offset_frame
    middle = (surface.bb_min + surface.bb_max)*0.5
    p_SI_S = middle
    p_SI_S[2] = surface.bb_min[2] + 1e-3
    X_WS = plant.CalcRelativeTransform(plant_context, plant.world_frame(), S)
    p_WS_W = X_WS.translation()
    R_WS = X_WS.rotation()
    p_WI_W = p_WS_W + R_WS.multiply(p_SI_S)
    R_WI = RotationMatrix.MakeZRotation(np.random.uniform(0, 2*np.pi)) #RotationMatrix()
    return RigidTransform(R_WI, p_WI_W)

def find_ik_with_relaxed(
    station,
    station_context,
    object_info,
    X_HI,
    panda_info,
    q_initial = Q_NOMINAL,
):
    """
    Find a solution to the IK problem that the hand must be at
    X_HI relative to shape_info (which is already positioned
    in the world frame). This solver will first solve without
    considering collisions, and then use that as the initial
    solution to solve while considering collisions
    """
    q0, cost = find_ik_with_handpose(
        station,
        station_context,
        object_info,
        X_HI,
        panda_info,
        q_initial = q_initial,
        relax = True
    )
    #return q0, cost
    if not np.isfinite(cost):
        return q0, cost
    return find_ik_with_handpose(
        station,
        station_context,
        object_info,
        X_HI,
        panda_info,
        q_initial = q0,
        relax = False
    )

def find_ik_with_handpose(
    station,
    station_context,
    object_info,
    X_HI,
    panda_info,
    q_initial = Q_NOMINAL,
    q_nominal = Q_NOMINAL,
    relax = False,
):
    """
    Find a solution to the IK problem that the hand must be at
    X_HI relative to shape_info (which is already positioned
    in the world frame).
    """
    plant, plant_context = get_plant_and_context(station, station_context)

    plant.SetPositions(plant_context, panda_info.panda, q_nominal)
    q_nominal = plant.GetPositions(plant_context)

    plant.SetPositions(plant_context, panda_info.panda, q_initial)
    q_initial = plant.GetPositions(plant_context)

    H = plant.GetFrameByName(HAND_FRAME_NAME, panda_info.hand)  # hand frame
    G = object_info.get_frame() #shape_info.offset_frame  # geometry frasinkme
    W = plant.world_frame()
    ik = InverseKinematics(plant, plant_context)
    if not relax:
        ik.AddMinimumDistanceConstraint(COL_MARGIN, CONSIDER_MARGIN)
    lower = X_HI.translation() - np.array([0.001, 0.001, 0.01])
    upper = X_HI.translation() + np.array([0.001, 0.001, 0.01])
    ik.AddPositionConstraint(
        H,
        np.zeros(3),
        G,
        lower,
        upper,
    )
    R_I = X_HI.rotation()
    ik.AddOrientationConstraint(
        H,
        RotationMatrix(),
        G,
        R_I,
        THETA_TOL
    )
    prog = ik.prog()
    q = ik.q()
    prog.AddQuadraticErrorCost(np.identity(len(q)), q_nominal, q)
    prog.SetInitialGuess(q, q_initial)
    result = Solve(prog)
    cost = result.get_optimal_cost()
    if not result.is_success():
        cost = np.inf
    return plant.GetPositions(plant_context, panda_info.panda), cost

def check_colfree_block(station, station_context, panda_name, q):
    """
    Checks for collisions if the panda is in conf q
    """
    plant, plant_context = get_plant_and_context(station, station_context)
    scene_graph = station.get_scene_graph()
    scene_graph_context = station.GetSubsystemContext(scene_graph, station_context)

    query_output_port = scene_graph.GetOutputPort("query")

    for panda_info in station.panda_infos.values():
        if panda_info.panda_name == panda_name:
            plant.SetPositions(plant_context, panda_info.panda, q)
        else:
            # this will guarentee no collisions with other panda
            plant.SetPositions(plant_context, panda_info.panda, Q_NOMINAL)

    query_object = query_output_port.Eval(scene_graph_context)
    #pairs = query_object.ComputePointPairPenetration()
    #for p in pairs:
        #print(p.id_A, p.id_B)
        #print(plant.GetCollisionGeometriesForBody(plant.GetBodyByName("panda_hand")))
        ##print(plant.GetCollisionGeometriesForBody(plant.GetBodyByName("panda_leftfinger")))
        ##print(plant.GetCollisionGeometriesForBody(plant.GetBodyByName("panda_rightfinger")))
    #for name, (info, _) in station.object_infos.items():
        #print(name)
        #print(plant.GetCollisionGeometriesForBody(info.main_body_info.body))
    return not query_object.HasCollisions()

def check_colfree_arms(station, station_context, arm1_name, q1, arm2_name, q2):

    plant, plant_context = get_plant_and_context(station, station_context)
    scene_graph = station.get_scene_graph()
    scene_graph_context = station.GetSubsystemContext(scene_graph, station_context)

    query_output_port = scene_graph.GetOutputPort("query")

    for q, arm_name in zip([q1, q2], [arm1_name, arm2_name]):
        plant.SetPositions(
            plant_context, station.panda_infos[arm_name].panda, q
        )

    query_object = query_output_port.Eval(scene_graph_context)
    return not query_object.HasCollisions()



def get_plant_and_context(station, station_context):
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    return plant, plant_context



