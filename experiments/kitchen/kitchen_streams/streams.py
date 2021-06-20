"""
This module contains functions that assist with grasping and placing
in the kitchen environment
"""
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
COL_MARGIN = 0.001  # acceptable margin of error for collisions
CONSIDER_MARGIN = 0.1
GRASP_MARGIN = 0.006  # margin for grasp planning
Q_NOMINAL = np.array([0.0, 0.55, 0.0, -1.45, 0.0, 1.58, 0.0])
HAND_FRAME_NAME = "panda_hand"
THETA_TOL = np.pi * 0.01
np.random.seed(0)
random.seed(0)

def find_place_q(
    station,
    station_context,
    shape_info,
    surface,
    q_nominal = Q_NOMINAL,
    q_initial = Q_NOMINAL
):
    """
    Find a joint conf to place shape_info on surface
    """
    border = None
    length = None
    if isinstance(shape_info.shape, Box):
        border = max(shape_info.shape.width(), shape_info.shape.depth())
        length = shape_info.shape.height()
    if isinstance(shape_info.shape, Cylinder):
        border = shape_info.shape.radius()
        length = shape_info.shape.length()
    border = np.ones(3)*border
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    hand = station.get_hand()
    # R: random frame, S: surface frame, H: hand frame, G: shape frame
    H = plant.GetFrameByName(HAND_FRAME_NAME, hand)  # hand frame
    G = shape_info.offset_frame  # geometry frame
    S = surface.shape_info.offset_frame
    max_tries = 100
    tries = 0
    while tries < max_tries:
        p_SR = np.random.uniform(surface.bb_min + border, surface.bb_max - border)
        lower = p_SR.copy()
        lower[2] = surface.bb_min[2]
        lower[0:2] -= np.ones(2)*0.01
        upper = p_SR.copy()
        upper[2] = surface.bb_max[2]
        upper[0:2] += np.ones(2)*0.01
        theta = np.random.uniform(0, 2 * np.pi)

        ik = InverseKinematics(plant, plant_context)
        ik.AddMinimumDistanceConstraint(COL_MARGIN, CONSIDER_MARGIN)
        ik.AddPositionConstraint(
            G,
            [0, 0, -length/2],
            S,
            lower,
            upper
        )
        ik.AddAngleBetweenVectorsConstraint(
            G, [0, 0, 1], plant.world_frame(), surface.z, 0, THETA_TOL
        )
        prog = ik.prog()
        q = ik.q()
        prog.AddQuadraticErrorCost(np.identity(len(q)), q_nominal, q)
        prog.SetInitialGuess(q, q_initial)
        result = Solve(prog)
        if not result.is_success():
            tries +=1 
            continue
        cost = result.get_optimal_cost()
        return result.GetSolution(q), cost

    return None, np.inf

def find_grasp_q(
    station,
    station_context,
    shape_info,
    q_nominal = Q_NOMINAL,
    q_initial = Q_NOMINAL
):
    """
    Returns a grasp configuration for grasping one of the objects 
    in the kitchen environment (cabbage, raddish, or glass)
    """  
    # cabbage, raddish and glass all have one shape 
    if isinstance(shape_info.shape, Box):
        return box_grasp_q(
            station,
            station_context,
            shape_info,
            q_nominal = q_nominal,
            q_initial = q_initial
        )
    if isinstance(shape_info.shape, Cylinder):
        return cylinder_grasp_q(
            station,
            station_context,
            shape_info,
            q_nominal = q_nominal,
            q_initial = q_initial
        )
    assert False, "Trying to grasp invalid shape"


def box_grasp_q(
    station,
    station_context,
    shape_info,
    q_nominal = Q_NOMINAL,
    q_initial = Q_NOMINAL
):
    box = shape_info.shape
    height = box.height() # we know that these will be upright

    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    hand = station.get_hand()
    H = plant.GetFrameByName(HAND_FRAME_NAME, hand)  # hand frame
    G = shape_info.offset_frame  # geometry frame

    axes = [0, 1]
    random.shuffle(axes)
    signs = [-1, 1]
    random.shuffle(signs)
    
    for sign in signs:
        for axis in axes:
            ik = InverseKinematics(plant, plant_context)
            lower = -GRASP_MARGIN*np.ones(3)
            upper = [GRASP_MARGIN, GRASP_MARGIN, 0]
            ik.AddMinimumDistanceConstraint(COL_MARGIN, CONSIDER_MARGIN)
            ik.AddPositionConstraint(
                H, [0, 0, HAND_HEIGHT + height/2 - 0.02], G, lower, upper
            )
            ik.AddAngleBetweenVectorsConstraint(
                H,
                [0, 0, -1],
                G,
                [0, 0, 1],
                0.0,
                THETA_TOL
            )
            n = np.zeros(3)
            n[axis] = sign
            ik.AddAngleBetweenVectorsConstraint(
                H,
                [0, 1, 0],
                G,
                n,
                0.0,
                THETA_TOL
            )
            prog = ik.prog()
            q = ik.q()
            prog.AddQuadraticErrorCost(np.identity(len(q)), q_nominal, q)
            prog.SetInitialGuess(q, q_initial)
            result = Solve(prog)
            if not result.is_success():
                continue
            cost = result.get_optimal_cost()
            return result.GetSolution(q), cost
    return None, np.inf

def cylinder_grasp_q(
    station,
    station_context,
    shape_info,
    q_nominal = Q_NOMINAL,
    q_initial = Q_NOMINAL
):
    cylinder = shape_info.shape
    length = cylinder.length() # we know that these will be upright
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    hand = station.get_hand()
    H = plant.GetFrameByName(HAND_FRAME_NAME, hand)  # hand frame
    G = shape_info.offset_frame  # geometry frame
    
    ik = InverseKinematics(plant, plant_context)
    ik.AddMinimumDistanceConstraint(COL_MARGIN, CONSIDER_MARGIN)
    lower = -0.005*np.ones(3)
    upper = [0.005, 0.005, 0]
    ik.AddPositionConstraint(
        H, [0, 0, HAND_HEIGHT + length/2 - 0.015], G, lower, upper
    )
    ik.AddAngleBetweenVectorsConstraint(
        H,
        [0, 0, -1],
        G,
        [0, 0, 1],
        0.0,
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
    return result.GetSolution(q), cost