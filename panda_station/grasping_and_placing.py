"""
This module contains functions to assist with grasping and placing:

cylinder_grasp_q(station, station_context, shape_info)
box_grasp_q(station, station_context, shape_info)
sphere_grasp_q(station, station_context, shape_info)
"""
import numpy as np
from pydrake.all import (
    RotationMatrix,
    RigidTransform,
    Solve,
    InverseKinematics,
    Box,
    Cylinder,
    Sphere,
) 

NUM_Q = 7  # DOF of panda arm
GRASP_WIDTH = 0.08  # distance between fingers at max extension
GRASP_HEIGHT = 0.0535  # distance from hand to tips of fingers
COL_MARGIN = 0.001  # acceptable margin of error for collisions
GRASP_MARGIN = 0.006  # margin for grasp planning
Q_NOMINAL = np.array([0., 0.55, 0., -1.45, 0., 1.58, 0.])

def is_graspable(shape_info):
    """
    For a given ShapeInfo object, returns True if the object is
    graspable by the panda hand (not nessecarily reachable)
    """
    shape = shape_info.shape
    if isinstance(shape, Sphere):
        if (shape.radius() < GRASP_MARGIN) or (
            shape.radius() > GRASP_HEIGHT - GRASP_MARGIN
        ):
            return False
    if isinstance(shape, Cylinder):
        if (shape.radius() > GRASP_WIDTH - 2 * GRASP_MARGIN) and (
            shape.length() > GRASP_WIDTH - 2 * GRASP_MARGIN
        ):
            return False
    if isinstance(shape, Box):
        min_dim = min([shape.depth(), shape.width(), shape.height()])
        if min_dim > GRASP_WIDTH - 2 * GRASP_MARGIN:
            return False
    return True

def box_grasp_q(
    station, 
    station_context, 
    shape_info, 
    q_nominal = Q_NOMINAL,
    initial_guess = Q_NOMINAL,
):
    """
    Find a grasp configuration for the panda arm grasping 
    shape_info, given that it is  Box

    Args:
        station: a PandaStation with welded fingers

        station_context: the Context for the station

        shape_info: the ShapeInfo instance to try and grasp. 
        it is assumed that isinstance(shape_info.shape, Box) 
        is True

        q_nominal: comfortable joint positions

        q_inital: initial guess for mathematical program
    Returns:
        A tuple of the form
        (grasp_q, cost)
    """
    pass


def cylinder_grasp_q(
    station, 
    station_context, 
    shape_info, 
    q_nominal = Q_NOMINAL,
    initial_guess = Q_NOMINAL,
):
    """
    Find a grasp configuration for the panda arm grasping 
    shape_info, given that it is Cylinder

    Args:
        station: a PandaStation with welded fingers

        station_context: the Context for the station

        shape_info: the ShapeInfo instance to try and grasp. 
        it is assumed that isinstance(shape_info.shape, Cylinder) 
        is True

        q_nominal: comfortable joint positions

        q_inital: initial guess for mathematical program
    Returns:
        A tuple of the form
        (grasp_q, cost)
    """
    pass

def sphere_grasp_q(
    station, 
    station_context, 
    shape_info, 
    q_nominal = Q_NOMINAL,
    initial_guess = Q_NOMINAL,
):
    """
    Find a grasp configuration for the panda arm grasping 
    shape_info, given that it is Sphere

    Args:
        station: a PandaStation with welded fingers

        station_context: the Context for the station

        shape_info: the ShapeInfo instance to try and grasp. 
        it is assumed that isinstance(shape_info.shape, Sphere) 
        is True

        q_nominal: comfortable joint positions

        q_inital: initial guess for mathematical program
    Returns:
        A tuple of the form
        (grasp_q, cost)
    """
    pass