"""
This module contains the functions/generators used in TAMP: picking, placing,
and collision free motion planning
"""
import numpy as np
from ompl import base as ob
from ompl import geometric as og
import pydrake.all
from pydrake.all import (
    Box,
    Cylinder,
    Sphere,
)
from .grasping_and_placing import (
    Q_NOMINAL,
    cylinder_grasp_q,
    box_grasp_q,
    is_placeable,
    is_safe_to_place,
    q_to_X_WH,
    sphere_grasp_q,
    is_graspable,
    sphere_place_q,
    box_place_q,
    cylinder_place_q,
)
from .utils import *
NUM_Q = 7


def state_to_q(state):
    """
    Parses ompl RealVectorStateSpae::StateType into a numpy array
    """
    q = []
    for i in range(NUM_Q):
        q.append(state[i])
    return np.array(q)


def q_to_state(space, q):
    """
    Turns a numpy array in to a RealVectorStateSpae::StateType
    """
    state = ob.State(space)
    for i in range(len(q)):
        state[i] = q[i]
    return state


def find_traj(
    station, station_context, q_start, q_goal, ignore_endpoint_collisions=True
):
    """
    Find a collision free trajectory from the configurations
    q_start to q_end (np.array) for the panda arm in
    PandaStation station with Context station context
    """
    plant, scene_graph = station.get_plant_and_scene_graph()
    plant_context = station.GetSubsystemContext(plant, station_context)
    scene_graph_context = station.GetSubsystemContext(scene_graph, station_context)
    panda = station.get_panda()
    query_output_port = scene_graph.GetOutputPort("query")

    def is_colliding(q):
        #if np.all(q == q_start) or (np.all(q == q_goal) and ignore_endpoint_collisions):
            #return False
        plant.SetPositions(plant_context, panda, q)
        query_object = query_output_port.Eval(scene_graph_context)
        return query_object.HasCollisions()
        """
        sdps = query_object.ComputeSignedDistancePairwiseClosestPoints(0.1)
        for sdp in sdps:
            if sdp.distance < 0.001:
                return True
        return False
        pairs = query_object.ComputePointPairPenetration()
        max_pen = -np.inf
        for p in pairs:
            max_pen = max(max_pen, p.depth)
        if np.all(q == q_start) or np.all(q == q_goal):
            print("MAX PEN:", max_pen)
        return max_pen > 0.001
        """

    def isStateValid(state):
        q = state_to_q(state)
        return not is_colliding(q)

    joint_limits = station.get_panda_joint_limits()
    space = ob.RealVectorStateSpace(NUM_Q)
    bounds = ob.RealVectorBounds(NUM_Q)
    for i in range(NUM_Q):
        bounds.setLow(i, joint_limits[i][0])
        bounds.setHigh(i, joint_limits[i][1])

    space.setBounds(bounds)
    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
    start = q_to_state(space, q_start)
    if not isStateValid(start):
        print(f"{Colors.RED}INVALID OMPL START STATE {Colors.RESET}")
        return None
    goal = q_to_state(space, q_goal)
    if not isStateValid(goal):
        print(f"{Colors.RED}INVALID OMPL GOAL STATE{Colors.RESET}")
        return None
    ss.setStartAndGoalStates(start, goal)
    solved = ss.solve()
    if not solved:
        print(f"{Colors.RED}FAILED TO FIND OMPL SOLUTION{Colors.RESET}")
        return None
    ss.simplifySolution()
    path = ss.getSolutionPath()
    res = np.array([state_to_q(state) for state in path.getStates()])
    return res

def best_grasp_for_shapes(
    station,
    station_context, 
    shape_infos,
    q_nominal = Q_NOMINAL,
    initial_guess = Q_NOMINAL,
):
    """
    Return the best grasp config, trying to grasp all shapes in 
    `shape_infos`.

    Args:
        station: PandaStation
        station_context: the Context for station
        shape_infos: the ShapeInfo's for the shapes to try and grab
        q_nominal: nominal joint positions (np.array)
        initial_guess: initial guess for joint positions (np.array)

    Returns:
        A tuple
        (grasp_q, cost)
        represnting the lowest cost grasp configuration `grasp_q`
        that could be found
    """
    grasp_qs, costs = [], []
    grasp_funcs = [box_grasp_q, cylinder_grasp_q, sphere_grasp_q]
    for shape_info in shape_infos:
        func = None
        if isinstance(shape_info.shape, Box):
            func = grasp_funcs[0]
        if isinstance(shape_info.shape, Cylinder):
            func = grasp_funcs[1]
        if isinstance(shape_info.shape, Sphere):
            func = grasp_funcs[2]
        grasp_q, cost = func(
            station,
            station_context,
            shape_info,
            initial_guess = initial_guess,
            q_nominal = q_nominal
        )
        grasp_qs.append(grasp_q)
        costs.append(cost)
    # return the lowest cost grasp for all ShapeInfo's
    indices = np.argsort(costs)
    return grasp_qs[indices[0]], costs[indices[0]]

def best_place_shapes_surfaces(
    station,
    station_context,
    holding_shape_infos,
    target_surfaces,
    initial_guess = Q_NOMINAL
):
    """
    Return the best place config, trying to place all shapes in 
    holding_shapes on the Surfaces in target_surfaces 

    Args:
        station: PandaStation
        station_context: the Context for station
        holding_shape_infos: the ShapeInfo's for object being held.
        It is assumed that for all shape_infos in holding_shape_infos,
        is_placeable(shape_info) -> True
        target_surfaces: a list of the target surfaces to try. It is 
        assumed that these surfaces are safe to place on

    Returns:
        A tuple
        (place_q, cost)
        representing the lowest cost grasp configuration `grasp_q`
        that could be found
    """

    place_funcs = [box_place_q, cylinder_place_q, sphere_place_q]
    place_qs = []
    costs = []
    for shape_info in holding_shape_infos:
        func = None
        if isinstance(shape_info.shape, Box):
            func = place_funcs[0]
        if isinstance(shape_info.shape, Cylinder):
            func = place_funcs[1]
        if isinstance(shape_info.shape, Sphere):
            func = place_funcs[2]
        for surface in target_surfaces:
            place_q, cost = func(
                station,
                station_context,
                shape_info,
                surface,
                initial_guess = initial_guess
            )
            place_qs.append(place_q)
            costs.append(cost)

    indices = np.argsort(costs) 
    return place_qs[indices[0]], costs[indices[0]]
            