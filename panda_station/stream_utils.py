"""
This module contains the functions/generators used in TAMP: picking, placing,
and collision free motion planning
"""
import numpy as np
from ompl import base as ob
from ompl import geometric as og
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
    station,
    station_context,
    q_start,
    q_goal,
    ignore_endpoint_collisions=False,
    panda = None,
    verbose = False,
    use_min_clearance = None,
    interpolate = False,  # return (approx) all q's on the trajectory used for collision checking
):
    """
    Find a collision free trajectory from the configurations
    q_start to q_end (np.array) for the panda arm in
    PandaStation station with Context station context
    """
    plant, scene_graph = station.get_plant_and_scene_graph()
    plant_context = station.GetSubsystemContext(plant, station_context)
    scene_graph_context = station.GetSubsystemContext(scene_graph, station_context)
    if panda is None:
        panda = station.get_panda()
    query_output_port = scene_graph.GetOutputPort("query")

    class MyStateValidityChecker(ob.StateValidityChecker):
        """
        Custom state validity checker
        """

        def __init__(self, si):
            """
            Construct state validitiy checker
            """
            super().__init__(si)

        def isValid(self, state):
            """
            Check if a state is valid
            """
            q = state_to_q(state)
            if ignore_endpoint_collisions and ((np.all(q == q_start)) or (np.all(q == q_goal))):
                return True
            if use_min_clearance is not None:
                cl = self.clearance(state)
                return cl < use_min_clearance
            plant.SetPositions(plant_context, panda, q)
            query_object = query_output_port.Eval(scene_graph_context)
            return not query_object.HasCollisions()

        @staticmethod
        def clearance(state):
            """
            Compute min clearance if arm is in state `state`
            """
            q = state_to_q(state)
            plant.SetPositions(plant_context, panda, q)
            query_object = query_output_port.Eval(scene_graph_context)
            sdps = query_object.ComputeSignedDistancePairwiseClosestPoints(1.0)
            min_dist = np.inf
            for sdp in sdps:
                min_dist = min(sdp.distance, min_dist)
            return min_dist

    joint_limits = station.get_panda_joint_limits()
    space = ob.RealVectorStateSpace(NUM_Q)
    bounds = ob.RealVectorBounds(NUM_Q)
    for i in range(NUM_Q):
        bounds.setLow(i, joint_limits[i][0])
        bounds.setHigh(i, joint_limits[i][1])

    space.setBounds(bounds)
    si = ob.SpaceInformation(space)
    checker = MyStateValidityChecker(si)
    si.setStateValidityChecker(checker)
    si.setStateValidityCheckingResolution(0.005) # half of default
    si.setup()

    start = q_to_state(space, q_start)
    if not checker.isValid(start):
        if verbose:
            print(f"{Colors.RED}INVALID OMPL START STATE {Colors.RESET}")
        return None
    goal = q_to_state(space, q_goal)
    if not checker.isValid(goal):
        if verbose:
            print(f"{Colors.RED}INVALID OMPL GOAL STATE{Colors.RESET}")
        return None

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)

    planner = og.LBKPIECE1(si)
    planner.setProblemDefinition(pdef)
    planner.setBorderFraction(0.1)
    planner.setup()

    solved = planner.solve(ob.CostConvergenceTerminationCondition(pdef))
    if not solved:
        if verbose:
            print(f"{Colors.RED}FAILED TO FIND OMPL SOLUTION{Colors.RESET}")
        return None

    simplifier = og.PathSimplifier(si)
    path = pdef.getSolutionPath()
    simplifier.simplify(path, 10)

    if interpolate:
        path.interpolate()

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
            