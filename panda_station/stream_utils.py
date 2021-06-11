"""
This module contains the functions/generators used in TAMP: picking, placing,
and collision free motion planning
"""
import numpy as np
from ompl import base as ob
from ompl import geometric as og
from pydrake.all import Box, Cylinder, Sphere
from .grasping_and_placing import (
    cylinder_grasp_q,
    box_grasp_q,
    sphere_grasp_q,
    is_graspable
) 

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


def find_traj(station, station_context, q_start, q_goal):
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
        plant.SetPositions(plant_context, panda, q)
        query_object = query_output_port.Eval(scene_graph_context)
        return query_object.HasCollisions()

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
        print("INVALID OMPL START STATE")
        return None
    goal = q_to_state(space, q_goal)
    if not isStateValid(goal):
        print("INVALID OMPL GOAL STATE")
        return None
    ss.setStartAndGoalStates(start, goal)
    solved = ss.solve()
    assert solved
    ss.simplifySolution()
    path = ss.getSolutionPath()
    res = np.array([state_to_q(state) for state in path.getStates()])
    return res



def find_grasp_q(station, station_context, shape_info):
    """
    Find a grasp configuration for the panda arm grasping 
    shape_info

    Args:
        station: a PandaStation with welded fingers
        station_context: the Context for the station
        shape_info: the ShapeInfo instance to try and grasp. 
    
    Returns:
        A tuple of the form
        (grasp_q, cost)
    """
    if not is_graspable(shape_info):
        return None, np.inf
    if isinstance(shape_info.shape, Sphere):
        return sphere_grasp_q(station, station_context, shape_info)
    if isinstance(shape_info.shape, Box):
        return box_grasp_q(station, station_context, shape_info)
    if isinstance(shape_info.shape, Cylinder):
        return cylinder_grasp_q(station, station_context, shape_info)
