"""
This module contains the functions/generators used in TAMP: picking, placing,
and collision free motion planning
"""
import numpy as np
from ompl import base as ob
from ompl import geometric as og
import pydrake.all

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
        print(q)
        query_object = query_output_port.Eval(scene_graph_context)
        pairs = query_object.ComputePointPairPenetration()
        print(pairs)
        #print(query_object.HasCollisions())
        return len(pairs) > 0

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
        print("INVALID START STATE")
        return None
    goal = q_to_state(space, q_goal)
    if not isStateValid(goal):
        print("INVALID GOAL STATE")
        return None
    ss.setStartAndGoalStates(start, goal)
    solved = ss.solve()
    assert solved
    ss.simplifySolution()
    path = ss.getSolutionPath()
    res = np.array([state_to_q(state) for state in path.getStates()])
    return res
