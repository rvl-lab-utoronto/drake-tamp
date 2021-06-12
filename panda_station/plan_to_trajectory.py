"""
A module for converting the plan supplied from pddlstream to
two PiecewisePolynominal's that the panda and its hand can follow
"""
import numpy as np
from pydrake.all import PiecewisePolynomial

JOINTSPACE_SPEED = np.pi/3 # rad/s
MAX_OPEN = 0.08
MAX_CLOSE = 0.0
GRASP_TIME = 2.0

def distance(q1, q2):
    """
    Returns the distance between two joint configurations, q1 and q2
    (np.arrays)
    """
    return np.sqrt((q2 - q1).dot(q2 - q1))

def make_panda_traj(qs, start_time):
    """
    Given a list of joint configs `qs`, 
    return a trajectory for the panda arm starting at start_time,
    and the times along the trajectory
    """
    times = [start_time]

    for i in range(len(qs) - 1):
        q_now = qs[i]
        q_next = qs[i+1]
        dist = distance(q_now, q_next)
        times.append(times[-1] + dist/JOINTSPACE_SPEED)

    times = np.array(times)
    panda_traj = None
    if len(qs) == 2:
        panda_traj = PiecewisePolynomial.FirstOrderHold(times, qs.T)
    else:
        panda_traj = PiecewisePolynomial.CubicShapePreserving(times, qs.T)
    
    return panda_traj, times

def move_free_traj(args, start_time):
    """
    Given the args for a move-free action,
    return the trajectory the panda and the hand should follow,
    given that the start time of the trajectory is `start_time`
    """
    # args: (?arm ?start ?end ?t)
    traj = args[3]
    panda_traj, times = make_panda_traj(traj, start_time)
    hand_traj = np.array([[MAX_OPEN] for i in range(len(times))])
    hand_traj = PiecewisePolynomial.FirstOrderHold(times, hand_traj.T)
    return panda_traj, hand_traj

def move_holding_traj(args, start_time):
    """
    Given the args for a move-holding action,
    return the trajectory the panda and the hand should follow,
    given that the start time of the trajectory is `start_time`
    """
    # args (?arm ?item ?startconf ?endconf ?t)
    traj = args[4]
    panda_traj, times = make_panda_traj(traj, start_time)
    hand_traj = np.array([[MAX_CLOSE] for i in range(len(times))])
    hand_traj = PiecewisePolynomial.FirstOrderHold(times, hand_traj.T)
    return panda_traj, hand_traj

def pick_traj(args, start_time):
    """
    Given the args for a pick action, return the trajectory 
    the panda and the hand should follow starting at start_time
    """
    #args (?arm ?item ?pose ?grasppose ?graspconf ?pregraspconf ?postgraspconf)
    # down
    grasp_q = args[4]
    pregrasp_q = args[5]
    postgrasp_q = args[6]
    down_traj = np.array([pregrasp_q, grasp_q])
    panda_traj_down, times = make_panda_traj(down_traj, start_time)
    hand_traj_down = np.array([[MAX_OPEN] for i in range(len(times))])
    hand_traj_down = PiecewisePolynomial.FirstOrderHold(times, hand_traj_down.T)
    #closing
    times = np.array([times[-1], times[-1] + GRASP_TIME])
    closing_traj = np.array([grasp_q, grasp_q])
    panda_traj_closing = PiecewisePolynomial.FirstOrderHold(times, closing_traj.T)
    hand_traj_closing = np.array([[MAX_OPEN], [MAX_CLOSE]])
    hand_traj_closing = PiecewisePolynomial.FirstOrderHold(times, hand_traj_closing.T)
    # up
    up_traj = np.array([grasp_q, postgrasp_q])
    panda_traj_up, times = make_panda_traj(up_traj, times[-1])
    hand_traj_up = np.array([[MAX_CLOSE] for i in range(len(times))])
    hand_traj_up = PiecewisePolynomial.FirstOrderHold(times, hand_traj_up.T)
    # concatenate
    panda_traj = panda_traj_down
    panda_traj.ConcatenateInTime(panda_traj_closing)
    panda_traj.ConcatenateInTime(panda_traj_up)
    hand_traj = hand_traj_down
    hand_traj.ConcatenateInTime(hand_traj_closing)
    hand_traj.ConcatenateInTime(hand_traj_up)
    return panda_traj, hand_traj


def place_traj(args, start_time):
    """
    Given the args for a place action, return the trajectory 
    the panda and the hand should follow starting at start_time
    """
    #args (?arm ?item ?region ?grasppose ?placepose ?placeconf ?preplaceconf ?postplaceconf)
    # down
    place_q = args[5]
    preplace_q = args[6]
    postplace_q = args[7]
    down_traj = np.array([preplace_q, place_q])
    panda_traj_down, times = make_panda_traj(down_traj, start_time)
    hand_traj_down = np.array([[MAX_CLOSE] for i in range(len(times))])
    hand_traj_down = PiecewisePolynomial.FirstOrderHold(times, hand_traj_down.T)
    #closing
    times = np.array([times[-1], times[-1] + GRASP_TIME])
    closing_traj = np.array([place_q, place_q])
    panda_traj_closing = PiecewisePolynomial.FirstOrderHold(times, closing_traj.T)
    hand_traj_closing = np.array([[MAX_CLOSE], [MAX_OPEN]])
    hand_traj_closing = PiecewisePolynomial.FirstOrderHold(times, hand_traj_closing.T)
    # up
    up_traj = np.array([place_q, postplace_q])
    panda_traj_up, times = make_panda_traj(up_traj, times[-1])
    hand_traj_up = np.array([[MAX_CLOSE] for i in range(len(times))])
    hand_traj_up = PiecewisePolynomial.FirstOrderHold(times, hand_traj_up.T)
    # concatenate
    panda_traj = panda_traj_down
    panda_traj.ConcatenateInTime(panda_traj_closing)
    panda_traj.ConcatenateInTime(panda_traj_up)
    hand_traj = hand_traj_down
    hand_traj.ConcatenateInTime(hand_traj_closing)
    hand_traj.ConcatenateInTime(hand_traj_up)
    return panda_traj, hand_traj


#TODO(agro): generalize this with a (better) action map
ACTION_MAP = {
    "move-free": move_free_traj,
    "move-holding": move_holding_traj,
    "pick": pick_traj,
    "place": place_traj
}

def plan_to_trajectory(plan, director, start_time):
    """
    Converts a pddl plan into two PiecewisePolynomial trajectories,
    one for the panda to follow, and the other for it's hand to follow.
    It adds all plans to the provided TrajectoryDirector `director`

    Args:
        plan: an array of the form
            [action1, action2, ...]
            where each action has the attributes:
            name: the name of the action (string)
            args: A tuple with the action arguments
        director: the TrajectoryDirectoy
        start_time: the start time for the trajectory
    Returns:
        panda_traj: the trajectory for the panda to follow
        hand_traj: the trajectory for the hand to follow
    """

    time = start_time
    for action in plan:
        new_panda_traj, new_hand_traj = ACTION_MAP[action.name](action.args, time)
        time = new_panda_traj.end_time()
        director.add_panda_traj(new_panda_traj)
        director.add_hand_traj(new_hand_traj)

