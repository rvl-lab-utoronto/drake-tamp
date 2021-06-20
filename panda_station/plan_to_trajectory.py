import numpy as np
from numpy.core.defchararray import join
from pydrake.all import PiecewisePolynomial

JOINTSPACE_SPEED = np.pi*0.1 # rad/s
MAX_OPEN = 0.08
MAX_CLOSE = 0.0
GRASP_TIME = 2.0
hand_q = MAX_OPEN

def jointspace_distance(q1, q2):
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
    #TODO(agro): make the speed slowest at start and end 
    times = [start_time]
    for i in range(len(qs) - 1):
        q_now = qs[i]
        q_next = qs[i+1]
        dist = max(jointspace_distance(q_now, q_next), 1e-2)
        times.append(times[-1] + dist/JOINTSPACE_SPEED)

    times = np.array(times)
    panda_traj = PiecewisePolynomial.FirstOrderHold(times, qs.T)
    return panda_traj, times

def make_hand_traj(q_start, q_end, start_time, end_time):
    """
    Given a list of joint configs `qs`, 
    return a trajectory for the panda arm starting at start_time,
    and the times along the trajectory
    """
    # make the speed slowest at start and end 
    times = np.array([start_time, end_time])
    qs = np.array([[q_start], [q_end]])
    hand_traj = PiecewisePolynomial.FirstOrderHold(times, qs.T)
    return hand_traj, times

def move(traj, start_time):
    """
    Return a panda trajectory along np.array `traj` with the hand static 
    at `hand_q` the hole time
    """
    global hand_q
    panda_traj, times = make_panda_traj(traj, start_time)
    hand_traj = make_hand_traj(hand_q, hand_q, times[0], times[-1])[0]
    return panda_traj, hand_traj

def pick_or_place(
    q_pre,
    q,
    q_post,
    start_time,
    hand_start = MAX_OPEN,
    hand_end = MAX_CLOSE
):
    """
    Return pick or place trajectories, depending on 
    `hand_start` and `hand_end`
    """
    panda_traj = np.array([q_pre, q, q, q_post])
    hand_traj = np.array([[hand_start], [hand_start], [hand_end], [hand_end]])
    down_time = max(jointspace_distance(q_pre, q)/JOINTSPACE_SPEED, 1e-2)
    up_time = max(jointspace_distance(q_post, q)/JOINTSPACE_SPEED, 1e-2)
    times = np.array([
        start_time,
        start_time + down_time,
        start_time + down_time + GRASP_TIME,
        start_time + down_time + GRASP_TIME + up_time
    ])
    panda_traj = PiecewisePolynomial.FirstOrderHold(times, panda_traj.T)
    hand_traj = PiecewisePolynomial.FirstOrderHold(times, hand_traj.T)
    return panda_traj, hand_traj

def pick(q_pre, q, q_post, start_time):
    """
    Return pick trajectories 
    """
    return pick_or_place(
        q_pre,
        q,
        q_post,
        start_time,
        hand_start = MAX_OPEN,
        hand_end = MAX_CLOSE
    )

def place(q_pre, q, q_post, start_time):
    """
    Return place trajectories
    """
    return pick_or_place(
        q_pre,
        q,
        q_post,
        start_time,
        hand_start = MAX_CLOSE,
        hand_end = MAX_OPEN
    )

def make_trajectory(plan, director, start_time, action_map):
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
        action_map:
            A dict of the form
            action_map = {
                "move": (
                    plan_to_trajectory.move,
                    [1],
                ),
                "pick":(
                    plan_to_trajectory.pick,
                    [0, 1, 0]
                ),
                "place":(
                    plan_to_trajectory.place,
                    [0, 1, 0]
                )
                "<name>":(
                    plan_to_trajectory.<function>,
                    <argmap>
                )
            }
            where <name> is the name of the action,
            <function> is the function that will produce
            the trajectory based on the action arguments,
            and <argmap> is a list of integers
            such that action[<argmap>[i]] is supplied to the
            ith argument to <function>
    Returns:
        None, modifies the directory
    """
    global hand_q
    time = start_time
    for action in plan:
        if action.name not in action_map:
            continue
        func = action_map[action.name][0]
        arg_map = action_map[action.name][1]
        args = [action.args[i] for i in arg_map]
        args.append(time)
        new_panda_traj, new_hand_traj = func(*args)
        director.add_panda_traj(new_panda_traj)
        director.add_hand_traj(new_hand_traj)
        time = new_panda_traj.end_time()
        hand_q = new_hand_traj.value(time).flatten()[0]