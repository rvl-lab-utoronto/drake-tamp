from panda_station.trajectory_generation import MotionGenerator
import numpy as np
from pydrake.all import PiecewisePolynomial
from enum import Enum

JOINTSPACE_SPEED = np.pi*0.1 # rad/s
MAX_OPEN = 0.08
MAX_CLOSE = 0.0
GRASP_TIME = 1.35
#TODO(agro): pass in as argument
Q_INITIAL =np.array([0.0, 0.1, 0, -1.2, 0, 1.6, 0])
SPEED_FACTOR = 0.2
DOF = 7

class TrajType(Enum):

    LINEAR = 1
    CUBIC = 2
    GENERATOR = 3

class PlanToTrajectory:

    def __init__(self, station, traj_mode = TrajType.LINEAR):
        self.station = station
        self.trajectories = {}
        for panda_name in self.station.panda_infos:
            self.trajectories[panda_name] = {
                "panda_traj": None,
                "hand_traj": None,
            }
        self.curr_time = 0
        self.traj_mode = traj_mode

    @staticmethod
    def jointspace_distance(q1, q2):
        """
        Returns the distance between two joint configurations, q1 and q2
        (np.arrays)
        """
        return np.sqrt((q2 - q1).dot(q2 - q1))

    def get_trajs(self, panda_name):
        hand_traj = self.trajectories[panda_name]["hand_traj"]
        panda_traj = self.trajectories[panda_name]["panda_traj"]
        return panda_traj, hand_traj

    def add_trajs(self, panda_name, panda_traj, hand_traj):
        if self.trajectories[panda_name]["panda_traj"] is None:
            self.trajectories[panda_name]["panda_traj"] = panda_traj
            self.trajectories[panda_name]["hand_traj"] = hand_traj
        else:
            self.trajectories[panda_name]["panda_traj"].ConcatenateInTime(panda_traj)
            self.trajectories[panda_name]["hand_traj"].ConcatenateInTime(hand_traj)

    def get_curr_time(self, panda_name):
        if self.trajectories[panda_name]['panda_traj'] is None:
            return self.curr_time
        else:
            panda_traj, _ = self.get_trajs(panda_name)
            return panda_traj.end_time()

    def get_curr_q(self, panda_name):
        time = self.get_curr_time(panda_name)
        panda_traj, hand_traj = self.get_trajs(panda_name)
        if panda_traj is None:
            return Q_INITIAL, [[MAX_OPEN]]
        return panda_traj.value(time).flatten(), hand_traj.value(time)

    @staticmethod
    def make_generator_panda_traj(qs, start_time):
        panda_traj = []
        times = []
        res_qs = []
        for q1, q2 in zip(qs[:-1], qs[1:]):
            gen = MotionGenerator(SPEED_FACTOR, q1, q2)
            t = 0
            while True:
                times.append(t + start_time)
                q, finished = gen(t)
                res_qs.append(q.reshape((DOF, 1)))
                if finished:
                    break
                t += 1e-3

        qs = np.concatenate(res_qs, axis = 1)
        times = np.array(times)
        panda_traj = PiecewisePolynomial.FirstOrderHold(times, qs)
        return panda_traj, times

    def make_panda_traj(self, qs, start_time):
        """
        Given a list of joint configs `qs`, 
        return a trajectory for the panda arm starting at start_time,
        and the times along the trajectory
        """
        #TODO(agro): make the speed slowest at start and end 

        if self.traj_mode == TrajType.GENERATOR:
            return self.make_generator_panda_traj(qs, start_time)

        times = [start_time]
        for i in range(len(qs) - 1):
            q_now = qs[i]
            q_next = qs[i+1]
            dist = max(PlanToTrajectory.jointspace_distance(q_now, q_next), 1e-2)
            times.append(times[-1] + dist/JOINTSPACE_SPEED)

        times = np.array(times)
        if self.traj_mode == TrajType.LINEAR:
            panda_traj = PiecewisePolynomial.FirstOrderHold(times, qs.T)
        elif self.traj_mode == TrajType.CUBIC:
            panda_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(times, qs.T, np.zeros((DOF, 1)), np.zeros((DOF, 1)))
        return panda_traj, times


    @staticmethod
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

    def stay_still(self, panda_name, start_time, end_time):
        panda_q, hand_q = self.get_curr_q(panda_name)
        times = np.array([start_time, end_time])
        qs = np.array([panda_q, panda_q])
        panda_traj = PiecewisePolynomial.FirstOrderHold(times, qs.T)
        qs = np.array([hand_q[0], hand_q[0]])
        hand_traj = PiecewisePolynomial.FirstOrderHold(times, qs.T)
        self.add_trajs(panda_name, panda_traj, hand_traj)

    def move(self, panda_name, traj):
        """
        Return a panda trajectory along np.array `traj` with the hand static 
        at `hand_q` the hole time
        """
        panda_traj, hand_traj = self.get_trajs(panda_name)
        _, last_hand_q = self.get_curr_q(panda_name)
        start_time = self.get_curr_time(panda_name)
        panda_traj, times = self.make_panda_traj(traj, start_time)
        hand_traj = self.make_hand_traj(
            last_hand_q[0][0], last_hand_q[0][0], times[0], times[-1]
        )[0]
        self.add_trajs(panda_name, panda_traj, hand_traj)

    def pick_or_place(
        self,
        panda_name,
        q_pre,
        q,
        q_post,
        hand_start = MAX_OPEN,
        hand_end = MAX_CLOSE
    ):
        """
        Return pick or place trajectories, depending on 
        `hand_start` and `hand_end`
        """
        qs = np.array([q_pre, q, q, q_post])
        start_time = self.get_curr_time(panda_name)
        hand_traj = np.array([[hand_start], [hand_start], [hand_end], [hand_end]])

        if self.traj_mode == TrajType.GENERATOR:
            panda_traj_pre, times_pre = self.make_generator_panda_traj(qs[:2], start_time)
            panda_traj_post, times_post = self.make_generator_panda_traj(qs[2:], times_pre[-1] + GRASP_TIME)
            panda_traj = panda_traj_pre
            panda_traj.ConcatenateInTime(PiecewisePolynomial.FirstOrderHold(np.array([times_pre[-1], times_post[0]]), qs[1:3].T))
            panda_traj.ConcatenateInTime(panda_traj_post)
            hand_traj = PiecewisePolynomial.FirstOrderHold(np.array([times_pre[0], times_pre[-1] , times_post[0], times_post[-1]]), hand_traj.T)
        else:
            down_time = max(self.jointspace_distance(q_pre, q)/JOINTSPACE_SPEED, 1e-2)
            up_time = max(self.jointspace_distance(q_post, q)/JOINTSPACE_SPEED, 1e-2)
            times = np.array([
                start_time,
                start_time + down_time,
                start_time + down_time + GRASP_TIME,
                start_time + down_time + GRASP_TIME + up_time
            ])
            if self.traj_mode == TrajType.LINEAR:
                panda_traj = PiecewisePolynomial.FirstOrderHold(times, qs.T)
            elif self.traj_mode == TrajType.CUBIC:
                panda_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(times[:2], qs[:2].T, np.zeros((DOF, 1)), np.zeros((DOF, 1)))
                panda_traj.ConcatenateInTime(PiecewisePolynomial.FirstOrderHold(times[1:3], qs[1:3].T))
                panda_traj.ConcatenateInTime(PiecewisePolynomial.CubicWithContinuousSecondDerivatives(times[2:], qs[2:].T, np.zeros((DOF, 1)), np.zeros((DOF, 1))))
            hand_traj = PiecewisePolynomial.FirstOrderHold(times, hand_traj.T)

        self.add_trajs(panda_name, panda_traj, hand_traj)

    def pick(self,panda_name, q_pre, q, q_post):
        """
        Return pick trajectories 
        """
        return self.pick_or_place(
            panda_name,
            q_pre,
            q,
            q_post,
            hand_start = MAX_OPEN,
            hand_end = MAX_CLOSE
        )

    def place(self, panda_name, q_pre, q, q_post):
        """
        Return place trajectories
        """
        return self.pick_or_place(
            panda_name,
            q_pre,
            q,
            q_post,
            hand_start = MAX_CLOSE,
            hand_end = MAX_OPEN
        )

    @staticmethod
    def numpy_conf_to_str(q):
        res = ""
        for i in range(len(q)):
            res += str(q[i])
            if i < len(q) - 1:
                res += ", "
            else:
                res += "\n"
        return res
        

    def write_conf_file(self, plan, action_map, save_path):
        """
        Get a conf.txt we can execute on the panda hardware 
        """ 

        f = open(save_path, "w")
        for i, action in enumerate(plan):
            if action.name not in action_map:
                continue
            func = action_map[action.name]["function"]
            arg_map = action_map[action.name]["argument_indices"]
            args = [action.args[i] for i in arg_map]
            if func == PlanToTrajectory.pick:
                f.write(self.numpy_conf_to_str(args[-2]))
                f.write("grasp\n")
                if i == len(plan) - 1:
                    f.write(self.numpy_conf_to_str(args[-1]))
            elif func == PlanToTrajectory.place:
                f.write(self.numpy_conf_to_str(args[-2]))
                f.write("release\n")
                if i == len(plan) - 1:
                    f.write(self.numpy_conf_to_str(args[-1]))
            else:
                traj = args[-1]
                for q in traj:
                    f.write(self.numpy_conf_to_str(q))
        f.close()

    def make_trajectory(self, plan, start_time, action_map):
        #TODO(agro): fix docstring
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
                ith argument to <function>. If a non-integer is 
                included in the list, it is taken as an input 
                to that argument
        Returns:
            None, modifies the directory
        """
        self.curr_time = start_time
        for action in plan:
            if action.name not in action_map:
                continue
            func = action_map[action.name]["function"]
            arg_map = action_map[action.name]["argument_indices"]

            panda_name = action_map[action.name]["arm_name"]
            if isinstance(panda_name,int):
                panda_name = action.args[panda_name]

            args = [self, panda_name] + [action.args[i] for i in arg_map]
            start_time = self.curr_time
            func(*args)
            self.curr_time = self.get_curr_time(panda_name) 
            for name in self.station.panda_infos:
                if panda_name != name:
                    self.stay_still(name, start_time, self.curr_time)