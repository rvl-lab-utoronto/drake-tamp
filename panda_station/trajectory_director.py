"""
Contains a simple system for executing provided Drake trajectories
on a PandaStation. If no trajectories are supplied, the panda arm
is told to stay stil
"""
from pydrake.all import LeafSystem, BasicVector


class TrajectoryDirector(LeafSystem):
    """
    TrajectoryDirector Class
    """

    def __init__(self):
        """
        Construct a TrajectoryDirector class
        """
        LeafSystem.__init__(self)
        self.nq = 7
        self.panda_position_input_port = self.DeclareVectorInputPort(
            "panda_position", BasicVector(self.nq)
        )
        self.hand_state_input_port = self.DeclareVectorInputPort(
            "hand_state", BasicVector(2)
        )
        self.panda_position_command_output_port = self.DeclareVectorOutputPort(
            "panda_position_command", BasicVector(self.nq), self.panda_output
        )
        self.hand_position_command_output_port = self.DeclareVectorOutputPort(
            "hand_position_command", BasicVector(1), self.hand_output
        )
        self.panda_traj = None
        self.hand_traj = None

    def get_end_time(self):
        """
        Return when this director has no more trajectories
        """
        p_time = 0
        if self.panda_traj is not None:
            p_time = self.panda_traj.end_time()
        h_time = 0
        if self.hand_traj is not None:
            h_time = self.hand_traj.end_time()
        return max(p_time, h_time)

    def add_panda_traj(self, panda_traj):
        """
        Adds a trajectory (in time) to the existing panda trajectory

        Args:
            panda_traj: PiecewisePolynomial trajectory with
            self.panda_traj.end_time() == panda_traj.start_time()
        """
        if self.panda_traj is None:
            self.panda_traj = panda_traj
            return
        self.panda_traj.ConcatenateInTime(panda_traj)

    def add_hand_traj(self, hand_traj):
        """
        Adds a trajectory (in time) to the existing hand trajectory

        Args:
            hand_traj: PiecewisePolynomial trajectory with
            self.hand_traj.end_time() == hand_traj.start_time()
        """
        if self.hand_traj is None:
            self.hand_traj = hand_traj
            return
        self.hand_traj.ConcatenateInTime(hand_traj)

    def panda_output(self, context, output):
        """
        Calculates the output trajectory for the panda arm
        """
        time = context.get_time()
        if (self.panda_traj is None) or (time > self.panda_traj.end_time()):
            panda_q = self.EvalVectorInput(
                context, self.panda_position_input_port.get_index()
            ).get_value()
            output.set_value(panda_q)
            return
        output.set_value(self.panda_traj.value(time).flatten())

    def hand_output(self, context, output):
        """
        Calculates the output trajectory for the hand
        """
        time = context.get_time()
        if (self.hand_traj is None) or (time > self.hand_traj.end_time()):
            hand_q = self.EvalVectorInput(
                context, self.hand_state_input_port.get_index()
            ).get_value()
            output.set_value([hand_q[0]])
            return
        output.set_value(self.hand_traj.value(time).flatten())
