"""
PandaHandPositionController class to control the panda hand with joint positions
reference: https://github.com/RussTedrake/drake/blob/master/manipulation/
schunk_wsg/schunk_wsg_position_controller.cc
"""
import numpy as np
import pydrake.all


def make_multibody_state_to_panda_hand_state_system():
    """
    A system to transform the multibody state to a panda hand state with
    a matrix gain. The multibody plant outputs a state in the form

    [[q1],
     [q2],
     [v1],
     [v2]]

    and we convert this to the form

     [[w],
      [w_dot]]

    where w is the distance between the fingers

    Returns:
        the pydrake.systems.primitives.MatrixGain system
    """
    D = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    return pydrake.systems.primitives.MatrixGain(D)


class PandaHandPdController(pydrake.systems.framework.LeafSystem):
    """
    A controller for the panda hand. That implements two PD controllers,
    one to keep the panda hand centered, and the other implements opening/closing
    finger control
    See documentation here for more details:
    https://drake.mit.edu/doxygen_cxx/
    classdrake_1_1manipulation_1_1schunk__wsg_1_1_schunk_wsg_pd_controller.html

    Input Ports:
        desired_state: specify desired state (distance between fingers and rate
        of change of distance between fingers)
        force_limit: specify force limit of hand
        state: specify two positions and two velocities, for each finger
    Output Ports:
        generalized_force: vector of generalized forces, for each finger
        grip-force: force measurement from diver
    """

    # TODO(agro): currently, these values are based on wsg hand
    def __init__(
        self,
        kp_command=200.0,
        kd_command=5.0,
        kp_constraint=2000.0,
        kd_constraint=5.0,
        default_force_limit=70.0,
    ):
        """
        Constructor for the PandaHandPdController

        Args:
            kp_command: proportional gains [float] for movement control
            kd_command: derivative gains [float] for movement control
            kp_constraint: proportional gains [float] for centering control
            kd_constraint: derivative gains [float] for centering control
            default_force_limit: force limits for hand [float]
        """
        pydrake.systems.framework.LeafSystem.__init__(self)

        self.num_joints = 2

        self.kp_command = kp_command
        self.kd_command = kd_command
        self.kp_constraint = kp_constraint
        self.kd_constraint = kd_constraint
        self.default_force_limit = default_force_limit

        self.desired_state_input_port = self.DeclareVectorInputPort(
            "desired_state", pydrake.systems.framework.BasicVector(2)
        )
        self.force_limit_input_port = self.DeclareVectorInputPort(
            "force_limit", pydrake.systems.framework.BasicVector(1)
        )
        self.state_input_port = self.DeclareVectorInputPort(
            "state", pydrake.systems.framework.BasicVector(2 * self.num_joints)
        )

        self.generalized_force_output_port = self.DeclareVectorOutputPort(
            "generalized_force",
            pydrake.systems.framework.BasicVector(self.num_joints),
            self.calc_generalized_force_output,
        )
        self.grip_force_output_port = self.DeclareVectorOutputPort(
            "grip_force",
            pydrake.systems.framework.BasicVector(1),
            self.calc_grip_force_output,
        )

        self.set_name("panda_hand_controller")

    def get_desired_state_input_port(self):
        """
        Returns desired state input port
        """
        return self.desired_state_input_port

    def get_force_limit_input_port(self):
        """
        Returns force limit input port
        """
        return self.force_limit_input_port

    def get_state_input_port(self):
        """
        Returns state input port
        """
        return self.state_input_port

    def get_generalized_force_output_port(self):
        """
        Returns generalized force output port
        """
        return self.generalized_force_output_port

    def get_grip_force_output_port(self):
        """
        Returns grip force output port
        """
        return self.grip_force_output_port

    def calc_generalized_force(self, context):
        """
        Returns the generalized force (N) for the panda hand given the context
        """
        desired_state = self.desired_state_input_port.Eval(context)
        if self.force_limit_input_port.HasValue(context):
            force_limit = self.force_limit_input_port.Eval(context)[0]
        else:
            force_limit = self.default_force_limit

        if force_limit <= 0:
            raise Exception("Force limit must be greater than 0")

        state = self.state_input_port.Eval(context)

        f0_plus_f1 = -self.kp_constraint * (
            -state[0] + state[1]
        ) - self.kd_constraint * (
            -state[2] + state[3]
        )  # enforces symetrical state

        neg_f0_plus_f1 = self.kp_command * (
            desired_state[0] - state[0] - state[1]
        ) + self.kd_command * (desired_state[1] - state[2] - state[3])

        neg_f0_plus_f1 = np.clip(neg_f0_plus_f1, -force_limit, force_limit)
        return np.array(
            [
                -0.5 * f0_plus_f1 + 0.5 * neg_f0_plus_f1,
                0.5 * f0_plus_f1 + 0.5 * neg_f0_plus_f1,
            ]
        )

    def calc_generalized_force_output(self, context, output_vector):
        """
        Sets the generalized force ouput port
        """
        output_vector.SetFromVector(self.calc_generalized_force(context))

    def calc_grip_force_output(self, context, output_vector):
        """
        sets the grip force output port
        """
        force = self.calc_generalized_force(context)
        output_vector.SetAtIndex(0, np.abs(force[0] + force[1]))


class PandaHandPositionController(pydrake.systems.framework.Diagram):
    """
    A thin wrapper class around PandaHandPdController that is of type
    pydrake.systems.framework.Diagram
    """

    def __init__(
        self,
        time_step=0.05,
        kp_command=200.0,
        kd_command=5.0,
        kp_constraint=2000.0,
        kd_constraint=5.0,
        default_force_limit=40.0,
    ):
        """
        Constructor for the PandaHandPositionController

        Args:
            time_step: the update time step for the diagram
            kp_command: proportional gains [float] for movement control
            kd_command: derivative gains [float] for movement control
            kp_constraint: proportional gains [float] for centering control
            kd_constraint: derivative gains [float] for centering control
            default_force_limit: force limits for hand [float]
        """
        pydrake.systems.framework.Diagram.__init__(self)

        self.time_step = time_step
        self.kp_command = kp_command
        self.kd_command = kd_command
        self.kp_constraint = kp_constraint
        self.kd_constraint = kd_constraint
        self.default_force_limit = default_force_limit

        builder = pydrake.systems.framework.DiagramBuilder()
        self.pd_controller = builder.AddSystem(
            PandaHandPdController(
                kp_command,
                kd_command,
                kp_constraint,
                kd_constraint,
                default_force_limit,
            )
        )

        self.state_interpolator = builder.AddSystem(
            pydrake.systems.primitives.StateInterpolatorWithDiscreteDerivative(
                1, time_step, suppress_initial_transient=True
            )
        )

        builder.Connect(
            self.state_interpolator.get_output_port(),
            self.pd_controller.get_desired_state_input_port(),
        )

        self.desired_position_input_port = builder.ExportInput(
            self.state_interpolator.get_input_port(), "desired_position"
        )
        self.force_limit_input_port = builder.ExportInput(
            self.pd_controller.get_force_limit_input_port(), "force_limit"
        )
        self.state_input_port = builder.ExportInput(
            self.pd_controller.get_state_input_port(), "state"
        )

        self.generalized_force_output_port = builder.ExportOutput(
            self.pd_controller.get_generalized_force_output_port(), "generalized_force"
        )
        self.grip_force_output_port = builder.ExportOutput(
            self.pd_controller.get_grip_force_output_port(), "grip_force"
        )

        builder.BuildInto(self)
