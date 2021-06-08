"""
A system used for TAMP experiments with the Franka Emika Panda arm and the Franka Emika Gripper
"""
import numpy as np
import pydrake.all
from . import construction_utils
from .panda_hand_position_controller import (
    PandaHandPositionController,
    make_multibody_state_to_panda_hand_state_system
)


class ObjectInfo:
    """
    Simple struct for carrying information about objects added to PandaStation
    """

    def __init__(self, path, main_body_index, Xinit_WO, name):
        """
        Construct an ObjectInfo struct

        Args:
            path: the path the the model file [string]
            main_body_index: the pydrake.multibody.tree.BodyIndex of
            the main link of the object
            Xinit_WO: the pydrake.math.RigidTransform represnting the initial
            world pose of the object
            name: the name of the added model
        """
        self.path = path
        self.main_body_index = main_body_index
        self.Xinit_WO = Xinit_WO
        self.name = name


class PandaStation(pydrake.systems.framework.Diagram):
    """
    The PandaStation class
    TODO(agro): add cameras if nessecary
    """

    def __init__(self, time_step=0.001):
        """
        Construct a panda station

        Args:
            time_step: simulation time step [float]
        """
        pydrake.systems.framework.Diagram.__init__(self)
        self.time_step = time_step
        self.builder = pydrake.systems.framework.DiagramBuilder()
        (
            self.plant,
            self.scene_graph,
        ) = pydrake.multibody.plant.AddMultibodyPlantSceneGraph(
            self.builder, time_step=self.time_step
        )
        self.object_infos = []
        self.controller_plant = pydrake.multibody.plant.MultibodyPlant(
            time_step=self.time_step
        )
        self.weld_fingers = False  # is the hand welded open in this plant
        self.directive = None  # the directive used to setup the environment
        self.plant.set_name("plant")
        self.set_name("panda_station")
        self.panda = None
        self.hand = None

    def fix_collisions(self):
        """
        fix collisions for the hand and arm in this plant
        by removing collisions between panda_link5<->panda_link7 and
        panda_link7<->panda_hand
        """
        assert self.panda is not None, "No panda has been added"
        assert self.hand is not None, "No panda hand has been added"
        # get geometry indices and create geometry sets
        panda_link5 = self.plant.GetFrameByName("panda_link5", self.panda).body()
        panda_link5 = pydrake.geometry.GeometrySet(
            self.plant.GetCollisionGeometriesForBody(panda_link5)
        )
        panda_link7 = self.plant.GetFrameByName("panda_link7", self.panda).body()
        panda_link7 = pydrake.geometry.GeometrySet(
            self.plant.GetCollisionGeometriesForBody(panda_link7)
        )
        panda_hand = self.plant.GetFrameByName("panda_hand", self.hand).body()
        panda_hand = pydrake.geometry.GeometrySet(
            self.plant.GetCollisionGeometriesForBody(panda_hand)
        )
        # exclude collisions
        self.scene_graph.ExcludeCollisionsBetween(panda_link7, panda_hand)
        self.scene_graph.ExcludeCollisionsBetween(panda_link5, panda_link7)

    def add_panda_with_hand(
        self, weld_fingers=False, q_initial=np.array([0.0, 0.1, 0, -1.2, 0, 1.6, 0])
    ):
        """
        Add the panda hand and panda arm (at the world origin) to the station

        Args:
            weld_fingers: [bool] True iff the fingers are welded in the open position
            q_initial: the initial positions to set the panda arm (np.array)
        """
        self.panda = construction_utils.add_panda(self.plant, q_initial=q_initial)
        self.hand = construction_utils.add_panda_hand(
            self.plant, panda_model_instance_index=self.panda, weld_fingers=weld_fingers
        )
        self.weld_fingers = weld_fingers
        self.fix_collisions()

    def setup_from_file(self, filename):
        """
        Setup a station with the path to the directive in `filename`

        Args:
            filename: [string] that path (starting from this directory of this
            file) to the directive. e.g `directives/table_top.yaml`
        """
        self.directive = construction_utils.find_resource(filename)
        parser = pydrake.multibody.parsing.Parser(self.plant)
        construction_utils.add_package_paths(parser)
        pydrake.multibody.parsing.ProcessModelDirectives(
            pydrake.multibody.parsing.LoadModelDirectives(self.directive),
            self.plant,
            parser,
        )

    def add_model_from_file(self, path, Xinit_WO, main_body_name=None, name=None):
        """
        Add a model to this plant from the full path provided in `path`
        at initial world position Xinit_WO

        Args:
            path: [string] full path to model file (eg. from FindResourceOrThrow or
            find_resource)
            Xinit_WO: the initial world pose of the object
            main_body_name: [string] provide the name of the body link to set the
            position of the model, if there is more than one link in the model.
            name: [string] optional name for the model.
        Returns:
            the model instance index of the added model
        """
        parser = pydrake.multibody.parsing.Parser(self.plant)
        if name is None:
            num = str(len(self.object_infos))
            name = "added_model_" + num
        model = parser.AddModelFromFile(path, name)
        indices = self.plant.GetBodyIndices(model)
        assert (len(indices) == 1) or (
            main_body_name is not None
        ), "You must specify the main link name"
        index = indices[0]
        if main_body_name is not None:
            for i in indices:
                test_name = self.plant.get_body(i).name()
                if test_name == main_body_name:
                    index = i
        self.object_infos.append(ObjectInfo(path, index, Xinit_WO, name))
        return model

    def get_multibody_plant(self):
        """
        Returns the multibody plant of this panda station
        """
        return self.plant

    def get_scene_graph(self):
        """
        Returns the scene graph of this panda station
        """
        return self.scene_graph

    def get_plant_and_scene_graph(self):
        """
        Returns the plant and scene graph of this panda station
        """
        return self.plant, self.scene_graph

    def get_panda(self):
        """
        Returns the panda arm ModelInstanceIndex of this station
        """
        return self.panda
        
    def get_hand(self):
        """
        Returns the panda hand ModelInstanceIndex of this station
        """
        return self.hand

    def finalize(self):
        """finalize the panda station"""

        assert self.panda is not None, "No panda added, run add_panda_with_hand"
        assert self.hand is not None, "No panda hand model added"
        self.plant.Finalize()

        for info in self.object_infos:
            body = self.plant.get_body(info.main_body_index)
            self.plant.SetDefaultFreeBodyPose(body, info.Xinit_WO)
        num_panda_positions = self.plant.num_positions(self.panda)

        panda_position = self.builder.AddSystem(
            pydrake.systems.primitives.PassThrough(num_panda_positions)
        )
        self.builder.ExportInput(panda_position.get_input_port(), "panda_position")
        self.builder.ExportOutput(
            panda_position.get_output_port(), "panda_position_command"
        )

        demux = self.builder.AddSystem(
            pydrake.systems.primitives.Demultiplexer(
                2 * num_panda_positions, num_panda_positions
            )
        )
        self.builder.Connect(
            self.plant.get_state_output_port(self.panda), demux.get_input_port()
        )
        self.builder.ExportOutput(demux.get_output_port(0), "panda_position_measured")
        self.builder.ExportOutput(demux.get_output_port(1), "panda_velocity_estimated")
        self.builder.ExportOutput(
            self.plant.get_state_output_port(self.panda), "panda_state_estimated"
        )

        # plant for the panda controller
        controller_panda = construction_utils.add_panda(self.controller_plant)
        # welded so the controller doesn't care about the hand joints
        construction_utils.add_panda_hand(
            self.controller_plant,
            panda_model_instance_index=controller_panda,
            weld_fingers=True,
        )
        self.controller_plant.Finalize()

        panda_controller = self.builder.AddSystem(
            pydrake.systems.controllers.InverseDynamicsController(
                self.controller_plant,
                kp=[100] * num_panda_positions,
                ki=[1] * num_panda_positions,
                kd=[20] * num_panda_positions,
                has_reference_acceleration=False,
            )
        )

        panda_controller.set_name("panda_controller")
        self.builder.Connect(
            self.plant.get_state_output_port(self.panda),
            panda_controller.get_input_port_estimated_state(),
        )

        # feedforward torque
        adder = self.builder.AddSystem(
            pydrake.systems.primitives.Adder(2, num_panda_positions)
        )
        self.builder.Connect(
            panda_controller.get_output_port_control(), adder.get_input_port(0)
        )
        # passthrough to make the feedforward torque optional (default to zero values)
        torque_passthrough = self.builder.AddSystem(
            pydrake.systems.primitives.PassThrough([0] * num_panda_positions)
        )
        self.builder.Connect(
            torque_passthrough.get_output_port(), adder.get_input_port(1)
        )
        self.builder.ExportInput(
            torque_passthrough.get_input_port(), "panda_feedforward_torque"
        )
        self.builder.Connect(
            adder.get_output_port(), self.plant.get_actuation_input_port(self.panda)
        )

        # add a discete derivative to find velocity command based on positional commands
        desired_state_from_position = self.builder.AddSystem(
            pydrake.systems.primitives.StateInterpolatorWithDiscreteDerivative(
                num_panda_positions, self.time_step, suppress_initial_transient=True
            )
        )
        desired_state_from_position.set_name("desired_state_from_position")
        self.builder.Connect(
            desired_state_from_position.get_output_port(),
            panda_controller.get_input_port_desired_state(),
        )
        self.builder.Connect(
            panda_position.get_output_port(),
            desired_state_from_position.get_input_port(),
        )

        if not self.weld_fingers:
            # TODO(agro): make sure this hand controller is accurate
            hand_controller = self.builder.AddSystem(PandaHandPositionController())
            hand_controller.set_name("hand_controller")
            self.builder.Connect(
                hand_controller.GetOutputPort("generalized_force"),
                self.plant.get_actuation_input_port(self.hand),
            )
            self.builder.Connect(
                self.plant.get_state_output_port(self.hand),
                hand_controller.GetInputPort("state"),
            )
            self.builder.ExportInput(
                hand_controller.GetInputPort("desired_position"), "hand_position"
            )
            self.builder.ExportInput(
                hand_controller.GetInputPort("force_limit"), "hand_force_limit"
            )
            hand_mbp_state_to_hand_state = self.builder.AddSystem(
                make_multibody_state_to_panda_hand_state_system()
            )
            self.builder.Connect(
                self.plant.get_state_output_port(self.hand),
                hand_mbp_state_to_hand_state.get_input_port(),
            )
            self.builder.ExportOutput(
                hand_mbp_state_to_hand_state.get_output_port(), "hand_state_measured"
            )
            self.builder.ExportOutput(
                hand_controller.GetOutputPort("grip_force"), "hand_force_measured"
            )

        # TODO(agro): cameras if needed

        # export cheat ports
        self.builder.ExportOutput(
            self.scene_graph.get_query_output_port(), "geometry_query"
        )
        self.builder.ExportOutput(
            self.plant.get_contact_results_output_port(), "contact_results"
        )
        self.builder.ExportOutput(
            self.plant.get_state_output_port(), "plant_continuous_state"
        )

        # for visualization

        self.builder.ExportOutput(
            self.scene_graph.get_query_output_port(), "query_object"
        )

        self.builder.BuildInto(self)
