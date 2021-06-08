import unittest
import os
import numpy as np
from pydrake.all import (
    Parser,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    ModelInstanceIndex,
    RigidTransform,
    Simulator,
)
from panda_station import *


class TestPandaHandPositionController(unittest.TestCase):
    def test_PandaHandPositionController(self):
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.001)
        parser = Parser(plant)
        hand = add_panda_hand(plant)
        plant.WeldFrames(
            plant.world_frame(),
            plant.GetFrameByName("panda_hand", hand),
            RigidTransform(),
        )
        plant.Finalize()
        hand_controller = builder.AddSystem(PandaHandPositionController())
        builder.Connect(
            hand_controller.GetOutputPort("generalized_force"),
            plant.get_actuation_input_port(hand),
        )
        builder.Connect(
            plant.get_state_output_port(hand), hand_controller.GetInputPort("state")
        )
        builder.ExportInput(
            hand_controller.GetInputPort("desired_position"), "hand_position"
        )
        builder.ExportInput(
            hand_controller.GetInputPort("force_limit"), "hand_force_limit"
        )
        hand_mbp_state_to_hand_state = builder.AddSystem(
            make_multibody_state_to_panda_hand_state_system()
        )
        builder.Connect(
            plant.get_state_output_port(hand),
            hand_mbp_state_to_hand_state.get_input_port(),
        )
        builder.ExportOutput(
            hand_mbp_state_to_hand_state.get_output_port(), "hand_state_measured"
        )
        builder.ExportOutput(
            hand_controller.GetOutputPort("grip_force"), "hand_force_measured"
        )
        diagram = builder.Build()
        simulator = Simulator(diagram)
        simulator_context = simulator.get_context()
        diagram_context = diagram.GetMyContextFromRoot(simulator_context)
        # opening
        diagram.GetInputPort("hand_position").FixValue(diagram_context, [0.08])
        simulator.AdvanceTo(1.0)
        plant_context = plant.GetMyContextFromRoot(simulator_context)
        test_q = plant.GetPositions(plant_context, hand)
        correct_q = np.ones(2) * 0.04
        self.assertTrue(np.all(np.isclose(correct_q, test_q)))
        # closing
        diagram.GetInputPort("hand_position").FixValue(diagram_context, [0])
        simulator.AdvanceTo(2.0)
        plant_context = plant.GetMyContextFromRoot(simulator_context)
        test_q = plant.GetPositions(plant_context, hand)
        correct_q = np.zeros(2)
        self.assertTrue(np.all(np.isclose(correct_q, test_q)))
        # can we reach force limit
        diagram.GetInputPort("hand_position").FixValue(diagram_context, [-1])
        simulator.AdvanceTo(3.0)
        plant_context = plant.GetMyContextFromRoot(simulator_context)
        force = diagram.GetOutputPort("hand_force_measured").Eval(diagram_context)[0]
        self.assertTrue(np.isclose(force, 40))


if __name__ == "__main__":
    unittest.main()
