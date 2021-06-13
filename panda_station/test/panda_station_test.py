import unittest
import os
import numpy as np
from pydrake.all import (
    Parser,
    DiagramBuilder,
    ModelInstanceIndex,
    RigidTransform,
    RotationMatrix,
    MultibodyPlant,
    SceneGraph,
    FindResourceOrThrow,
    Simulator,
)
from panda_station import *


class TestPandaStation(unittest.TestCase):
    def test_PandaStation(self):
        builder = DiagramBuilder()
        station = builder.AddSystem(PandaStation())
        station.setup_from_file("directives/three_tables.yaml")
        q_initial = np.zeros(7)
        station.add_panda_with_hand(q_initial=q_initial)
        brick = station.add_model_from_file(
            find_resource("models/manipulands/sdf/foam_brick.sdf"),
            RigidTransform(RotationMatrix(), [0.6, 0, 0.2]),
            main_body_name="base_link",
            name="brick",
        )
        station.finalize()
        diagram = builder.Build()
        plant, scene_graph = station.get_plant_and_scene_graph()
        self.assertIsInstance(plant, MultibodyPlant)
        self.assertIsInstance(scene_graph, SceneGraph)
        panda = station.get_panda()
        hand = station.get_hand()
        self.assertIsInstance(panda, ModelInstanceIndex)
        self.assertIsInstance(hand, ModelInstanceIndex)
        diagram_context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(diagram_context)
        q_init = plant.GetPositions(plant_context, panda)
        self.assertTrue(np.all(q_init == q_initial))

        builder = DiagramBuilder()
        station = builder.AddSystem(PandaStation())
        station.setup_from_file("directives/three_tables.yaml")
        station.add_panda_with_hand()
        station.add_model_from_file(
            find_resource("models/manipulands/sdf/meat_can.sdf"),
            RigidTransform(RotationMatrix(), [0.6, 0, 0.2]),
        )
        station.finalize()
        diagram = builder.Build()
        plant, scene_graph = station.get_plant_and_scene_graph()
        simulator = Simulator(diagram)
        simulator_context = simulator.get_context()
        diagram_context = diagram.GetMyContextFromRoot(simulator_context)
        station_context = station.GetMyContextFromRoot(simulator_context)
        plant_context = plant.GetMyContextFromRoot(simulator_context)
        des_q = np.array([np.pi / 4, 0.1, 0, -1, 0, 1.5, 0])
        station.GetInputPort("panda_position").FixValue(station_context, des_q)

        station.GetInputPort("hand_position").FixValue(station_context, [0.08])
        simulator.AdvanceTo(1.0)
        panda = station.get_panda()
        hand = station.get_hand()
        test_q = plant.GetPositions(plant_context, panda)
        self.assertTrue(np.all(np.isclose(des_q, test_q, atol=np.pi * 0.01)))


if __name__ == "__main__":
    unittest.main()
