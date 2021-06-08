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
            FindResourceOrThrow(
                "drake/examples/manipulation_station/models/061_foam_brick.sdf"
            ),
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
        brick_body = plant.GetBodyByName("base_link", brick)


if __name__ == "__main__":
    unittest.main()
