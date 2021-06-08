import unittest
import os
import numpy as np
from pydrake.all import (
    Parser,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    ModelInstanceIndex,
    RigidTransform,
    RotationMatrix,
)
from panda_station import *


class TestConstructionUtils(unittest.TestCase):
    def test_find_resource(self):
        test_path = find_resource("")
        correct_path = os.path.dirname(os.path.abspath("../" + __file__)) + "/"
        self.assertEqual(test_path, correct_path)
        test_path = find_resource("models/modified_panda_hand/panda_hand.sdf")
        correct_path = (
            os.path.dirname(os.path.abspath("../" + __file__))
            + "/models/modified_panda_hand/panda_hand.sdf"
        )
        self.assertEqual(test_path, correct_path)

    def test_add_package_paths(self):
        builder = DiagramBuilder()
        plant, _ = AddMultibodyPlantSceneGraph(builder, 0.0)
        parser = Parser(plant)
        add_package_paths(parser)
        packages = ["manipulation_station", "modified_panda_hand", "tables"]
        for p in packages:
            self.assertTrue(parser.package_map().Contains(p))

    def test_add_panda(self):
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
        parser = Parser(plant)
        q_init = np.zeros(7)
        Xinit_WB = RigidTransform(RotationMatrix(), [1, 1, 0])
        panda = add_panda(plant, q_initial=q_init, X_WB=Xinit_WB)
        plant.Finalize()
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(diagram_context)
        q_test = plant.GetPositions(plant_context, panda)
        base_body = plant.GetBodyByName("panda_link0")
        test_bodies = plant.GetBodiesWeldedTo(plant.world_body())
        Xtest_WB = base_body.EvalPoseInWorld(plant_context)
        self.assertIsInstance(panda, ModelInstanceIndex)
        self.assertTrue(base_body in test_bodies)
        self.assertTrue(np.all(q_init == q_test))
        self.assertTrue(np.all(Xtest_WB.translation() == Xinit_WB.translation()))
        self.assertTrue(
            np.all(Xtest_WB.rotation().matrix() == Xinit_WB.rotation().matrix())
        )

    def test_add_panda_hand(self):
        # welded fingers
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
        parser = Parser(plant)
        q_init = np.zeros(7)
        Xinit_WB = RigidTransform(RotationMatrix(), [1, 1, 0])
        panda = add_panda(plant, q_initial=q_init, X_WB=Xinit_WB)
        hand = add_panda_hand(
            plant, panda_model_instance_index=panda, weld_fingers=True
        )
        plant.Finalize()
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(diagram_context)
        self.assertEqual(plant.num_positions(), 7)
        self.assertIsInstance(hand, ModelInstanceIndex)
        # free fingers
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
        parser = Parser(plant)
        q_init = np.zeros(7)
        panda = add_panda(plant)
        hand = add_panda_hand(plant, panda_model_instance_index=panda)
        plant.Finalize()
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(diagram_context)
        self.assertEqual(plant.num_positions(), 9)
        self.assertIsInstance(hand, ModelInstanceIndex)


if __name__ == "__main__":
    unittest.main()
