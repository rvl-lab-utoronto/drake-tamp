import unittest
import os
from pydrake.all import Parser, AddMultibodyPlantSceneGraph, DiagramBuilder
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


if __name__ == "__main__":
    unittest.main()
