import unittest
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


class TestPlanningUtils(unittest.TestCase):
    def setUp(self):
        builder = DiagramBuilder()
        self.station = builder.AddSystem(PandaStation())
        self.X_WO1 = RigidTransform(RotationMatrix(), [1, 1, 1])
        self.X_WO2 = RigidTransform(RotationMatrix.MakeZRotation(np.pi / 2), [-1, 0, 0])
        self.station.setup_from_file("test/test_files/three_tables.yaml")
        self.q0 = np.ones(7)
        self.station.add_panda_with_hand(q_initial=self.q0)
        self.station.add_model_from_file(
            find_resource("models/manipulands/sdf/foam_brick.sdf"),
            self.X_WO1,
            name="foam_brick",
        )
        self.station.add_model_from_file(
            find_resource("models/manipulands/sdf/soup_can.sdf"),
            self.X_WO2,
            name="soup_can",
        )
        self.station.finalize()
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()
        self.station_context = self.station.GetMyContextFromRoot(diagram_context)

    def test_parse_start_poses(self):
        res = parse_start_poses(self.station, self.station_context)
        self.assertIn("foam_brick", list(res.keys()))
        self.assertIn("soup_can", list(res.keys()))
        self.assertTrue(
            np.all(res["foam_brick"].translation() == self.X_WO1.translation())
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    res["foam_brick"].rotation().matrix(),
                    self.X_WO1.rotation().matrix(),
                )
            )
        )
        self.assertTrue(
            np.all(res["soup_can"].translation() == self.X_WO2.translation())
        )
        self.assertTrue(
            np.all(
                np.isclose(
                    res["soup_can"].rotation().matrix(), self.X_WO2.rotation().matrix()
                )
            )
        )

    def test_parse_tables(self):
        res = parse_tables(self.station.directive)
        self.assertTrue(np.all(res == ["table", "table_square", "table_round"]))


if __name__ == "__main__":
    unittest.main()
