#!/usr/bin/env python3
import argparse
import pydrake.all
from pydrake.all import (
    RigidTransform,
    RotationMatrix
)
from panda_station import *

parser = argparse.ArgumentParser(
    description="Use --url to specify the zmq_url for a meshcat server\nuse --problem to specify a .yaml problem file"
)
parser.add_argument("-u", "--url", nargs="?", default=None)
parser.add_argument("-d", "--directive", nargs="?", default="directives/blocks_world.yaml")
args = parser.parse_args()

size = 0.06

builder = pydrake.systems.framework.DiagramBuilder()
station = builder.AddSystem(PandaStation())
station.setup_from_file(args.directive)
station.add_panda_with_hand(weld_fingers = True)
station.add_model_from_file(
    find_resource("models/blocks_world/sdf/red_cube.sdf"),
    RigidTransform(
        RotationMatrix(),
        [0.5, 0 ,0]
    )
)
station.add_model_from_file(
    find_resource("models/blocks_world/sdf/orange_cube.sdf"),
    RigidTransform(
        RotationMatrix(),
        [0.5, 0 ,size]
    )
)
station.add_model_from_file(
    find_resource("models/blocks_world/sdf/yellow_cube.sdf"),
    RigidTransform(
        RotationMatrix(),
        [0.5, 0 ,2*size]
    )
)
station.add_model_from_file(
    find_resource("models/blocks_world/sdf/green_cube.sdf"),
    RigidTransform(
        RotationMatrix(),
        [0.5, 0 , 3*size]
    )
)
station.add_model_from_file(
    find_resource("models/blocks_world/sdf/cyan_cube.sdf"),
    RigidTransform(
        RotationMatrix(),
        [0.5, 0 , 4*size]
    )
)
station.add_model_from_file(
    find_resource("models/blocks_world/sdf/blue_cube.sdf"),
    RigidTransform(
        RotationMatrix(),
        [0.5, 0 , 5*size]
    )
)
station.finalize()
scene_graph = station.get_scene_graph()

if args.url is not None:
    meshcat = pydrake.systems.meshcat_visualizer.ConnectMeshcatVisualizer(
        builder,
        scene_graph,
        output_port=station.GetOutputPort("query_object"),
        delete_prefix_on_load=True,
        zmq_url=args.url,
    )
    meshcat.load()
else:
    print("No meshcat server url provided, running without gui")
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
diagram.Publish(diagram_context)
input("Press Enter")