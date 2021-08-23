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
args = parser.parse_args()
builder = pydrake.systems.framework.DiagramBuilder()
problem_info = ProblemInfo(
    "/home/agrobenj/drake-tamp/experiments/blocks_world/problems/blocks_world_problem.yaml"
)
#problem_info = ProblemInfo(
    #"/home/agrobenj/drake-tamp/experiments/kitchen_no_fluents/problems/kitchen_problem.yaml"
#)

stations = problem_info.make_all_stations()
print(stations)
station = stations["move_free"]
builder.AddSystem(station)
scene_graph = station.get_scene_graph()

meshcat = None
zmq_url = args.url
if zmq_url is not None:
    meshcat = pydrake.systems.meshcat_visualizer.ConnectMeshcatVisualizer(
        builder,
        scene_graph,
        output_port=station.GetOutputPort("query_object"),
        delete_prefix_on_load=True,
        zmq_url=zmq_url,
    )
    meshcat.load()
else:
    print("No meshcat server url provided, running without gui")

diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
diagram.Publish(diagram_context)

plant = station.get_multibody_plant()
plant_context = plant.GetMyContextFromRoot(diagram_context)
print(plant.GetPositions(plant_context))

input("Press Enter")