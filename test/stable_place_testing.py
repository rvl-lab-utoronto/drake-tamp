#!/usr/bin/env python3

import argparse
import time
import numpy as np
import pydrake.all
from pydrake.all import (
    DiagramBuilder, 
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    ConnectMeshcatVisualizer,
)
import kitchen_streamsv2
from panda_station import *


def find_place_gen(station, station_context, shape_info, surface):
    while True:
        yield kitchen_streamsv2.find_place(
            station, station_context, shape_info, surface
        )

def find_ik_gen(station, station_context, shape_info, X_HI):
    q0 = kitchen_streamsv2.find_ik_with_handpose(
        station, station_context, shape_info, X_HI, relax = True
    )[0]
    res =  kitchen_streamsv2.find_ik_with_handpose(
        station, station_context, shape_info, X_HI, q_initial = q0
    )
    print(res[1])
    return res


def make_and_init_station(zmq_url, prob):
    """
    Make the simulation, and let it run for 0.2 s to let all the objects
    settle into their stating positions
    """
    builder = pydrake.systems.framework.DiagramBuilder()
    problem_info = ProblemInfo(prob)
    stations = problem_info.make_all_stations()
    station = stations["move_free"]
    builder.AddSystem(station)
    scene_graph = station.get_scene_graph()

    meshcat = None
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

    if meshcat is not None:
        meshcat.start_recording()
    return stations, diagram, meshcat, problem_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use --url to specify the zmq_url for a meshcat server"
    )
    parser.add_argument("-u", "--url", nargs="?", default=None)
    args = parser.parse_args()
    problem = "/home/agrobenj/drake-tamp/experiments/kitchen/problems/kitchen_problem.yaml"
    stations, diagram, meshcat, problem_info = make_and_init_station(args.url, problem)
    station_contexts = {}
    for station in stations:
        station_contexts[station] = stations[station].CreateDefaultContext()

    diagram_context = diagram.CreateDefaultContext()
    diagram.Publish(diagram_context)
    station_contexts["move_free"] = diagram.GetSubsystemContext(
        stations["move_free"], diagram_context
    )

    item = "cabbage1"
    region = "tray"
    target_object_info = stations["move_free"].object_infos[region][0]
    holding_object_info = stations["move_free"].object_infos[item][0]
    shape_info = update_placeable_shapes(holding_object_info)[0]
    surface = update_surfaces(
        target_object_info, "base_link", stations["move_free"], station_contexts["move_free"]
    )[0]
    flag = True
    for X_WI in find_place_gen(stations["move_free"], station_contexts["move_free"], shape_info, surface):
        print(f"place: {X_WI}")
        update_station(
            stations["move_free"],
            station_contexts["move_free"],
            [("atpose", item, X_WI)],
            set_others_to_inf= True
        ) 
        X_HI = kitchen_streamsv2.find_grasp(shape_info)
        print(f"grasp {X_HI}")
        q= find_ik_gen(
            stations["move_free"],
            station_contexts["move_free"],
            holding_object_info,
            X_HI
        )[0]
        plant, plant_context = kitchen_streamsv2.get_plant_and_context(
            stations["move_free"], station_contexts["move_free"]
        )
        plant.SetPositions(plant_context, q)
        print(q)
        diagram.Publish(diagram_context)
        inp = input()
        if inp == "a":
            break
                

    meshcat.stop_recording()
    meshcat.publish_recording()
    input("Press ENTER to stop")
