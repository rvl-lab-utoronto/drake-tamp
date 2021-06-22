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
import blocks_world_streams
from panda_station import *
Q_NOMINAL = np.array([0.0, 0.55, 0.0, -1.45, 0.0, 1.58, 0.0])


def find_place_gen(station, station_context, shape_info, surface):
    while True:
        yield blocks_world_streams.find_table_place(
            station, station_context, shape_info, surface
        )

def find_ik_gen(station, station_context, object_info, X_HI, panda_info):
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    X_WI = plant.CalcRelativeTransform(
        plant_context, plant.world_frame(), object_info.welded_to_frame
    )
    p_WI = X_WI.translation()
    p_WB = panda_info.X_WB.translation()
    rpy = RollPitchYaw(panda_info.X_WB.rotation())
    print(rpy.yaw_angle())
    dy = p_WI[1] - p_WB[1]
    dx = p_WI[0] - p_WB[0]
    print(dy, dx)
    theta = np.arctan2(dy,dx) - rpy.yaw_angle()
    q_initial = Q_NOMINAL[:]
    q_initial[0] = theta
    print(f"theta {theta}")
    return blocks_world_streams.find_ik_with_relaxed(
        station,
        station_context,
        object_info,
        X_HI,
        panda_info,
        q_initial = q_initial
    )


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
    problem = "/home/agrobenj/drake-tamp/experiments/blocks_world/problems/blocks_world_problem.yaml"
    stations, diagram, meshcat, problem_info = make_and_init_station(args.url, problem)
    station_contexts = {}
    for station in stations:
        if isinstance(stations[station], dict):
            station_contexts[station] = {}
            for name in stations[station]:
                station_contexts[station][name] = stations[station][name].CreateDefaultContext()
        else:
            station_contexts[station] = stations[station].CreateDefaultContext()

    diagram_context = diagram.CreateDefaultContext()
    diagram.Publish(diagram_context)
    station_contexts["move_free"] = diagram.GetSubsystemContext(
        stations["move_free"], diagram_context
    )

    item = "blue_block"
    region = "middle_table"
    holding_object_info = stations["move_free"].object_infos[item][0]
    shape_info = update_placeable_shapes(holding_object_info)[0]
    target_object_info = stations["move_free"].object_infos[region][0]
    surface = update_surfaces(
        target_object_info, "base_link", stations["move_free"], station_contexts["move_free"]
    )[0]
    flag = True
    fail = 0
    tot = 0
    for X_WI in find_place_gen(stations["move_free"], station_contexts["move_free"], shape_info, surface):
        print(f"place: {X_WI}")
        update_station(
            stations["move_free"],
            station_contexts["move_free"],
            [("atpose", item, X_WI)],
            set_others_to_inf= True
        ) 
        X_HI = blocks_world_streams.find_grasp(shape_info)
        print(f"grasp {X_HI}")
        panda_info = stations["move_free"].panda_infos["right_panda"]
        q,cost= find_ik_gen(
            stations["move_free"],
            station_contexts["move_free"],
            holding_object_info,
            X_HI,
            panda_info = panda_info
        )
        plant, plant_context = blocks_world_streams.get_plant_and_context(
            stations["move_free"], station_contexts["move_free"]
        )
        plant.SetPositions(plant_context, panda_info.panda, q)
        print(q, cost)
        if not np.isfinite(cost):
            fail +=1 
        tot +=1
        print(f"success rate: {1 - (fail/tot)}")
        diagram.Publish(diagram_context)
        inp = input()
        if inp == "a":
            break
                

    meshcat.stop_recording()
    meshcat.publish_recording()
    input("Press ENTER to stop")
