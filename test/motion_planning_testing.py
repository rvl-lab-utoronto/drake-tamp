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
from panda_station import *


def plan_motion(station, station_context, start, end):
    rot = RotationMatrix()
    rot.set(
        np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ]
        )
    )
    X_HO = RigidTransform(rot, [0.005, 0.005, 0.1503])
    mock_fluent = [("atgrasppose", "cabbage1", X_HO)]
    update_station(station, station_context, mock_fluent)
    traj = find_traj(
        station,
        station_context,
        start,
        end,
    )
    return traj

def make_and_init_station(zmq_url, prob):
    """
    Make the simulation, and let it run for 0.2 s to let all the objects
    settle into their stating positions
    """
    builder = pydrake.systems.framework.DiagramBuilder()
    problem_info = ProblemInfo(prob)
    station = problem_info.make_holding_station(
        name = "cabbage1"
    )
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
    return station, diagram, meshcat, problem_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use --url to specify the zmq_url for a meshcat server"
    )
    parser.add_argument("-u", "--url", nargs="?", default=None)
    args = parser.parse_args()
    problem = "/home/agrobenj/drake-tamp/experiments/kitchen/problems/kitchen_problem.yaml"
    station, diagram, meshcat, problem_info = make_and_init_station(args.url, problem)

    diagram_context = diagram.CreateDefaultContext()
    diagram.Publish(diagram_context)

    station_context = station.GetMyContextFromRoot(diagram_context)
    input("Press ENTER to proceed")

    traj = [
        "-0.4885  0.3839 -0.3469 -1.2419  0.0968  1.5989 -0.0112",
        "-0.903   0.0882 -0.8717 -1.6243  0.0532  1.6552  0.0024"
    ]
    traj = [np.array(list(map(float,a.split()))) for a in traj]
    
    """
    s = "0.0513 -0.1461  0.0555 -2.0601  0.0092  1.9131  0.0718".split()
    s= list(map(float, s))
    g = "-0.3673  0.3189 -0.2936 -1.4508  0.0783  1.7238 -0.0055".split()
    g= list(map(float, g))
    print(s, g)
    start = np.array(s)
    goal = np.array(g) 
    """
    flag = True
    while flag:
        traj = plan_motion(station, station_context, traj[0], traj[-1])
        print(traj)
        plant, scene_graph = station.get_plant_and_scene_graph()
        plant_context = station.GetSubsystemContext(plant, station_context)
        scene_graph_context = station.GetSubsystemContext(scene_graph, station_context)
        query_output_port = scene_graph.GetOutputPort("query")
        input("ENTER to see next traj")
        for i in range(len(traj)-1):
            q_now = traj[i]
            q_next = traj[i+1]
            for d in np.linspace(0,1,100):
                q = (q_next - q_now)*d + q_now
                plant.SetPositions(plant_context, q)
                query_object = query_output_port.Eval(scene_graph_context)
                if (query_object.HasCollisions()):
                    flag = False
                diagram.Publish(diagram_context)
                

    meshcat.stop_recording()
    meshcat.publish_recording()
    input("Press ENTER to stop")
