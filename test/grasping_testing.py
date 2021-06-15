#!/usr/bin/env python3
"""
This script is used to test the grasping streams
"""
import argparse
import time
import numpy as np
import pydrake.all
from pydrake.all import (
    DiagramBuilder, 
    RigidTransform,
    RollPitchYaw,
    ConnectMeshcatVisualizer,
)
from panda_station import *

# directive to setup world
DIRECTIVE = "directives/three_tables.yaml"
# model file the hand is holding
MODEL_PATH = "models/manipulands/sdf/foam_brick.sdf"
# the name of hte main body of the model
MAIN_BODY_NAME = "base_link"
# the relative transform between the hand and the object it is holding
X_WO = RigidTransform(
    RollPitchYaw(0, 0, np.random.uniform(-np.pi, np.pi)),
    np.random.uniform([0.3, -0.3, 0.05], [0.6, 0.3, 0.1])
)
# the target placement object
NAMES_AND_LINKS = [
    (name, "base_link") for name in parse_tables(find_resource(DIRECTIVE))
] 

def make_station(zmq_url):
    """
    Make the panda station used to find a placement pose,
    connect a visualizer 
    """
    # create panda station
    builder = DiagramBuilder()
    station = PandaStation(name = "placement_station")
    station.setup_from_file(DIRECTIVE, names_and_links=NAMES_AND_LINKS)
    station.add_panda_with_hand(weld_fingers=True)

    # add model welded to hand with relative pose X_HO
    plant = station.get_multibody_plant()
    station.add_model_from_file(
        find_resource(MODEL_PATH),
        X_WO,
        P = plant.world_frame(),
        main_body_name = MAIN_BODY_NAME,
        welded = True,
        name = "object"
    )
    station.finalize()
    builder.AddSystem(station)

    # connect visualizer
    scene_graph = station.get_scene_graph()
    v = None
    if zmq_url is not None:
        v = ConnectMeshcatVisualizer(
            builder,
            scene_graph,
            output_port=station.GetOutputPort("query_object"),
            delete_prefix_on_load=True,
            zmq_url=zmq_url,
        )
        v.load()
    else:
        print("No meshcat server url provided, running without gui")

    diagram = builder.Build()
    return diagram, station

def grasp_gen(station, station_context):
    object_info = station.object_infos["object"][0]
    shape_infos = update_graspable_shapes(object_info)
    iter = 1
    while True:
        print(f"{Colors.GREEN}Finding grasp{Colors.RESET}")
        start_time = time.time()
        grasp_q, cost = None, np.inf
        #how many times will we try before saying it can't be done
        max_tries, tries = 5, 0
        if iter == 1:
            q_initial = Q_NOMINAL
            q_nominal = Q_NOMINAL
        else:
            q_nominal = random_normal_q(station, Q_NOMINAL)
            q_initial = random_normal_q(station, Q_NOMINAL)
        while tries < max_tries:
            tries += 1
            print(f"{Colors.BOLD}grasp tries: {tries}{Colors.RESET}")
            grasp_q, cost = best_grasp_for_shapes(
                station,
                station_context,
                shape_infos,
                initial_guess = q_initial,
                q_nominal = q_nominal
            )
            if np.isfinite(cost):
                break
            q_initial = random_normal_q(station, Q_NOMINAL)
            q_nominal = random_normal_q(station, Q_NOMINAL)
        if not np.isfinite(cost):
            print(f"{Colors.RED}Ending grasp stream for{Colors.RESET}")
            return
        X_HO = q_to_X_HO(
                grasp_q, 
                object_info.main_body_info,
                station,
                station_context
            )
        print(f"X_HO:\n{X_HO}")
        print(
            f"{Colors.REVERSE}Yielding grasp in {(time.time() - start_time):.4f} s{Colors.RESET}"
        )
        iter += 1
        yield grasp_q, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use --url to specify the zmq_url for a meshcat server"
    )
    parser.add_argument("-u", "--url", nargs="?", default=None)
    args = parser.parse_args()
    diagram, station = make_station(args.url)
    diagram_context = diagram.CreateDefaultContext()
    station_context = station.GetMyContextFromRoot(diagram_context)
    plant = station.get_multibody_plant()
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    for grasp_q, cost in grasp_gen(station, station_context):
        # set positions 
        plant.SetPositions(plant_context, station.get_panda(), grasp_q)
        diagram.Publish(diagram_context)
        input("Press ENTER to see next grasp pose")
        start_time = time.time()
