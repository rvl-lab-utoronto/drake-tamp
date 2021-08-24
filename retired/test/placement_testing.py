#!/usr/bin/env python3
"""
This script is used to test the placement streams
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

np.set_printoptions(precision=3, suppress=True)
# directive to setup world
DIRECTIVE = "directives/three_tables.yaml"
# model file the hand is holding
MODEL_PATH = "models/manipulands/sdf/foam_brick.sdf"
# the name of hte main body of the model
MAIN_BODY_NAME = "base_link"
# the relative transform between the hand and the object it is holding
X_HO = RigidTransform(
    RollPitchYaw(0, 0, 0),
    [0, 0, 0.08]
)
# the target placement object
TARGET = "table"
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
        X_HO,
        P = plant.GetFrameByName("panda_hand", station.get_hand()),
        main_body_name = MAIN_BODY_NAME,
        welded = True,
        name = "holding_object"
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

def placement_gen(station, station_context):
    holding_object_info = station.object_infos["holding_object"][0]
    W = station.get_multibody_plant().world_frame()
    target_object_info = station.object_infos[TARGET][0]
    shape_infos = update_placeable_shapes(holding_object_info)
    surfaces = update_surfaces(target_object_info, station, station_context)
    initial_guess = Q_NOMINAL
    while True:
        print(f"{Colors.GREEN}Finding place{Colors.RESET}")
        start_time = time.time()
        place_q, cost = None, np.inf
        #how many times will we try before saying it can't be done
        max_tries, tries = 10, 0
        while tries < max_tries:
            tries += 1
            print(f"{Colors.BOLD}place tries: {tries}{Colors.RESET}")
            place_q, cost = best_place_shapes_surfaces(
                station,
                station_context,
                shape_infos,
                surfaces,
                initial_guess = initial_guess
            )
            if np.isfinite(cost):
                break
        if not np.isfinite(cost):
            print(f"{Colors.RED}Ending place stream{Colors.RESET}")
            return
        X_WO = RigidTransformWrapper(
            q_to_X_PF(
                place_q,
                station.get_multibody_plant().world_frame(),
                holding_object_info.main_body_info.get_body_frame(),
                station,
                station_context,
            )
        )
        print(f"X_WO:\n{X_WO}")
        print(
            f"{Colors.REVERSE}Yielding placement in {(time.time() - start_time):.4f} s{Colors.RESET}"
        )
        yield place_q, cost
        initial_guess = place_q


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
    target_names = []
    for tup in NAMES_AND_LINKS:
        target_names.append(tup[0])
    for place_q, cost in placement_gen(station, station_context):
        plant.SetPositions(plant_context, station.get_panda(), place_q)
        diagram.Publish(diagram_context)
        input("Press ENTER to see next placement pose")
        start_time = time.time()
