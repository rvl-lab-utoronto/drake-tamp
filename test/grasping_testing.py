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
    RollPitchYaw(0, 0, 0),
    [0.6, 0, 0.06]
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
    body_infos = list(object_info.get_body_infos().values())
    for body_info in body_infos:
        for shape_info in body_info.get_shape_infos():
            for grasp_q, cost in find_grasp_q(station, station_context, shape_info):
                """
                postgrasp_q, pregrasp_q = grasp_q.copy(), grasp_q.copy()
                grasp_height = 0.07
                while grasp_height > 0 and (
                    np.all(pregrasp_q == grasp_q) or np.all(postgrasp_q == grasp_q)
                ):
                    test_pregrasp_q, pre_cost = backup_on_hand_z(
                        grasp_q, station, station_context, d=grasp_height
                    )
                    test_postgrasp_q, post_cost = backup_on_world_z(
                        grasp_q, station, station_context, d=grasp_height
                    )
                    if np.isfinite(pre_cost) and np.all(pregrasp_q == grasp_q):
                        pregrasp_q = test_pregrasp_q
                        print(
                            f"{Colors.REVERSE}pregrasp distance: {grasp_height}{Colors.RESET}"
                        )
                    if np.isfinite(post_cost) and np.all(postgrasp_q == grasp_q):
                        postgrasp_q = test_postgrasp_q
                        print(
                            f"{Colors.REVERSE}postgrasp distance: {grasp_height}{Colors.RESET}"
                        )
                    grasp_height -= 0.01
                # relative transform from hand to main_body of object_info
                X_HO = q_to_X_HO(
                    grasp_q, object_info.main_body_info, station, station_context
                )
                """
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
    target_names = []
    for tup in NAMES_AND_LINKS:
        target_names.append(tup[0])
    start_time = time.time()
    for place_q, cost in grasp_gen(station, station_context):
        # set positions 
        end_time = time.time()
        print(f"{Colors.BOLD}Evaluation time: {end_time - start_time}{Colors.RESET}")
        print(f"{Colors.BLUE}grasp_q: {place_q}{Colors.RESET}")
        print(f"{Colors.CYAN}cost: {cost}{Colors.RESET}")
        plant.SetPositions(plant_context, station.get_panda(), place_q)
        diagram.Publish(diagram_context)
        """
        print("Available target names:")
        for name in target_names:
            print(name)
        target = input(f"Type a new target name, or press ENTER to use the current target {TARGET}\n")
        if target in target_names:
            TARGET = target
        """
        input("Press ENTER to see next grasp pose")
        start_time = time.time()
