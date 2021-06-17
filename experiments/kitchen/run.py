#!/usr/bin/env python3
"""
The module for running the kitchen TAMP problem.
See `problem 5` at this link for details:
http://tampbenchmark.aass.oru.se/index.php?title=Problems
"""
import time
import os
import copy
from re import I
import sys
import argparse
import numpy as np
import random
import pydrake.all
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import str_from_object
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.algorithms.meta import solve
#TODO(agro): see which of these are necessary
from panda_station import (
    ProblemInfo,
    parse_start_poses,
    parse_config,
    parse_tables,
    update_station,
    TrajectoryDirector,
    find_traj,
    q_to_X_HO,
    best_place_shapes_surfaces,
    q_to_X_PF,
    plan_to_trajectory,
    Colors,
    RigidTransformWrapper,
    update_graspable_shapes,
    update_placeable_shapes,
    update_surfaces,
    best_grasp_for_shapes,
    pre_and_post_grasps,
    plan_to_trajectory,
    Q_NOMINAL,
    random_normal_q
)

np.set_printoptions(precision=4, suppress=True)
np.random.seed(seed = 0)
random.seed(0)
ARRAY = tuple
SIM_INIT_TIME = 0.2

domain_pddl = """

"""

stream_pddl = """

"""


def construct_problem_from_sim(simulator, stations):
    pass



def make_and_init_simulation(zmq_url, prob):
    """
    Make the simulation, and let it run for 0.2 s to let all the objects
    settle into their stating positions
    """
    builder = pydrake.systems.framework.DiagramBuilder()
    problem_info = ProblemInfo(prob)
    stations = problem_info.make_all_stations()
    station = stations["main"]
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

    director = builder.AddSystem(TrajectoryDirector())
    builder.Connect(
        station.GetOutputPort("panda_position_measured"),
        director.GetInputPort("panda_position"),
    )
    builder.Connect(
        station.GetOutputPort("hand_state_measured"),
        director.GetInputPort("hand_state"),
    )
    builder.Connect(
        director.GetOutputPort("panda_position_command"),
        station.GetInputPort("panda_position"),
    )
    builder.Connect(
        director.GetOutputPort("hand_position_command"),
        station.GetInputPort("hand_position"),
    )

    diagram = builder.Build()
    simulator = pydrake.systems.analysis.Simulator(diagram)

    # let objects fall for a bit in the simulation
    if meshcat is not None:
        meshcat.start_recording()
    simulator.AdvanceTo(SIM_INIT_TIME)
    return simulator, stations, director, meshcat


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Use --url to specify the zmq_url for a meshcat server\nuse --problem to specify a .yaml problem file"
    )
    parser.add_argument("-u", "--url", nargs="?", default=None)
    parser.add_argument(
        "-p", "--problem", nargs="?", default="problems/kitchen_problem.yaml"
    )
    args = parser.parse_args()
    sim, station_dict, traj_director, meshcat_vis = make_and_init_simulation(
        args.url, args.problem
    )
    problem = construct_problem_from_sim(sim, station_dict)

    print("Initial:", str_from_object(problem.init))
    print("Goal:", str_from_object(problem.goal))
    for algorithm in [
        "adaptive",
    ]:
        solution = solve(problem, algorithm=algorithm, verbose=True)
        print(f"\n\n{algorithm} solution:")
        print_solution(solution)

        plan, _, _ = solution
        if plan is None:
            print(f"{Colors.RED}No solution found, exiting{Colors.RESET}")
            sys.exit(0)
        """
        for action in plan:
            print(action.name)
            print(action.args)
        """

        """
        plan_to_trajectory(plan, traj_director, SIM_INIT_TIME)

        sim.AdvanceTo(traj_director.get_end_time())
        if meshcat_vis is not None:
            meshcat_vis.stop_recording()
            meshcat_vis.publish_recording()

        save = input(
            (
                "Type ENTER to exit without saving.\n "
                "To save the video to the file\n"
                "media/<filename.html>, input <filename>\n"
            )
        )
        if save != "":
            if not os.path.isdir("media"):
                os.mkdir("media")
            file = open("media/" + save + ".html", "w")
            file.write(meshcat_vis.vis.static_html())
            file.close()
        """
