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
    pre_and_post_grasps,
    plan_to_trajectory,
    Q_NOMINAL,
    random_normal_q
)
import kitchen_streamsv2

np.set_printoptions(precision=4, suppress=True)
np.random.seed(seed = 0)
random.seed(0)
ARRAY = tuple
SIM_INIT_TIME = 0.2
GRASP_DIST = 0.04

domain_pddl = open("domain.pddl", "r").read()
stream_pddl = open("stream.pddl", "r").read()

def construct_problem_from_sim(simulator, stations, problem_info):
    """
    Construct pddlstream problem from simulator
    """
    init = []
    main_station = stations["main"]
    simulator_context = simulator.get_context()
    main_station_context = main_station.GetMyContextFromRoot(simulator_context)
    station_contexts = {}
    for name in stations:
        station_contexts[name] = stations[name].CreateDefaultContext()
    # start poses of all manipulands
    start_poses = parse_start_poses(main_station, main_station_context)
    for item, pose in start_poses.items():
        X_wrapper = RigidTransformWrapper(pose, name = f"X_W{item}")
        init += [
            ("item", item),
            ("worldpose", item, X_wrapper),
            ("atpose", item, X_wrapper),
        ]

    arms = parse_config(main_station, main_station_context)
    for arm, conf in arms.items():
        init += [("empty",), ("conf", conf), ("atconf", conf)]


    for object_name in problem_info.surfaces:
        for link_name in problem_info.surfaces[object_name]:
            region = (object_name, link_name)
            init += [("region", region)]
            if object_name == "sink":
                init += [("sink", region)]
            if "burner" in link_name:
                init += [("burner", region)]

    goal = ["and",
        ("in", "cabbage1", ("plate", "base_link")),
        #("cooked", "cabbage1"),
        #("clean", "glass1"),
        #("in", "glass1", ("placemat", "base_link")),
    ]

    def get_station(name):
        if name in stations:
            return stations[name], station_contexts[name]
        stations[name] = problem_info.make_holding_station(
            name = name,
        )
        station_contexts[name] = stations[name].CreateDefaultContext()
        return stations[name], station_contexts[name]

    def find_motion(q1, q2, fluents = []):
        print(f"{Colors.BLUE}Starting trajectory stream{Colors.RESET}")
        print(fluents)
        station, station_context = get_station("move_free")
        holding = False
        for fluent in fluents:
            if fluent[0] == "holding":
                holding = True
                station, station_context = get_station(fluent[1])
        update_station(station, station_context, fluents)
        while True:
            print(f"{Colors.GREEN}Planning trajectory {item}, holding: {holding}{Colors.RESET}")
            traj = find_traj(
                station, 
                station_context, 
                q1, 
                q2, 
                ignore_endpoint_collisions= True
            )
            if traj is None:  # if a trajectory could not be found (invalid)
                print(f"{Colors.RED}Closing move-holding stream for {item}{Colors.RESET}")
                return
            print(f"{Colors.REVERSE}Yielding trajectory holding {item}{Colors.RESET}")
            yield traj,
            update_station(station, station_context, fluents)

    def find_grasp(item):
        print(f"{Colors.BLUE}Starting grasp stream for {item}{Colors.RESET}")
        station = stations["move_free"]
        object_info = station.object_infos[item][0]
        shape_info = update_graspable_shapes(object_info)[0]
        while True:
            print(f"{Colors.REVERSE}Yielding X_H{item}{Colors.RESET}")
            yield RigidTransformWrapper(
                kitchen_streamsv2.find_grasp(shape_info),
                name = f"X_H{item}"
            ),

    def find_place(holdingitem, region)  :
        object_name, link_name = region
        print(f"{Colors.BLUE}Starting place stream for {holdingitem} on region {object_name}, {link_name}{Colors.RESET}")
        station = stations["move_free"]
        station_context = station_contexts["move_free"]
        target_object_info = station.object_infos[object_name][0]
        holding_object_info = station.object_infos[holdingitem][0]
        shape_info = update_placeable_shapes(holding_object_info)[0]
        surface = update_surfaces(target_object_info, link_name, station, station_context)[0]
        while True:
            print(f"{Colors.GREEN}Finding place for {item} on {object_name}{Colors.RESET}")
            yield RigidTransformWrapper(
                kitchen_streamsv2.find_place(station, station_context, shape_info, surface),
                name = f"X_W{item}_in_{region}"
            ),

    def find_ik(item, X_WI, X_HI):
        print(f"{Colors.BLUE}Starting ik stream for {item}{Colors.RESET}")
        station, station_context = get_station(item)
        #station = stations["move_free"]
        #station_context = station_contexts["move_free"]
        update_station(
            station,
            station_context,
            [("holding", item, X_HI)],
            set_others_to_inf = True
        )
        object_info = station.object_infos[item][0]
        shape_info = update_graspable_shapes(object_info)[0]
        relax = False
        q_initial = Q_NOMINAL
        X = X_WI
        if isinstance(X_HI, RigidTransformWrapper):
            X = X_WI.get_rt()
        while True:
            q, cost = kitchen_streamsv2.find_ik(
                station,
                station_context,
                shape_info,
                X,
                relax = relax,
                q_initial = q_initial
            )
            #TODO(agro): relax and retry if failure
            if not np.isfinite(cost):
                return
            pre_q, _ = pre_and_post_grasps(station, station_context, q, dist = 0.07)
            yield pre_q, q
            update_station(
                station,
                station_context,
                [("holding", item, X_HI)],
                set_others_to_inf = True
            )

    def check_safe(q, item, X_WI):
        print(f"checking collisions with {item}")
        return True

    #def check_safe_place(q, itemholding, X_HI, item, X_WI):
    #    print(f"checking collisions between {itemholding} and {item}")
    #    return True

    stream_map = {
        "find-traj": from_gen_fn(find_motion),
        "find-grasp": from_gen_fn(find_grasp),
        "find-place": from_gen_fn(find_place),
        "find-ik": from_gen_fn(find_ik),
        "check-safe": from_test(check_safe),
        #"check-safe-place": from_test(check_safe_place)
    }

    return PDDLProblem(domain_pddl, {}, stream_pddl, stream_map, init, goal)

def make_and_init_simulation(zmq_url, prob):
    """
    Make the simulation, and let it run for 0.2 s to let all the objects
    settle into their stating positions
    """
    builder = pydrake.systems.framework.DiagramBuilder()
    problem_info = ProblemInfo(prob)
    station = problem_info.make_main_station()
    stations = {"main": station, "move_free": problem_info.make_move_free_station()}
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
    simulator.set_target_realtime_rate(3.0)

    # let objects fall for a bit in the simulation
    if meshcat is not None:
        meshcat.start_recording()
    simulator.AdvanceTo(SIM_INIT_TIME)
    return simulator, stations, director, meshcat, problem_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Use --url to specify the zmq_url for a meshcat server\nuse --problem to specify a .yaml problem file"
    )
    parser.add_argument("-u", "--url", nargs="?", default=None)
    parser.add_argument(
        "-p", "--problem", nargs="?", default="problems/kitchen_problem.yaml"
    )
    args = parser.parse_args()
    sim, station_dict, traj_director, meshcat_vis, prob_info = make_and_init_simulation(
        args.url, args.problem
    )
    problem = construct_problem_from_sim(sim, station_dict, prob_info)

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

        plan_to_trajectory(plan, traj_director, SIM_INIT_TIME)

        sim.AdvanceTo(traj_director.get_end_time())
        if meshcat_vis is not None:
            meshcat_vis.stop_recording()
            meshcat_vis.publish_recording()

        save = input(
            (
                f"{Colors.BOLD}\nType ENTER to exit without saving.\n"
                "To save the video to the file\n"
                f"media/<filename.html>, input <filename>{Colors.RESET}\n"
            )
        )
        if save != "":
            if not os.path.isdir("media"):
                os.mkdir("media")
            file = open("media/" + save + ".html", "w")
            file.write(meshcat_vis.vis.static_html())
            file.close()
