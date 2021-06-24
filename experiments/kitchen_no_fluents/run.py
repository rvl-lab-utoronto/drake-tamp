#!/usr/bin/env python3
"""
The module for running the kitchen TAMP problem.
See `problem 5` at this link for details:
http://tampbenchmark.aass.oru.se/index.php?title=Problems
"""
import matplotlib
matplotlib.use("Agg")
import time
import os
import copy
from re import I
import sys
import argparse
import numpy as np
import random
from datetime import datetime
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
    update_station,
    PlanToTrajectory,
    TrajectoryDirector,
    find_traj,
    Colors,
    RigidTransformWrapper,
    update_graspable_shapes,
    update_placeable_shapes,
    update_surfaces,
    pre_and_post_grasps,
    Q_NOMINAL,
)
from tamp_statistics import (
    CaptureOutput,
    process_pickle,
    make_plot
)
import kitchen_streamsv2
import cProfile, pstats, io


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
    for name, pose in start_poses.items():
        X_wrapper = RigidTransformWrapper(pose, name = f"X_W{name}")
        init += [
            ("item", name),
            ("worldpose", name, X_wrapper),
            ("atpose", name, X_wrapper),
        ]
        if "contained" in problem_info.objects[name]:
            init += [
                ("contained", name, X_wrapper, tuple(problem_info.objects[name]["contained"]))
            ]

    #for item in problem_info.objects:
        #init += [
            #problem_info.objects[item]["contained"]
        #]

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
        ("in", "cabbage1", ("leftplate", "base_link")),
        ("cooked", "cabbage1"),
        ("in", "cabbage2", ("rightplate", "base_link")),
        ("cooked", "cabbage2"),
        ("clean", "glass1"),
        #("clean", "glass2"),
        #("in", "glass1", ("leftplacemat", "leftside")),
        #("in", "glass2", ("rightplacemat", "leftside")),
        ("in", "raddish1", ("tray", "base_link")),
        ("in", "raddish7", ("tray", "base_link")),
        #("in", "raddish4", ("tray", "base_link")),
        #("in", "raddish5", ("tray", "base_link")),
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
        """
        Find a collision free trajectory from initial configuration
        `q1` to `q2`. `fluents` is a list of tuples of the form
        [('atpose', item, X_WI), ..., ('handpose', holdingitem, X_HI), ...].
        If the hand is holding an item, then the `handpose` tuple will
        indicate the relative transformation between the hand and the object
        X_HI. All other tuples indicate the worldpose of the items X_WI.
        """
        print(f"{Colors.BLUE}Starting trajectory stream{Colors.RESET}")
        station, station_context = get_station("move_free")
        #print(f"{Colors.BOLD}FLUENTS FOR MOTION{Colors.RESET}")
        holdingitem = None
        for fluent in fluents:
            #print(fluent[0], fluent[1], fluent[2])
            if fluent[0] == "holding":
                holdingitem = fluent[1]
                station, station_context = get_station(fluent[1])
        update_station(station, station_context, fluents)
        iter = 1
        while True:
            if holdingitem:
                print(f"{Colors.GREEN}Planning trajectory holding {holdingitem}{Colors.RESET}")
            else:
                print(f"{Colors.GREEN}Planning trajectory{Colors.RESET}")
            print(f"Try: {iter}")
            traj = find_traj(
                station, 
                station_context, 
                q1, 
                q2, 
                ignore_endpoint_collisions= False
            )
            if traj is None:  # if a trajectory could not be found (invalid)
                if holdingitem:
                    print(f"{Colors.GREEN}Closing trajectory stream holding {holdingitem}{Colors.RESET}")
                else:
                    print(f"{Colors.GREEN}Closing trajectory stream{Colors.RESET}")
                return
            if holdingitem:
                print(f"{Colors.GREEN}Yielding trajectory holding {holdingitem}{Colors.RESET}")
            else:
                print(f"{Colors.GREEN}Yielding trajectory{Colors.RESET}")
            yield traj,
            iter += 1
            update_station(station, station_context, fluents)

    def find_grasp(item):
        """
        Find a pose of the hand relative to the item X_HI given
        the item name
        """
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

    def find_place(holdingitem, region):
        """
        Find a stable placement pose X_WI for the item
        `holdingitem` in the region `region`
        """
        object_name, link_name = region
        print(f"{Colors.BLUE}Starting place stream for {holdingitem} on region {object_name}, {link_name}{Colors.RESET}")
        station, station_context = get_station("move_free")
        target_object_info = station.object_infos[object_name][0]
        holding_object_info = station.object_infos[holdingitem][0]
        shape_info = update_placeable_shapes(holding_object_info)[0]
        surface = update_surfaces(target_object_info, link_name, station, station_context)[0]
        while True:
            print(f"{Colors.GREEN}Finding place for {holdingitem} on {object_name}{Colors.RESET}")
            yield RigidTransformWrapper(
                kitchen_streamsv2.find_place(station, station_context, shape_info, surface),
                name = f"X_W{holdingitem}_in_{region}"
            ),

    def find_ik(item, X_WI, X_HI):
        """
        Position `item` at the worldpose `X_WI` and yield an IK solution
        to the problem with the end effector at X_WH = X_WI(X_HI)^{-1}
        """
        print(f"{Colors.BLUE}Starting ik stream for {item}{Colors.RESET}")
        station, station_context = get_station("move_free")
        update_station(
            station,
            station_context,
            [("aspose", item, X_WI)],
            set_others_to_inf = True
        )
        object_info = station.object_infos[item][0]
        #shape_info = update_graspable_shapes(object_info)[0]
        q_initial = Q_NOMINAL
        while True:
            print(f"{Colors.GREEN}Finding ik for {item}{Colors.RESET}")
            q, cost = kitchen_streamsv2.find_ik_with_relaxed(
                station,
                station_context,
                object_info,
                X_HI.get_rt(),
                q_initial = q_initial
            )
            if not np.isfinite(cost):
                return
            pre_q, _ = pre_and_post_grasps(station, station_context, q, dist = 0.07)
            yield pre_q, q
            update_station(
                station,
                station_context,
                [("atpose", item, X_WI)],
                set_others_to_inf = True
            )

    def check_safe(q, item, X_WI):
        print(f"{Colors.BLUE}Checking for collisions with {item}{Colors.RESET}")
        station, station_context = get_station("move_free")
        update_station(
            station,
            station_context,
            [("atpose", item, X_WI)],
            set_others_to_inf = True
        )
        res = kitchen_streamsv2.check_safe_conf(station, station_context, q)
        if res:
            print(f"{Colors.GREEN}No collisions with {item}{Colors.RESET}")
        else:
            print(f"{Colors.RED}Found collisions with {item}{Colors.RESET}")
        return res

    #def dist_fn(traj):
    #    res = 0
    #    for i in range(len(traj) - 1):
    #        res += np.dot(traj[i], traj[i+1])
    #    return 1+res


    stream_map = {
        "find-traj": from_gen_fn(find_motion),
        "find-grasp": from_gen_fn(find_grasp),
        "find-place": from_gen_fn(find_place),
        "find-ik": from_gen_fn(find_ik),
        "check-safe": from_test(check_safe),
        #"distance": dist_fn,
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
    
    panda_info = list(station.panda_infos.values())[0]
    director = builder.AddSystem(TrajectoryDirector())
    builder.Connect(
        station.get_output_port(panda_info.get_panda_position_measured_port()),
        director.GetInputPort("panda_position"),
    )
    builder.Connect(
        station.get_output_port(panda_info.get_hand_state_measured_port()),
        director.GetInputPort("hand_state"),
    )
    builder.Connect(
        director.GetOutputPort("panda_position_command"),
        station.get_input_port(panda_info.get_panda_position_port()),
    )
    builder.Connect(
        director.GetOutputPort("hand_position_command"),
        station.get_input_port(panda_info.get_hand_position_port()),
    )

    diagram = builder.Build()
    simulator = pydrake.systems.analysis.Simulator(diagram)
    simulator.set_target_realtime_rate(3.0)
    simulator.Initialize()

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
        time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

        if not os.path.isdir("logs"):
            os.mkdir("logs")
        if not os.path.isdir(f"logs/{time}"):
            os.mkdir(f"logs/{time}")

        path = f"logs/{time}/"
        
        sys.stdout = CaptureOutput(
            path + "stdout_logs.txt",
            #keywords = ["Iteration", "Attempt"] 
        )
        pr = cProfile.Profile()
        pr.enable()
        solution = solve(problem, algorithm=algorithm, verbose=False)
        pr.disable()
        sys.stdout = sys.stdout.original
        sys.stdout = CaptureOutput(
            path + "stdout_logs.txt",
        )
        print(f"\n\n{algorithm} solution:")
        print_solution(solution)

        ps = pstats.Stats(pr).sort_stats(2)
        #ps.print_stats()

        sys.stdout = sys.stdout.original

        process_pickle("statistics/py3/kitchen.pkl", path + "pddlstream_statistics.json")

        plan, _, evaluations = solution
        if plan is None:
            print(f"{Colors.RED}No solution found, exiting{Colors.RESET}")
            sys.exit(0)

        make_plot(path + "stdout_logs.txt", save_path = path + "graphs.png")

        if meshcat_vis is None:
            sys.exit(0)

        action_map = {
            "move": {
                "function": PlanToTrajectory.move,
                "argument_indices": [1],
                "arm_name": "panda"
            },
            "pick": {
                "function": PlanToTrajectory.pick,
                "argument_indices": [3,4,3],
                "arm_name": "panda" # index of action or string
            },
            "place": {
                "function": PlanToTrajectory.place,
                "argument_indices": [3,4,3],
                "arm_name": "panda"
            },
        }

        traj_maker = PlanToTrajectory(station_dict["main"])
        traj_maker.make_trajectory(
            plan, SIM_INIT_TIME, action_map
        )
        traj_director.add_panda_traj(traj_maker.trajectories["panda"]["panda_traj"])
        traj_director.add_hand_traj(traj_maker.trajectories["panda"]["hand_traj"])

        sim.AdvanceTo(traj_director.get_end_time())

        meshcat_vis.stop_recording()
        meshcat_vis.publish_recording()
        file = open(path + "recording.html", "w")
        file.write(meshcat_vis.vis.static_html())
        file.close()


        #save = input(
        #    (
        #        f"{Colors.BOLD}\nType ENTER to exit without saving.\n"
        #        "Type any key to save the video to the file\n"
        #        f"logs/<todays_date>/recording.html\n"
        #    )
        #)
        #if save != "":
