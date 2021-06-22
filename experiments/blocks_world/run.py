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
    update_arm,
    TrajectoryDirector,
    find_traj,
    Colors,
    RigidTransformWrapper,
    update_graspable_shapes,
    update_placeable_shapes,
    update_surfaces,
    pre_and_post_grasps,
    plan_to_trajectory,
    Q_NOMINAL,
)
from tamp_statistics import (
    CaptureOutput,
    process_pickle,
)

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


    # this needs to be done first so all of the RTs are updated
    supports = []
    for name, pose in start_poses.items():
        start_poses[name] = RigidTransformWrapper(pose, name = f"X_W{name}")
        supports.append(problem_info.objects[name]["support"])

    for name, pose in start_poses.items():
        init += [
            ("block", name),
            ("worldpose", name, pose),
            ("atworldpose", name, pose),
        ]
        on = problem_info.objects[name]["support"] 
        if on == "table":
            init += [
                ("table-support", name, pose),
            ]
        else:
            init += [
                ("block-support", on, start_poses[on], name, pose),
            ]
        if not (name in supports):
            init += [
                ("clear", name),
            ]

    arms = parse_config(main_station, main_station_context)
    for arm, conf in arms.items():
        init += [
            ("arm", arm),
            ("empty", arm),
            ("conf", arm, conf),
            ("atconf", arm, conf)
        ]

    goal = ["and",
        ("on-block", "red_block", "blue_block"),
        #("on-block", "blue_block", "red_block"),
    ]

    def get_station(name, arm_name = None):
        if arm_name is None:
            return stations[name], station_contexts[name] # ie. move_free

        if arm_name not in stations:
            stations[arm_name] = {}
            station_contexts[arm_name] = {}
        if name in stations[arm_name]:
            return stations[arm_name][name], station_contexts[arm_name][name]

        stations[arm_name][name] = problem_info.make_holding_station(
            name,
            arm_name = arm_name
        )
        station_contexts[arm_name][name] = stations[arm_name][name].CreateDefaultContext()
        return stations[arm_name][name], station_contexts[arm_name][name]

    def find_motion(arm_name, q1, q2, fluents = []):
        print(f"{Colors.BLUE}Starting trajectory stream{Colors.RESET}")
        station, station_context = get_station("move_free")
        holding_block = None
        other_name = None
        q_other = None
        print(f"{Colors.BOLD}FLUENTS FOR MOTION{Colors.RESET}")
        fluents = fluents.copy()
        i = 0
        while i < len(fluents):
            fluent = fluents[i]
            if fluent[0] == "atconf":
                fluents.pop(i)
                if fluent[1] != arm_name:
                    other_name = fluent[1]
                    q_other = fluent[2]
                continue
            if fluent[0] == "athandpose":
                holding_block = fluent[2]
                station, station_context = get_station(holding_block, arm_name = arm_name)
                fluents[i] = ("athandpose", holding_block, fluent[3])
            i+=1

        update_station(station, station_context, fluents)
        update_arm(station, station_context, other_name, q_other)
        panda = station.panda_infos[arm_name].panda
        while True:
            yield f"{arm_name}_traj",
            traj = find_traj(
                station, 
                station_context, 
                q1, 
                q2, 
                ignore_endpoint_collisions= False
                panda = panda
            )
            if traj is None:
                return
            yield traj,
            update_station(station, station_context, fluents)
            update_arm(station, station_context, other_name, q_other)


    def find_grasp(block):
        while True:
            yield f"X_H{block}",

    def find_ik(arm_name, block, X_WB, X_HB):
        while True:
            yield "pre_q", "q"

    def find_table_place(block):
        while True:
            yield f'X_W{block}',

    def find_block_place(block, lowerblock, X_WL):
        while True:
            yield f"X_W{block}_on_{lowerblock}",

    def check_colfree_block(arm_name, q, block, X_WB):
        return True

    def check_colfree_arms(arm1_name, q1, arm2_name, q2):
        return True

    stream_map = {
        "find-traj": from_gen_fn(find_motion),
        "find-ik": from_gen_fn(find_ik),
        "find-grasp": from_gen_fn(find_grasp),
        "find-table-place": from_gen_fn(find_table_place),
        "find-block-place": from_gen_fn(find_block_place),
        "check-colfree-block": from_test(check_colfree_block),
        "check-colfree-arms": from_test(check_colfree_arms),
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

    directors = {}

    for panda_info in station.panda_infos.values():
        arm_name = panda_info.panda_name
        director = builder.AddSystem(TrajectoryDirector())
        directors[arm_name] = director
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

    # let objects fall for a bit in the simulation
    if meshcat is not None:
        meshcat.start_recording()
    simulator.AdvanceTo(SIM_INIT_TIME)
    return simulator, stations, directors, meshcat, problem_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Use --url to specify the zmq_url for a meshcat server\nuse --problem to specify a .yaml problem file"
    )
    parser.add_argument("-u", "--url", nargs="?", default=None)
    parser.add_argument(
        "-p", "--problem", nargs="?", default="problems/blocks_world_problem.yaml"
    )
    args = parser.parse_args()
    sim, station_dict, traj_directors, meshcat_vis, prob_info = make_and_init_simulation(
        args.url, args.problem
    )
    problem = construct_problem_from_sim(sim, station_dict, prob_info)

    print("Initial:", str_from_object(problem.init))
    print("Goal:", str_from_object(problem.goal))
    for algorithm in [
        "adaptive",
    ]:
        """
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
        """
        solution = solve(problem, algorithm=algorithm, verbose=False)

        """
        sys.stdout = sys.stdout.original
        sys.stdout = CaptureOutput(
            path + "stdout_logs.txt",
        )
        """
        print(f"\n\n{algorithm} solution:")
        print_solution(solution)
        """
        sys.stdout = sys.stdout.original
        """

        """
        process_pickle("statistics/py3/kitchen.pkl", path + "pddlstream_statistics.json")
        """

        plan, _, evaluations = solution
        if plan is None:
            print(f"{Colors.RED}No solution found, exiting{Colors.RESET}")
            sys.exit(0)

        print("VIZ NOT SETUP YET")
        sys.exit(0)

        action_map = {
            "move": (
                plan_to_trajectory.move,
                [1],
            ),
            "pick":(
                plan_to_trajectory.pick,
                [3, 4, 3]
            ),
            "place":(
                plan_to_trajectory.place,
                [3, 4, 3]
            )
        }

        plan_to_trajectory.make_trajectory(
            plan, traj_director, SIM_INIT_TIME, action_map
        )

        sim.AdvanceTo(traj_director.get_end_time())
        if meshcat_vis is not None:
            meshcat_vis.stop_recording()
            meshcat_vis.publish_recording()

        #save = input(
        #    (
        #        f"{Colors.BOLD}\nType ENTER to exit without saving.\n"
        #        "Type any key to save the video to the file\n"
        #        f"logs/<todays_date>/recording.html\n"
        #    )
        #)
        #if save != "":
        file = open(path + "recording.html", "w")
        file.write(meshcat_vis.vis.static_html())
        file.close()
