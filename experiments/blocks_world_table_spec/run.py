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
from pydrake.all import Role
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
    PlanToTrajectory,
    TrajectoryDirector,
    find_traj,
    Colors,
    RigidTransformWrapper,
    update_graspable_shapes,
    update_placeable_shapes,
    update_surfaces,
    find_pregrasp,
    Q_NOMINAL,
)
from tamp_statistics import (
    make_plot
)
import blocks_world_streams

VERBOSE = False

np.set_printoptions(precision=4, suppress=True)
np.random.seed(seed = 0)
random.seed(0)
ARRAY = tuple
SIM_INIT_TIME = 0.2
GRASP_DIST = 0.04

file = "pddl_spec"
file_path, _ = os.path.split(os.path.realpath(__file__))
domain_pddl = open(f"{file_path}/{file}/domain.pddl", "r").read()
stream_pddl = open(f"{file_path}/{file}/stream.pddl", "r").read()

def lprint(string):
    if VERBOSE:
        print(string)

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

    for object_name, link_name in problem_info.surfaces.items():
        for link_name in problem_info.surfaces[object_name]:
            table = (object_name, link_name)
            init += [("table", table)]

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
        if isinstance(on, list):
            init += [
                ("table-support", tuple(on), name, pose),
            ]
        else:
            init += [
                ("block-support", on, start_poses[on], name, pose),
            ]
        if not (name in supports):
            init += [
                ("clear", name),
            ]

    for arm in problem_info.arms:
        near_tables = problem_info.arms[arm]["near"]
        for table in near_tables:
            table = tuple(table)
            init += [
                ("near", arm, table)
            ]


    arms = parse_config(main_station, main_station_context)
    for arm, conf in arms.items():
        init += [
            ("arm", arm),
            ("empty", arm),
            ("conf", arm, conf),
            ("atconf", arm, conf)
        ]

    surfaces = []
    for object_name, link_name in problem_info.surfaces.items():
        surfaces.append((object_name, link_name))


    goal = ["and",
        #("on-block", "blue_block", "red_block"),
        ("on-block", "red_block", "green_block"),
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
        lprint(f"{Colors.BLUE}Starting trajectory stream{Colors.RESET}")
        station, station_context = get_station("move_free")
        holding_block = None
        other_name = None
        q_other = None
        lprint(f"{Colors.BOLD}FLUENTS FOR MOTION{Colors.RESET}")
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
        if other_name is not None:
            update_arm(station, station_context, other_name, q_other)
        panda = station.panda_infos[arm_name].panda
        while True:
            traj = find_traj(
                station, 
                station_context, 
                q1, 
                q2, 
                ignore_endpoint_collisions= False,
                panda = panda,
                verbose = VERBOSE
            )
            if traj is None:
                return
            yield traj,
            update_station(station, station_context, fluents)
            if other_name is not None:
                update_arm(station, station_context, other_name, q_other)


    def find_grasp(block):
        lprint(f"{Colors.BLUE}Starting grasp stream for {block}{Colors.RESET}")
        station = stations["move_free"]
        object_info = station.object_infos[block][0]
        shape_info = update_graspable_shapes(object_info)[0]
        while True:
            lprint(f"{Colors.REVERSE}Yielding X_H{block}{Colors.RESET}")
            yield RigidTransformWrapper(
                blocks_world_streams.find_grasp(shape_info),
                name = f"X_H{block}"
            ),

    def find_ik(arm_name, block, X_WB, X_HB):
        lprint(f"{Colors.BLUE}Starting ik stream for {block} at {X_WB}{Colors.RESET}")
        station, station_context = get_station("move_free")
        update_station(
            station,
            station_context,
            [("atworldpose", block, X_WB)],
            set_others_to_inf = True
        )
        object_info = station.object_infos[block][0]
        panda_info = station.panda_infos[arm_name]
        p_WB = X_WB.get_rt().translation()
        X_WP = panda_info.X_WB
        dy = p_WB[1] - X_WP.translation()[1] 
        dx = p_WB[0] - X_WP.translation()[0] 
        q0 = np.arctan2(dy, dx) - pydrake.math.RollPitchYaw(X_WP.rotation()).yaw_angle()
        q_initial = Q_NOMINAL[:]
        q_initial[0] = q0
        while True:
            lprint(f"{Colors.GREEN}Finding ik for {block}{Colors.RESET}")
            q, cost = blocks_world_streams.find_ik_with_relaxed(
                station,
                station_context,
                object_info,
                X_HB.get_rt(),
                panda_info,
                q_initial = q_initial
            )
            if not np.isfinite(cost):
                lprint(f"{Colors.RED}Failed ik for {block}{Colors.RESET}")
                return
            pre_q = find_pregrasp(
                station, station_context, q, 0.07, panda_info = panda_info
            )
            lprint(f"{Colors.REVERSE}Yielding ik for {block}{Colors.RESET}")
            yield pre_q, q
            update_station(
                station,
                station_context,
                [("atworldpose", block, X_WB)],
                set_others_to_inf = True
            )

    def find_table_place(table, block):
        lprint(f"{Colors.BLUE}Starting place stream for {block}{Colors.RESET}")
        station, station_context = get_station("move_free")
        holding_object_info = station.object_infos[block][0]
        shape_info = update_placeable_shapes(holding_object_info)[0]
        object_name, link_name = table
        while True:
            target_object_info = station.object_infos[object_name][0]
            surface = update_surfaces(target_object_info, link_name, station, station_context)[0]
            yield RigidTransformWrapper(
                blocks_world_streams.find_table_place(
                    station, station_context, shape_info, surface
                ),
                name = f"X_W{block}"
            ),

    def find_block_place(block, lowerblock, X_WL):
        station, station_context = get_station("move_free")
        update_station(
            station,
            station_context,
            [("atworldpose", lowerblock, X_WL)],
            set_others_to_inf = True
        )
        holding_object_info = station.object_infos[block][0]
        shape_info = update_placeable_shapes(holding_object_info)[0]
        target_object_info = station.object_infos[lowerblock][0]
        surface = update_surfaces(target_object_info, "base_link", station, station_context)[0]
        while True:
            yield RigidTransformWrapper(
                blocks_world_streams.find_block_place(
                    station, station_context, shape_info, surface
                ),
                name = f"X_W{block}_on_{lowerblock}"
            ),
            update_station(
                station,
                station_context,
                [("atworldpose", lowerblock, X_WL)],
                set_others_to_inf = True
            )

    def check_colfree_block(arm_name, q, block, X_WB):
        lprint(f"{Colors.BLUE}Checking for collisions between {arm_name} and {block}{Colors.RESET}")
        station, station_context = get_station("move_free")
        update_station(
            station,
            station_context,
            [("atpose", block, X_WB)],
            set_others_to_inf = True
        )
        return blocks_world_streams.check_colfree_block(
            station,
            station_context,
            arm_name,
            q
        )

    def check_colfree_arms(arm1_name, q1, arm2_name, q2):
        lprint(f"{Colors.BLUE}Checking for collisions between arms{Colors.RESET}")
        if arm1_name == arm2_name:
            return True
        station, station_context = get_station("move_free")
        update_station(
            station,
            station_context,
            [],
            set_others_to_inf = True
        )
        return blocks_world_streams.check_colfree_arms(
            station,
            station_context,
            arm1_name,
            q1,
            arm2_name,
            q2
        )

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
    station = problem_info.make_main_station(time_step = 1e-3)
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
            #role = Role.kProximity
        )
        meshcat.load()
    else:
        lprint("No meshcat server url provided, running without gui")

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
        "-p", "--problem", nargs="?", default=f"{file_path}/problems/blocks_world_problem.yaml"
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
        time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

        if not os.path.isdir("logs"):
            os.mkdir("logs")
        if not os.path.isdir(f"logs/{time}"):
            os.mkdir(f"logs/{time}")
        path = f"logs/{time}/"
        
        solution = solve(
            problem, algorithm=algorithm, verbose=VERBOSE, logpath = path
        )

        print(f"\n\n{algorithm} solution:")
        print_solution(solution)

        plan, _, evaluations = solution
        if plan is None:
            print(f"{Colors.RED}No solution found, exiting{Colors.RESET}")
            sys.exit(0)

        make_plot(path + "stats.json", save_path = path + "plots.png")

        if meshcat_vis is not None:
            action_map = {
                "move": {
                    "function": PlanToTrajectory.move,
                    "argument_indices": [2],
                    "arm_name": 0
                },
                "pick": {
                    "function": PlanToTrajectory.pick,
                    "argument_indices": [4,5,4],
                    "arm_name": 0
                },
                "place": {
                    "function": PlanToTrajectory.place,
                    "argument_indices": [4,5,4],
                    "arm_name": 0
                },
                "stack": {
                    "function": PlanToTrajectory.place,
                    "argument_indices": [6,7,6],
                    "arm_name": 0
                },
                "unstack": {
                    "function": PlanToTrajectory.pick,
                    "argument_indices": [5,6,5],
                    "arm_name": 0
                },
            }

            traj_maker = PlanToTrajectory(station_dict["main"])
            traj_maker.make_trajectory(
                plan, SIM_INIT_TIME, action_map
            )

            for panda_name in traj_directors:
                traj_director = traj_directors[panda_name]
                traj_director.add_panda_traj(traj_maker.trajectories[panda_name]["panda_traj"])
                traj_director.add_hand_traj(traj_maker.trajectories[panda_name]["hand_traj"])

            sim.AdvanceTo(traj_director.get_end_time())
            if meshcat_vis is not None:
                meshcat_vis.stop_recording()
                meshcat_vis.publish_recording()

            file = open(path + "recording.html", "w")
            file.write(meshcat_vis.vis.static_html())
            file.close()
