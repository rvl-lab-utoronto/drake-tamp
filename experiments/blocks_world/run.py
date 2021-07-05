#!/usr/bin/env python3
"""
The module for running the kitchen TAMP problem.
See `problem 5` at this link for details:
http://tampbenchmark.aass.oru.se/index.php?title=Problems
"""
import time
import numpy as np
import random
np.random.seed(seed = int(time.time()))
random.seed(int(time.time()))
import matplotlib
import yaml
matplotlib.use("Agg")
import os
import copy
from re import I
import sys
import argparse
from datetime import datetime
import pydrake.all
from pydrake.all import Role
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import str_from_object
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.algorithms.meta import solve
from pddlstream.algorithms.algorithm import reset_globals
#TODO(agro): see which of these are necessary
from learning import visualization
from learning import oracle as ora
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
from data_generation import make_problem

VERBOSE = False

np.set_printoptions(precision=4, suppress=True)
ARRAY = tuple
SIM_INIT_TIME = 0.2
GRASP_DIST = 0.04
DUMMY_STREAMS = False

file_path, _ = os.path.split(os.path.realpath(__file__))
domain_pddl = open(f"{file_path}/domain.pddl", "r").read()
stream_pddl = open(f"{file_path}/stream.pddl", "r").read()

def lprint(string):
    if VERBOSE:
        print(string)

def construct_problem_from_sim(simulator, stations, problem_info):
    """
    Construct pddlstream problem from simulator
    """

    init = []
    goal = problem_info.goal
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
        if "on-block" in problem_info.objects[name]:
            supports.append(problem_info.objects[name]["on-block"])

    for name, pose in start_poses.items():
        init += [
            ("block", name),
            ("worldpose", name, pose),
            ("atworldpose", name, pose),
        ]
        if "on-table" in problem_info.objects[name]:
            init += [
                ("table-support", name, pose, tuple(problem_info.objects[name]["on-table"])),
            ]
        elif "on-block" in problem_info.objects[name]:
            block = problem_info.objects[name]["on-block"]
            init += [
                ("block-support", name, pose, block, start_poses[block]),
            ]
        else:
            raise SyntaxError(f"Object {name} needs to specify on-table or on-block")
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


    for object_name in problem_info.surfaces:
        for link_name in problem_info.surfaces[object_name]:
            table = (object_name, link_name)
            init += [("table", table)]

    """
    goal = ["and",
        ("on-block", "block1", "block2"),
        #("on-block", "green_block", "blocker1"),
        #("on-table", "blocker1", ("blue_table", "base_link")),
        #("on-table", "blue_block", ("green_table", "base_link")),
    ]
    """

    oracle = ora.Oracle(
        domain_pddl,
        stream_pddl,
        init,
        goal,
    )

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

    def find_table_place(block, table):
        lprint(f"{Colors.BLUE}Starting place stream for {block} on {table}{Colors.RESET}")
        station, station_context = get_station("move_free")
        holding_object_info = station.object_infos[block][0]
        shape_info = update_placeable_shapes(holding_object_info)[0]
        while True:
            object_name, link_name = table
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

    stream_map = {
        "find-traj": from_gen_fn(find_motion),
        "find-ik": from_gen_fn(find_ik),
        "find-grasp": from_gen_fn(find_grasp),
        "find-table-place": from_gen_fn(find_table_place),
        "find-block-place": from_gen_fn(find_block_place),
        "check-colfree-block": from_test(check_colfree_block),
    }

    return PDDLProblem(domain_pddl, {}, stream_pddl, stream_map, init, goal), oracle

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

def setup_parser():
    parser = argparse.ArgumentParser(
        description= "Run the kitchen domain"
    )
    parser.add_argument(
        "-u",
        "--url",
        nargs="?",
        default=None,
        help = "Specify the zmq_url for a meshcat server",
    )
    parser.add_argument(
        "-p",
        "--problem",
        nargs="?",
        default=None,
        help = "Use --problem to specify a .yaml problem file",
    )
    parser.add_argument(
        "-m",
        "--mode",
        nargs = "?",
        default = "normal",
        choices=[
            'normal',
            'oracle',
            'save'
        ],
        help =  ("normal mode will run ORACLE=False, DEFAULT_UNIQUE=False\n"
                "save mode will run ORACLE=False, DEFAULT_UNIQUE=False and"
                "copy the stats.json to ~/drake_tamp/learning/data\n"
                "oracle mode will use the oracle with the latest stats.json")
    )
    return parser


def run_blocks_world(
    num_blocks = 2,
    num_blockers = 2,
    problem_file = None,
    mode = "normal",
    url = None,
    max_time = float("inf"),
    algorithm = "adaptive"
):

    time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    if not os.path.isdir(f"{file_path}/logs"):
        os.mkdir(f"{file_path}/logs")
    if not os.path.isdir(f"{file_path}/logs/{time}"):
        os.mkdir(f"{file_path}/logs/{time}")

    path = f"{file_path}/logs/{time}/"

    if problem_file is None:
        yaml_data = make_problem.make_random_problem(
            num_blocks=num_blocks, num_blockers=num_blockers, colorize=True
        )
        with open(f"{path}problem.yaml", "w") as stream:
            yaml.dump(yaml_data, stream, default_flow_style=False)
        problem_file = f"{path}problem.yaml"

    sim, station_dict, traj_directors, meshcat_vis, prob_info = make_and_init_simulation(
        url, problem_file
    )
    problem, oracle = construct_problem_from_sim(sim, station_dict, prob_info)

    print("Initial:", str_from_object(problem.init))
    print("Goal:", str_from_object(problem.goal))
        
    given_oracle = oracle if mode == "oracle" else None
    solution = solve(
        problem,
        algorithm=algorithm,
        verbose = VERBOSE,
        logpath = path,
        oracle = given_oracle,
        use_unique = mode == "oracle",
        max_time = max_time,
    )
    print(f"\n\n{algorithm} solution:")
    print_solution(solution)

    plan, _, evaluations = solution
    if plan is None:
        print(f"{Colors.RED}No solution found, exiting{Colors.RESET}")
        return False, problem_file

    if mode == "save":
        oracle.save_stats(
            path + "stats.json"
        )

    if mode == "oracle":
        oracle.save_labeled()

    make_plot(path + "stats.json", save_path = path + "plots.png")
    visualization.stats_to_graph(
        path + "stats.json", save_path = path + "preimage_graph.html"
    )

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

    return True, problem_file

def generate_data(num_blocks, num_blockers, max_time = float("inf")):

    res, problem_file = run_blocks_world(
        num_blocks = num_blocks,
        num_blockers = num_blockers,
        mode = "save",
        max_time = max_time,
    )
    if not res:
        return
    res, _ = run_blocks_world(
        problem_file = problem_file,
        mode = "oracle",
        max_time = max_time,
    )


if __name__ == "__main__":


    for num_blocks in range(2, 5):
        for num_blockers in range(3):
            generate_data(num_blocks, num_blockers)

    """
    parser = setup_parser()
    args = parser.parse_args()
    mode = args.mode.lower()
    run_blocks_world(
        num_blocks = 2,
        num_blockers = 1,
        problem_file = args.problem,
        mode = "oracle",
    )
    #reset_globals()
    run_blocks_world(
        problem_file = args.problem,
        mode = "oracle",
    )
    """