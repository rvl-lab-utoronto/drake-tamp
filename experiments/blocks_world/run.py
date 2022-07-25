#!/usr/bin/env python3
"""
The module for running the blocks world TAMP problem.
See `problem 2` at this link for details:
http://tampbenchmark.aass.oru.se/index.php?title=Problems
"""
import time
import psutil
import numpy as np
import random
import itertools
import json
import sys

import matplotlib
import yaml

matplotlib.use("Agg")
from experiments.shared import construct_oracle
import os
from re import I
import argparse
from datetime import datetime
import pydrake.all
from pydrake.all import RigidTransform
from pydrake.all import Role
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import str_from_object
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.algorithms.meta import solve
from pddlstream.algorithms.algorithm import reset_globals
from learning import visualization
from learning import oracle as ora
from panda_station import (
    rt_to_xyzrpy,
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
from tamp_statistics import make_plot
from experiments.blocks_world import blocks_world_streams
from experiments.blocks_world.data_generation import make_problem
from pddlstream.language.stream import StreamInfo

VERBOSE = False

np.set_printoptions(precision=4, suppress=True)
ARRAY = tuple
SIM_INIT_TIME = 0.0
GRASP_DIST = 0.04
DUMMY_STREAMS = False
rams = []

file_path, _ = os.path.split(os.path.realpath(__file__))
domain_pddl = open(f"{file_path}/domain.pddl", "r").read()
stream_pddl = open(f"{file_path}/stream.pddl", "r").read()


def lprint(string):
    if VERBOSE:
        print(string)

def retrieve_model_poses(
    main_station, main_station_context, model_names, link_names,
):
    """
    For each model = model_names[i], return the worldpose of main_link_name[i]

  
    where X_WM is a RigidTransform reprsenting the model worldpose
    """

    plant = main_station.get_multibody_plant()
    plant_context = main_station.GetSubsystemContext(plant, main_station_context)

    res = []

    for model_name, link_name in zip(model_names, link_names):
        body = plant.GetBodyByName(
            link_name,
            plant.GetModelInstanceByName(model_name)
        )
        X_WB = plant.EvalBodyPoseInWorld(plant_context, body)
        res.append(
            {
                "name": (model_name,link_name),
                "X": X_WB,
            }
        )

    return res

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
    transforms = {}
    for name, pose in start_poses.items():
        transform = RigidTransformWrapper(pose, name=f"X_W{name}")
        transforms[name] = transform
        start_poses[name] = transform
        if "on-block" in problem_info.objects[name]:
            supports.append(problem_info.objects[name]["on-block"])

    for i, fact in enumerate(goal):
        if not isinstance(fact, tuple):
            continue
        goal[i] = list(goal[i])
        for j, obj in enumerate(fact):
            if isinstance(obj, list):
                goal[i][j] = transforms[fact[j-1]]
        goal[i] = tuple(goal[i])

    for name, pose in start_poses.items():
        init += [
            ("block", name),
            ("worldpose", name, pose),
            ("atworldpose", name, pose),
        ]
        if "on-table" in problem_info.objects[name]:
            init += [
                (
                    "table-support",
                    name,
                    pose,
                    tuple(problem_info.objects[name]["on-table"]),
                ),
                ("on-table", name, tuple(problem_info.objects[name]["on-table"]))
            ]
        elif "on-block" in problem_info.objects[name]:
            block = problem_info.objects[name]["on-block"]
            init += [
                ("block-support", name, pose, block, start_poses[block]),
                ("on-block", name, block)
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
            ("atconf", arm, conf),
        ]

    static_model_names = []
    static_link_names = []

    for object_name in problem_info.surfaces:
        for link_name in problem_info.surfaces[object_name]:
            table = (object_name, link_name)
            init += [("table", table)]
            static_model_names.append(object_name)
            static_link_names.append(link_name)

    static_model_poses = retrieve_model_poses(
        main_station,
        main_station_context,
        static_model_names,
        static_link_names
    )

    for s in static_model_poses:
        s["static"] = True
    
    start_poses_d = []
    for s in start_poses.items():
        start_poses_d.append(
            {
                "name": s[0],
                "X": s[1],
                "static": False
            }
        )

    model_poses = start_poses_d + static_model_poses
    model_poses.append(
        {
            "name": "panda",
            "X": RigidTransform(),
            "static": True,
        }
    )

    def get_station(name, arm_name=None):
        if arm_name is None:
            return stations[name], station_contexts[name]  # ie. move_free

        if arm_name not in stations:
            stations[arm_name] = {}
            station_contexts[arm_name] = {}
        if name in stations[arm_name]:
            return stations[arm_name][name], station_contexts[arm_name][name]

        stations[arm_name][name] = problem_info.make_holding_station(
            name, arm_name=arm_name
        )
        station_contexts[arm_name][name] = stations[arm_name][
            name
        ].CreateDefaultContext()
        return stations[arm_name][name], station_contexts[arm_name][name]

    def find_motion(arm_name, q1, q2, fluents=[]):
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
                station, station_context = get_station(holding_block, arm_name=arm_name)
                fluents[i] = ("athandpose", holding_block, fluent[3])
            i += 1

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
                ignore_endpoint_collisions=False,
                panda=panda,
                verbose=VERBOSE,
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
                blocks_world_streams.find_grasp(shape_info), name=f"X_H{block}"
            ),

    def find_ik(arm_name, block, X_WB, X_HB):
        lprint(f"{Colors.BLUE}Starting ik stream for {block} at {X_WB}{Colors.RESET}")
        station, station_context = get_station("move_free")
        update_station(
            station,
            station_context,
            [("atworldpose", block, X_WB)],
            set_others_to_inf=True,
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
                q_initial=q_initial,
            )
            if not np.isfinite(cost):
                lprint(f"{Colors.RED}Failed ik for {block}{Colors.RESET}")
                return
            pre_q = find_pregrasp(
                station, station_context, q, 0.07, panda_info=panda_info
            )
            lprint(f"{Colors.REVERSE}Yielding ik for {block}{Colors.RESET}")
            yield pre_q, q
            update_station(
                station,
                station_context,
                [("atworldpose", block, X_WB)],
                set_others_to_inf=True,
            )

    def find_table_place(block, table):
        lprint(
            f"{Colors.BLUE}Starting place stream for {block} on {table}{Colors.RESET}"
        )
        station, station_context = get_station("move_free")
        holding_object_info = station.object_infos[block][0]
        shape_info = update_placeable_shapes(holding_object_info)[0]
        while True:
            object_name, link_name = table
            target_object_info = station.object_infos[object_name][0]
            surface = update_surfaces(
                target_object_info, link_name, station, station_context
            )[0]
            yield RigidTransformWrapper(
                blocks_world_streams.find_table_place(
                    station, station_context, shape_info, surface
                ),
                name=f"X_W{block}",
            ),

    def find_block_place(block, lowerblock, X_WL):
        station, station_context = get_station("move_free")
        update_station(
            station,
            station_context,
            [("atworldpose", lowerblock, X_WL)],
            set_others_to_inf=True,
        )
        holding_object_info = station.object_infos[block][0]
        shape_info = update_placeable_shapes(holding_object_info)[0]
        target_object_info = station.object_infos[lowerblock][0]
        surface = update_surfaces(
            target_object_info, "base_link", station, station_context
        )[0]
        while True:
            yield RigidTransformWrapper(
                blocks_world_streams.find_block_place(
                    station, station_context, shape_info, surface
                ),
                name=f"X_W{block}_on_{lowerblock}",
            ),
            update_station(
                station,
                station_context,
                [("atworldpose", lowerblock, X_WL)],
                set_others_to_inf=True,
            )

    def check_colfree_block(arm_name, q, block, X_WB):
        lprint(
            f"{Colors.BLUE}Checking for collisions between {arm_name} and {block}{Colors.RESET}"
        )
        station, station_context = get_station("move_free")
        update_station(
            station, station_context, [("atpose", block, X_WB)], set_others_to_inf=True
        )
        return blocks_world_streams.check_colfree_block(
            station, station_context, arm_name, q
        )

    stream_map = {
        "find-traj": from_gen_fn(find_motion),
        "find-ik": from_gen_fn(find_ik),
        "find-grasp": from_gen_fn(find_grasp),
        "find-table-place": from_gen_fn(find_table_place),
        "find-block-place": from_gen_fn(find_block_place),
        "check-colfree-block": from_test(check_colfree_block),
    }

    return PDDLProblem(domain_pddl, {}, stream_pddl, stream_map, init, goal), model_poses


def make_and_init_simulation(zmq_url, prob):
    """
    Make the simulation, and let it run for 0.2 s to let all the objects
    settle into their stating positions
    """

    builder = pydrake.systems.framework.DiagramBuilder()
    problem_info = ProblemInfo(prob)
    station = problem_info.make_main_station(time_step=1e-3)
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
            # role = Role.kProximity
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
    parser = argparse.ArgumentParser(description="Run the kitchen domain")
    parser.add_argument(
        "-u",
        "--url",
        nargs="?",
        default=None,
        help="Specify the zmq_url for a meshcat server",
    )
    parser.add_argument(
        "-p",
        "--problem",
        nargs="?",
        default=None,
        help="Use --problem to specify a .yaml problem file",
    )
    parser.add_argument(
        "-m",
        "--mode",
        nargs="?",
        default="normal",
        choices=["normal", "oracle", "save"],
        help=(
            "normal mode will run ORACLE=False, DEFAULT_UNIQUE=False\n"
            "save mode will run ORACLE=False, DEFAULT_UNIQUE=False and"
            "copy the stats.json to ~/drake_tamp/learning/data\n"
            "oracle mode will use the oracle with the latest stats.json"
        ),
    )
    return parser


def run_blocks_world(
    num_blocks=2,
    num_blockers=2,
    problem_file=None,
    mode="normal",
    url=None,
    max_time=float("inf"),
    algorithm="adaptive",
    buffer_radius=0,
    simulate=False,
    max_stack_num = None,
    use_unique=False,
    oracle_kwargs={},
    should_save=False,
    eager_mode=False,
    path=None,  
    max_planner_time = 10,
):

    memory_percent = psutil.virtual_memory().percent
    rams.append(memory_percent)
    if memory_percent >= 95:
        print(f"{Colors.RED}You have used up all the memory!{Colors.RESET}")
        with open("mem.json", "w") as f:
            json.dump(rams, f)
        sys.exit(1)

    time = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
    if not os.path.isdir(f"{file_path}/logs"):
        os.mkdir(f"{file_path}/logs")
    if not os.path.isdir(f"{file_path}/logs/{time}"):
        os.mkdir(f"{file_path}/logs/{time}")

    if path is None:
        path = f"{file_path}/logs/{time}/"

    if problem_file is None:
        yaml_data = make_problem.make_random_problem(
            num_blocks=num_blocks,
            num_blockers=num_blockers,
            colorize=True,
            buffer_radius=buffer_radius,
            max_start_stack = 1,
            max_goal_stack=max_stack_num
        )
        with open(f"{path}problem.yaml", "w") as stream:
            yaml.dump(yaml_data, stream, default_flow_style=False)
        problem_file = f"{path}problem.yaml"

    (
        sim,
        station_dict,
        traj_directors,
        meshcat_vis,
        prob_info,
    ) = make_and_init_simulation(url, problem_file)
    problem, model_poses = construct_problem_from_sim(sim, station_dict, prob_info)
    oracle = construct_oracle(mode, problem, prob_info, model_poses, **oracle_kwargs)

    print("Initial:", str_from_object(problem.init))
    print("Goal:", str_from_object(problem.goal))

    given_oracle = oracle if mode not in ["normal", "save"] else None
    search_sample_ratio = 1 if (mode == "save" or mode == "normal") else 1
    solution = solve(
        problem,
        algorithm=algorithm,
        verbose=VERBOSE,
        logpath=path,
        oracle=given_oracle,
        use_unique=use_unique,
        max_time=max_time,
        eager_mode=eager_mode,
        search_sample_ratio=search_sample_ratio,
        max_planner_time = max_planner_time,
        problem_file_path = problem_file,
        # stream_info = {
        #     'find-ik': StreamInfo(use_unique=False)
        # },
    )
    print(f"\n\n{algorithm} solution:")
    print_solution(solution)

    plan, _, evaluations = solution
    if plan is None:
        print(f"{Colors.RED}No solution found, exiting{Colors.RESET}")
        return False, problem_file

    if len(plan) == 0:
        print(f"{Colors.BOLD}Empty plan, no real problem provided, exiting.{Colors.RESET}")
        return False, problem_file

    if should_save or mode == 'save':
        oracle.save_stats(path + "stats.json")

    if not algorithm.startswith("informed"):
        make_plot(path + "stats.json", save_path=path + "plots.png")
        visualization.stats_to_graph(
            path + "stats.json", save_path=path + "preimage_graph.html"
        )

    if simulate:
        action_map = {
            "move": {
                "function": PlanToTrajectory.move,
                "argument_indices": [2],
                "arm_name": 0,
            },
            "pick": {
                "function": PlanToTrajectory.pick,
                "argument_indices": [5, 6, 5],
                "arm_name": 0,
            },
            "place": {
                "function": PlanToTrajectory.place,
                "argument_indices": [5, 6, 5],
                "arm_name": 0,
            },
            "stack": {
                "function": PlanToTrajectory.place,
                "argument_indices": [6, 7, 6],
                "arm_name": 0,
            },
            "unstack": {
                "function": PlanToTrajectory.pick,
                "argument_indices": [5, 6, 5],
                "arm_name": 0,
            },
        }

        traj_maker = PlanToTrajectory(station_dict["main"])
        traj_maker.make_trajectory(plan, SIM_INIT_TIME, action_map)

        for panda_name in traj_directors:
            traj_director = traj_directors[panda_name]
            traj_director.add_panda_traj(
                traj_maker.trajectories[panda_name]["panda_traj"]
            )
            traj_director.add_hand_traj(
                traj_maker.trajectories[panda_name]["hand_traj"]
            )

        sim.AdvanceTo(traj_director.get_end_time())
        if meshcat_vis is not None:
            meshcat_vis.stop_recording()
            meshcat_vis.publish_recording()
            file = open(path + "recording.html", "w")
            file.write(meshcat_vis.vis.static_html())
            file.close()

    return True, problem_file


def generate_data(
    num_blocks,
    num_blockers,
    max_time=float("inf"),
    buffer_radius=0,
    url=None,
    simulate=False,
    max_stack_num=None,
    num_repeat_per_problem = 3
):

    """
    params: 
        num_blocks: the number of blocks that will be used for stacking

        num_blockers: the number of larger blocks that will be extraneous,
        but possibly obstructive

        max_time: the maximum time until pddlstream times out

        buffer_radius: An addition to the minimum distance between objects
            in the world (both manipulands and static objects). This is provided
            in meters.

        max_stack_num:
            The maximum number of blocks that will be stacked ontop of one another
            in the start or goal states.
            The having many blocks stacked ontop of one another becomes very 
            computatinoaly expensive to simulate even the first 0.2 s during
            the initialization phase (if SIM_INIT_TIME > 0).

        url: zmq_url where meshcat server is running. 
            Use drake-tamp/experiments/start_meshcat_server.py to create one

        simulate: whether or not the plan should be simulated
    """

    res, problem_file = run_blocks_world(
        num_blocks=num_blocks,
        num_blockers=num_blockers,
        mode="save",
        max_time=max_time,
        buffer_radius=buffer_radius,
        url=url,
        simulate=simulate,
        max_stack_num=max_stack_num,
    )
    if not res:
        return
    mode = "oracle"
    for i in range((num_repeat_per_problem*2) - 1):
        mode = "oracle" if (i % 2 == 0) else "save"
        res, _ = run_blocks_world(
            problem_file=problem_file,
            mode="oracle",
            url = url,
            simulate = simulate,
            max_time=max_time,
        )
        if not res:
            data = []
            if os.path.isfile(os.path.join(file_path, "oracle_failures.json")):
                with open("oracle_failures.json", "r") as f:
                    data = json.load(f)
            data.append(problem_file)
            with open("oracle_failures.json", "w") as f:
                json.dump(data, f, sort_keys=True, indent = 4)



def main_generation_loop():
    url = None#"tcp://127.0.0.1:6000"

    max_num_blocks = 6
    max_num_blockers = 6

    for num_blocks, num_blockers in itertools.product(
        range(1, max_num_blocks + 1), range(max_num_blockers + 1)
    ):

        for max_stack in range(1, num_blocks + 1):
            generate_data(
                num_blocks,
                num_blockers,
                buffer_radius=0,
                url=url,
                max_stack_num= max_stack,
                simulate = False,
                max_time = 360,
                num_repeat_per_problem= 3
            )

if __name__ == '__main__':
    # main_generation_loop()
    #url = "tcp://127.0.0.1:6003"

    res, _ = run_blocks_world(
        problem_file=os.path.join(file_path, "data_generation", "test_problem.yaml"),
        mode="save",
        #url = url,
        #simulate = False,
        max_time = 180
    )

    res, _ = run_blocks_world(
        problem_file=os.path.join(file_path, "data_generation", "test_problem.yaml"),
        mode="oracle",
        #url = url,
        #simulate = False,
        max_time = 180
    )
