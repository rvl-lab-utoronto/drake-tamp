#!/usr/bin/env python3
"""
The module for running the kitchen TAMP problem.
See `problem 5` at this link for details:
http://tampbenchmark.aass.oru.se/index.php?title=Problems
"""
import time
import numpy as np
import random

np.random.seed(seed=int(time.time()))
random.seed(int(time.time()))
import matplotlib

matplotlib.use("Agg")
import time
import yaml
import os
from re import I
import sys
import argparse
from datetime import datetime
import pydrake.all
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import str_from_object
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.algorithms.meta import solve
from learning import oracle as ora
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
    find_pregrasp,
    Q_NOMINAL,
)
from tamp_statistics import make_plot
from learning import visualization
import kitchen_streamsv2
from kitchen_data_generation import make_problem

VERBOSE = False

np.set_printoptions(precision=4, suppress=True)
np.random.seed(seed=0)
random.seed(0)
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
    for name, pose in start_poses.items():
        X_wrapper = RigidTransformWrapper(pose, name=f"X_W{name}")
        init += [
            ("item", name),
            ("worldpose", name, X_wrapper),
            ("atpose", name, X_wrapper),
        ]
        if "contained" in problem_info.objects[name]:
            init += [
                (
                    "contained",
                    name,
                    X_wrapper,
                    tuple(problem_info.objects[name]["contained"]),
                )
            ]

    # for item in problem_info.objects:
    # init += [
    # problem_info.objects[item]["contained"]
    # ]

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

    """
    goal = ["and",
        #("in", "cabbage1", ("leftplate", "base_link")),
        ("cooked", "cabbage1"),
        #("in", "cabbage2", ("rightplate", "base_link")),
        #("cooked", "cabbage2"),
        ("clean", "glass1"),
        #("clean", "glass2"),
        ("in", "glass1", ("leftplacemat", "leftside")),
        #("in", "glass2", ("rightplacemat", "leftside")),
        #("in", "raddish1", ("tray", "base_link")),
        #("in", "raddish7", ("tray", "base_link")),
        #("in", "raddish4", ("tray", "base_link")),
        #("in", "raddish5", ("tray", "base_link")),
    ]
    """

    oracle = ora.Oracle(
        domain_pddl,
        stream_pddl,
        init,
        goal,
    )

    def get_station(name):
        if name in stations:
            return stations[name], station_contexts[name]
        stations[name] = problem_info.make_holding_station(
            name=name,
        )
        station_contexts[name] = stations[name].CreateDefaultContext()
        return stations[name], station_contexts[name]

    def find_motion(q1, q2, fluents=[]):
        """
        Find a collision free trajectory from initial configuration
        `q1` to `q2`. `fluents` is a list of tuples of the form
        [('atpose', item, X_WI), ..., ('handpose', holdingitem, X_HI), ...].
        If the hand is holding an item, then the `handpose` tuple will
        indicate the relative transformation between the hand and the object
        X_HI. All other tuples indicate the worldpose of the items X_WI.
        """
        while DUMMY_STREAMS:
            yield f"traj_{q1}_{q2}",
        lprint(f"{Colors.BLUE}Starting trajectory stream{Colors.RESET}")
        station, station_context = get_station("move_free")
        # print(f"{Colors.BOLD}FLUENTS FOR MOTION{Colors.RESET}")
        holdingitem = None
        for fluent in fluents:
            # print(fluent[0], fluent[1], fluent[2])
            if fluent[0] == "holding":
                holdingitem = fluent[1]
                station, station_context = get_station(fluent[1])
        update_station(station, station_context, fluents)
        iter = 1
        while True:
            if holdingitem:
                lprint(
                    f"{Colors.GREEN}Planning trajectory holding {holdingitem}{Colors.RESET}"
                )
            else:
                lprint(f"{Colors.GREEN}Planning trajectory{Colors.RESET}")
            lprint(f"Try: {iter}")
            traj = find_traj(
                station,
                station_context,
                q1,
                q2,
                ignore_endpoint_collisions=False,
                verbose=False,
            )
            if traj is None:  # if a trajectory could not be found (invalid)
                if holdingitem:
                    lprint(
                        f"{Colors.GREEN}Closing trajectory stream holding {holdingitem}{Colors.RESET}"
                    )
                else:
                    lprint(f"{Colors.GREEN}Closing trajectory stream{Colors.RESET}")
                return
            if holdingitem:
                lprint(
                    f"{Colors.GREEN}Yielding trajectory holding {holdingitem}{Colors.RESET}"
                )
            else:
                lprint(f"{Colors.GREEN}Yielding trajectory{Colors.RESET}")
            yield traj,
            iter += 1
            update_station(station, station_context, fluents)

    def find_grasp(item):
        """
        Find a pose of the hand relative to the item X_HI given
        the item name
        """
        while DUMMY_STREAMS:
            yield f"grasp_{item}",
        lprint(f"{Colors.BLUE}Starting grasp stream for {item}{Colors.RESET}")
        station = stations["move_free"]
        object_info = station.object_infos[item][0]
        shape_info = update_graspable_shapes(object_info)[0]
        while True:
            lprint(f"{Colors.REVERSE}Yielding X_H{item}{Colors.RESET}")
            yield RigidTransformWrapper(
                kitchen_streamsv2.find_grasp(shape_info), name=f"X_H{item}"
            ),

    def find_place(holdingitem, region):
        """
        Find a stable placement pose X_WI for the item
        `holdingitem` in the region `region`
        """
        while DUMMY_STREAMS:
            yield f"place_{holdingitem}_in_{region}",
        object_name, link_name = region
        lprint(
            f"{Colors.BLUE}Starting place stream for {holdingitem} on region {object_name}, {link_name}{Colors.RESET}"
        )
        station, station_context = get_station("move_free")
        target_object_info = station.object_infos[object_name][0]
        holding_object_info = station.object_infos[holdingitem][0]
        shape_info = update_placeable_shapes(holding_object_info)[0]
        surface = update_surfaces(
            target_object_info, link_name, station, station_context
        )[0]
        while True:
            lprint(
                f"{Colors.GREEN}Finding place for {holdingitem} on {object_name}{Colors.RESET}"
            )
            yield RigidTransformWrapper(
                kitchen_streamsv2.find_place(
                    station, station_context, shape_info, surface
                ),
                name=f"X_W{holdingitem}_in_{region}",
            ),

    def find_ik(item, X_WI, X_HI):
        """
        Position `item` at the worldpose `X_WI` and yield an IK solution
        to the problem with the end effector at X_WH = X_WI(X_HI)^{-1}
        """
        while DUMMY_STREAMS:
            yield "pre_q", "q"
        lprint(f"{Colors.BLUE}Starting ik stream for {item}{Colors.RESET}")
        station, station_context = get_station("move_free")
        update_station(
            station, station_context, [("aspose", item, X_WI)], set_others_to_inf=True
        )
        object_info = station.object_infos[item][0]
        # shape_info = update_graspable_shapes(object_info)[0]
        q_initial = Q_NOMINAL
        while True:
            lprint(f"{Colors.GREEN}Finding ik for {item}{Colors.RESET}")
            q, cost = kitchen_streamsv2.find_ik_with_relaxed(
                station,
                station_context,
                object_info,
                X_HI.get_rt(),
                q_initial=q_initial,
            )
            if not np.isfinite(cost):
                return
            pre_q = find_pregrasp(station, station_context, q, 0.07)
            yield pre_q, q
            update_station(
                station,
                station_context,
                [("atpose", item, X_WI)],
                set_others_to_inf=True,
            )

    def check_safe(q, item, X_WI):
        if DUMMY_STREAMS:
            return True
        lprint(f"{Colors.BLUE}Checking for collisions with {item}{Colors.RESET}")
        station, station_context = get_station("move_free")
        update_station(
            station, station_context, [("atpose", item, X_WI)], set_others_to_inf=True
        )
        res = kitchen_streamsv2.check_safe_conf(station, station_context, q)
        if res:
            lprint(f"{Colors.GREEN}No collisions with {item}{Colors.RESET}")
        else:
            lprint(f"{Colors.RED}Found collisions with {item}{Colors.RESET}")
        return res

    # def dist_fn(traj):
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
        # "distance": dist_fn,
    }

    return PDDLProblem(domain_pddl, {}, stream_pddl, stream_map, init, goal), oracle


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
        default=f"{file_path}/problems/kitchen_problem.yaml",
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


def run_kitchen(
    num_cabbages=1,
    num_raddishes=0,
    num_glasses=2,
    problem_file=None,
    mode="normal",
    url=None,
    max_time=float("inf"),
    algorithm="adaptive",
    simulate=False,
    prob_tray=0.4,
    prob_sink=0.1,
    buffer_radius=0,
    prob_goal = 1,
):

    time = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
    if not os.path.isdir(f"{file_path}/logs"):
        os.mkdir(f"{file_path}/logs")
    if not os.path.isdir(f"{file_path}/logs/{time}"):
        os.mkdir(f"{file_path}/logs/{time}")

    path = f"{file_path}/logs/{time}/"

    if problem_file is None:
        yaml_data = make_problem.make_random_problem(
            num_cabbages,
            num_raddishes,
            num_glasses,
            prob_tray=prob_tray,
            prob_sink=prob_sink,
            buffer_radius=buffer_radius,
            prob_goal = prob_goal,
        )
        with open(f"{path}problem.yaml", "w") as stream:
            yaml.dump(yaml_data, stream, default_flow_style=False)
        problem_file = f"{path}problem.yaml"

    sim, station_dict, traj_director, meshcat_vis, prob_info = make_and_init_simulation(
        url, problem_file
    )
    problem, oracle = construct_problem_from_sim(sim, station_dict, prob_info)

    print("Initial:", str_from_object(problem.init))
    print("Goal:", str_from_object(problem.goal))

    given_oracle = oracle if mode == "oracle" else None
    solution = solve(
        problem,
        algorithm=algorithm,
        verbose=VERBOSE,
        logpath=path,
        oracle=given_oracle,
        use_unique=mode == "oracle",
        max_time=max_time,
    )

    print(f"\n\n{algorithm} solution:")
    print_solution(solution)

    plan, _, evaluations = solution
    if plan is None:
        print(f"{Colors.RED}No solution found, exiting{Colors.RESET}")
        return False, problem_file
        # sys.exit(0)

    make_plot(path + "stats.json", save_path=path + "plots.png")
    visualization.stats_to_graph(
        path + "stats.json", save_path=path + "preimage_graph.html"
    )

    if mode == "save":
        oracle.save_stats(path + "stats.json")

    if mode == "oracle":
        oracle.save_labeled()

    if simulate:

        action_map = {
            "move": {
                "function": PlanToTrajectory.move,
                "argument_indices": [1],
                "arm_name": "panda",
            },
            "pick": {
                "function": PlanToTrajectory.pick,
                "argument_indices": [3, 4, 3],
                "arm_name": "panda",  # index of action or string
            },
            "place": {
                "function": PlanToTrajectory.place,
                "argument_indices": [3, 4, 3],
                "arm_name": "panda",
            },
        }

        traj_maker = PlanToTrajectory(station_dict["main"])
        traj_maker.make_trajectory(plan, SIM_INIT_TIME, action_map)
        traj_director.add_panda_traj(traj_maker.trajectories["panda"]["panda_traj"])
        traj_director.add_hand_traj(traj_maker.trajectories["panda"]["hand_traj"])

        sim.AdvanceTo(traj_director.get_end_time())

        if meshcat_vis is not None:
            meshcat_vis.stop_recording()
            meshcat_vis.publish_recording()
            file = open(path + "recording.html", "w")
            file.write(meshcat_vis.vis.static_html())
            file.close()

    return True, problem_file


def generate_data(
    num_cabbages,
    num_raddishes,
    num_glasses,
    max_time=float("inf"),
    url=None,
    simulate=False,
    prob_sink=0.1,
    prob_tray=0.4,
    buffer_radius=0,
    prob_goal = 1,
):
    """
    
    params:
        num_cabbage: number of cabbages added 

        num_raddishes: number of raddishes added

        num_glasses: number of glasses added

        max_time: maximum time until timeout for pddlstream

        url: zmq_url where meshcat server is running. 
            Use drake-tamp/experiments/start_meshcat_server.py to create one

        simulate: whether or not the plan should be simulated

        prob_sink: the `target` probability for items in the sink.
            By target I mean that if the object cannot be placed in the sink
            with that probability, it will be placed elsewhere

        prob_tray: the target probability for item in the tray

        buffer_radius: An addition to the minimum distance between objects
            in the world (both manipulands and static objects). This is provided
            in meters.

        prob_goal: Approximately the probabilty of an item appearing in the goal state.
        The raddishes by default have only a 10% chance of being part of a goal, so making
        this value lower than one will given them a prob_goal * 0.1 percent chance of having
        a goal
    """

    res, problem_file = run_kitchen(
        num_cabbages=num_cabbages,
        num_raddishes=num_raddishes,
        num_glasses=num_glasses,
        max_time=max_time,
        mode="save",
        url=url,
        simulate=simulate,
        prob_sink=prob_sink,
        prob_tray=prob_tray,
        buffer_radius=buffer_radius,
        prob_goal = prob_goal
    )
    if not res:
        return
    res, _ = run_kitchen(
        problem_file=problem_file,
        mode="oracle",
        max_time=max_time,
    )


if __name__ == "__main__":

    url = None#"tcp://127.0.0.1:6000"

    num_cabbages = 3
    num_raddishes = 3
    num_glasses = 3

    generate_data(
        num_cabbages=num_cabbages,
        num_raddishes=num_raddishes,
        num_glasses=num_glasses,
        prob_tray=0.4,
        prob_sink=0.1,
        prob_goal = 0.5,
        buffer_radius=0,
        url=url,
        simulate=False,
        # max_time = (num_cabbages + num_raddishes + num_glasses)*10
    )

    """
    parser = setup_parser()
    args = parser.parse_args()
    mode = args.mode.lower()
    run_kitchen(
        num_cabbages=2,
        num_raddishes= 2,
        num_glasses = 2,
        url = args.url
    )
    """
