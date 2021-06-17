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
            ("pose", item, X_wrapper),
            ("atpose", item, X_wrapper),
        ]

    arms = parse_config(main_station, main_station_context)
    for arm, conf in arms.items():
        init += [("arm", arm), ("empty", arm), ("conf", conf), ("at", arm, conf)]


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
        ("cooked", "cabbage1"),
    ]

    cached_place_confs = {}
    
    def plan_motion_gen(start, end, fluents = []):
        """
        Yields a collision free trajectory from <start> to <end> 
        while the arm is not holding anything.
        Returns if no trajectory could be found

        Args:
            start: np.array of length 7, starting arm configuration
            end: np.array of length 7, ending arm configuration
            fluents: a list of tuples of the form
            ("atpose", <item>, <X_WO>) defining the current pose of 
            all items, where <item> is a string with the item name
            and <X_WO> is a RigidTransformWrapper with the world pose
            of the objects
        """
        start = start.copy()
        end = end.copy()
        fluents = copy.deepcopy(fluents)
        print(f"{Colors.BLUE}Starting move-free stream{Colors.RESET}")
        station = stations["move_free"]
        station_context = station_contexts["move_free"]
        update_station(station, station_context, fluents)
        while True:
            print(f"{Colors.GREEN}Planning free trajectory{Colors.RESET}")
            traj = find_traj(
                station, 
                station_context, 
                start, 
                end, 
                ignore_endpoint_collisions= True
            )
            if traj is None:  # if a trajectory could not be found (invalid)
                print(f"{Colors.RED}Closing move-free stream{Colors.RESET}")
                return
            print(f"{Colors.REVERSE}Yielding free trajectory{Colors.RESET}")
            yield traj,
            update_station(station, station_context, fluents)

    def plan_motion_holding_gen(item, start, end, fluents=[]):
        """
        Yields a collision free trajectory from <start> to <end>
        while the arm is holding item <item>.
        Returns if no trajectory could be found

        Args:
            item: the name of the item that the arm is holding
            start: np.array of length 7, starting arm configuration
            end: np.array of length 7, ending arm configuration
            fluents: a list of tuples of the form
            ("atpose", <item>, <X_WO>) defining the current pose of 
            all items, where <item> is a string with the item name
            and <X_WO> is a RigidTransformWrapper with the world pose
            of the objects. One tuple has the form
            ("grasppose", <item>, <X_HO>), where H is the hand frame.
        """
        start = start.copy()
        end = end.copy()
        fluents = copy.deepcopy(fluents)
        print(f"{Colors.BLUE}Starting move-holding stream for {item}{Colors.RESET}")
        if not (item in stations):
            stations[item] = problem_info.make_holding_station(
                name = item,
            )
            station_contexts[item] = stations[item].CreateDefaultContext()
        station = stations[item]
        station_context = station_contexts[item]
        # udate poses in station
        update_station(station, station_context, fluents)
        while True:
            # find traj will return a np.array of configurations, but no time informatino
            # The actual peicewise polynominal traj will be reconstructed after planning
            print(f"{Colors.GREEN}Planning trajectory holding {item}{Colors.RESET}")
            traj = find_traj(
                station, 
                station_context, 
                start, 
                end, 
                ignore_endpoint_collisions= False
            )
            if traj is None:  # if a trajectory could not be found (invalid)
                print(f"{Colors.RED}Closing move-holding stream for {item}{Colors.RESET}")
                return
            print(f"{Colors.REVERSE}Yielding trajectory holding {item}{Colors.RESET}")
            yield (traj,)
            update_station(station, station_context, fluents)

    def plan_grasp_gen(item, X_WO):
        """
        Find a grasp configuration for the item <item>

        Args:
            item: the name of the item to be grabbed
            X_WO: the world pose of the item (RigidTransformWrapper)

        Yields:
            A tuple of the form:
            (<X_HO>, <grasp_q>, <pregrasp_q>, <postgrasp_q>)
        """
        print(f"{Colors.BLUE}Starting grasp stream for {item}{Colors.RESET}")
        station = stations["move_free"]
        station_context = station_contexts["move_free"]
        # udate poses in station
        update_station(
            station, station_context, [("atpose", item, X_WO)], set_others_to_inf=True
        )
        object_info = station.object_infos[item][0]
        shape_infos = update_graspable_shapes(object_info)
        iter = 1
        q_initial = Q_NOMINAL
        q_nominal = Q_NOMINAL
        while True:
            start_time = time.time()
            if item in cached_place_confs:
                if X_WO in cached_place_confs[item]:
                    print(f"{Colors.BOLD}FOUND CACHED CONF{Colors.RESET}")
                    grasp_q, postgrasp_q, pregrasp_q = cached_place_confs[item][X_WO]
            else:
                print(f"{Colors.GREEN}Finding grasp for {item}, iteration {iter}{Colors.RESET}")
                grasp_q, cost = None, np.inf
                #how many times will we try before saying it can't be done
                max_tries, tries = 5, 0
                while tries < max_tries:
                    tries += 1
                    print(f"{Colors.BOLD}{item} grasp tries: {tries}{Colors.RESET}")
                    grasp_q, cost = best_grasp_for_shapes(
                        station,
                        station_context,
                        shape_infos,
                        initial_guess = q_initial,
                        q_nominal = q_nominal
                    )
                    if np.isfinite(cost):
                        break
                    q_nominal = random_normal_q(station, Q_NOMINAL)
                    q_initial = random_normal_q(station, Q_NOMINAL)
                if not np.isfinite(cost):
                    print(f"{Colors.RED}Ending grasp stream for{item}{Colors.RESET}")
                    return
                pregrasp_q, postgrasp_q = pre_and_post_grasps(
                    station, 
                    station_context, 
                    grasp_q,
                    dist = 0.07
                )
            X_HO = RigidTransformWrapper(
                q_to_X_HO(
                    grasp_q, 
                    object_info.main_body_info,
                    station,
                    station_context
                ),
                name = f"X_H{item}"
            )
            print(
                f"{Colors.REVERSE}Yielding grasp for {item} in {(time.time() - start_time):.4f} s{Colors.RESET}"
            )
            iter += 1 
            yield X_HO, grasp_q, pregrasp_q, postgrasp_q
            q_initial = random_normal_q(station, Q_NOMINAL)
            q_nominal = random_normal_q(station, Q_NOMINAL)
            update_station(
                station, station_context, [("atpose", item, X_WO)], set_others_to_inf=True
            )

    def plan_place_gen(item, region, X_HO):
        """
        Find an arm configuration for placing <item> in <region>

        Args:
            item: the name of the item to be grabbed
            region: The region the object should be placed
            X_HO: the hand pose of the item (RigidTransformWrapper)

        Yields:
            A tuple of the form:
            (<X_WO>, <place_q>, <preplace_q>, <postplace_q>)
        """
        object_name, link_name = region
        print(f"{Colors.BLUE}Starting place stream for {item} on region {object_name}, {link_name}{Colors.RESET}")
        if not (item in stations):
            stations[item] = problem_info.make_holding_station(
                name = item
            )
            station_contexts[item] = stations[item].CreateDefaultContext()
        station = stations[item]
        station_context = station_contexts[item]
        # udate poses in station
        update_station(
            station,
            station_context,
            [("atgrasppose", item, X_HO)],
            set_others_to_inf=True,
        )
        target_object_info = station.object_infos[object_name][0]
        holding_object_info = station.object_infos[item][0]
        shape_infos = update_placeable_shapes(holding_object_info)
        surfaces = update_surfaces(target_object_info, link_name, station, station_context)
        q_initial = Q_NOMINAL
        while True:
            print(f"{Colors.GREEN}Finding place for {item}{Colors.RESET}")
            start_time = time.time()
            place_q, cost = None, np.inf
            #how many times will we try before saying it can't be done
            max_tries, tries = 10, 0
            while tries < max_tries:
                tries += 1
                print(f"{Colors.BOLD}{item} place tries: {tries}{Colors.RESET}")
                place_q, cost = best_place_shapes_surfaces(
                    station,
                    station_context,
                    shape_infos,
                    surfaces
                )
                if np.isfinite(cost):
                    break
            if not np.isfinite(cost):
                print(f"{Colors.RED}Ending place stream for{item} on region {region}{Colors.RESET}")
                return
            # pregrasp_q == postplace_q, postgrasp_q == preplace_q
            postplace_q, preplace_q = pre_and_post_grasps(
                station, 
                station_context, 
                place_q,
                dist = 0.07
            )
            X_WO = RigidTransformWrapper(
                q_to_X_PF(
                    place_q,
                    station.get_multibody_plant().world_frame(),
                    holding_object_info.main_body_info.get_body_frame(),
                    station,
                    station_context,
                ),
                name = f"X_W{item}"
            )
            print(
                f"{Colors.REVERSE}Yielding placement for {item} in {(time.time() - start_time):.4f} s{Colors.RESET}"
            )
            if not item in cached_place_confs:
                cached_place_confs[item] = {}
            cached_place_confs[item][X_WO] = (place_q, preplace_q, postplace_q)
            yield X_WO, place_q, preplace_q, postplace_q
            update_station(
                station,
                station_context,
                [("atgrasppose", item, X_HO)],
                set_others_to_inf=True,
            )

    stream_map = {
        "plan-motion-free": from_gen_fn(plan_motion_gen),
        "plan-motion-holding": from_gen_fn(plan_motion_holding_gen),
        "grasp-conf": from_gen_fn(plan_grasp_gen),
        "placement-conf": from_gen_fn(plan_place_gen),
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
