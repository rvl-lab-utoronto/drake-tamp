"""
The module for running the pick and and place TAMP problem
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
from panda_station import (
    ProblemInfo,
    parse_start_poses,
    parse_config,
    update_station,
    TrajectoryDirector,
    find_traj,
    q_to_X_HO,
    best_place_shapes_surfaces,
    q_to_X_PF,
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

domain_pddl = """(define (domain pickplaceregions)
    (:requirements :strips)
    (:predicates
        ; discret types
        (item ?item) ; item name
        (arm ?a) ; robotarm (in case we have more than one)
        (region ?r) ; e.g. table, red, green

        ; continuous types
        (pose ?item ?pose) ; a valid pose of an item
        (relpose ?item ?grasppose); a pose relative to the hand
        (conf ?conf) ; robot configuration
        (contained ?item ?region ?pose) ; if ?item were at ?pose, it would be inside ?region

        (grasp ?item ?pose ?grasppose ?graspconf ?pregraspconf ?postgraspconf)
        (place ?item ?region ?grasppose ?placementpose ?placeconf ?preplaceconf ?postplaceconf)
        (mftraj ?traj ?start ?end)
        (mhtraj ?item ?startconf ?endconf ?traj)

        ;fluents
        (at ?arm ?conf)
        (empty ?arm)
        (grasped ?arm ?item)
        (atpose ?item ?pose)
        (atgrasppose ?item ?grasppose)

        ; derived
        (in ?item ?region)
    )
    (:action move-free
        :parameters (?arm ?start ?end ?t)
        :precondition (and
            (arm ?arm)
            (conf ?start)
            (conf ?end)
            (empty ?arm)
            (at ?arm ?start)
            (mftraj ?t ?start ?end)
        )
        :effect (and
            (not (at ?arm ?start)) (at ?arm ?end))
    )
    (:action pick
        :parameters (?arm ?item ?pose ?grasppose ?graspconf ?pregraspconf ?postgraspconf) ; grasppose is X_Hand_Item
        :precondition (and
            (arm ?arm)
            (item ?item)
            (conf ?pregraspconf)
            (conf ?postgraspconf)
            (conf ?graspconf)
            (pose ?item ?pose)
            (grasp ?item ?pose ?grasppose ?graspconf ?pregraspconf ?postgraspconf)

            (empty ?arm)
            (atpose ?item ?pose)
            (at ?arm ?pregraspconf)
        )
        :effect (and
            (not (empty ?arm))
            (not (at ?arm ?pregraspconf))
            (not (atpose ?item ?pose))

            (grasped ?arm ?item)
            (at ?arm ?postgraspconf)
            (atgrasppose ?item ?grasppose)
        )
    )
    (:action move-holding
        :parameters (?arm ?item ?startconf ?endconf ?t)
        :precondition (and
            (arm ?arm)
            (item ?item)
            (conf ?startconf)
            (conf ?endconf)
            (at ?arm ?startconf)
            (grasped ?arm ?item)
            (mhtraj ?item ?startconf ?endconf ?t)
        )
        :effect (and
            (not (at ?arm ?startconf))
            (at ?arm ?endconf)
        )
    )
    (:action place
        :parameters (?arm ?item ?region ?grasppose ?placepose ?placeconf ?preplaceconf ?postplaceconf)
        :precondition (and
            (arm ?arm)
            (item ?item)
            (region ?region)
            (pose ?item ?placepose)
            (conf ?preplaceconf)
            (conf ?postplaceconf)
            (conf ?placeconf)
            (place ?item ?region ?grasppose ?placepose ?placeconf ?preplaceconf ?postplaceconf)
            

            (at ?arm ?preplaceconf)
            (grasped ?arm ?item)
        )
        :effect (and
            (not (grasped ?arm ?item))
            (not (at ?arm ?preplaceconf))
            (not (atgrasppose ?item ?grasppose))

            (empty ?arm)
            (at ?arm ?preplaceconf)
            (atpose ?item ?placepose)
        )
    )
    (:derived (in ?item ?region)
        (exists
            (?pose)
            (and
                (pose ?item ?pose)
                (region ?region)
                (item ?item)
                (contained ?item ?region ?pose)
                (atpose ?item ?pose)
            )
        )
    )
)"""
stream_pddl = """(define (stream example)
  (:stream plan-motion-free
    :inputs (?start ?end)
    :fluents (atpose)
    :domain (and
        (conf ?start)
        (conf ?end)
    )
    :outputs (?t)
    :certified (and
        (mftraj ?t ?start ?end)
    )
  )
  (:stream plan-motion-holding
    :inputs (?item ?start ?end)
    :domain (and
        (conf ?start)
        (conf ?end)
        (item ?item)
    )
    :fluents (atpose atgrasppose)
    :outputs (?t)
    :certified (and
        (mhtraj ?item ?start ?end ?t)
    )
  )
  (:stream grasp-conf
    :inputs (?item ?pose)
    :outputs (?grasppose ?graspconf ?pregraspconf ?postgraspconf)
    :domain (and
        (item ?item)
        (pose ?item ?pose)
    )
    :certified (and
        (conf ?pregraspconf)
        (conf ?postgraspconf)
        (conf ?graspconf)
        (relpose ?item ?grasppose)
        (grasp ?item ?pose ?grasppose ?graspconf ?pregraspconf ?postgraspconf)
    )
  )
  (:stream placement-conf
    :inputs (?item ?region ?grasppose)
    :outputs (?placementpose ?placeconf ?preplaceconf ?postplaceconf)
    :domain (and
        (item ?item)
        (region ?region)
        (relpose ?item ?grasppose)
    )
    :certified (and
        (pose ?item ?placementpose)
        (conf ?preplaceconf)
        (conf ?postplaceconf)
        (conf ?placeconf)
        (contained ?item ?region ?placementpose)
        (place ?item ?region ?grasppose ?placementpose ?placeconf ?preplaceconf ?postplaceconf)
    )
  )
)"""


def construct_problem_from_sim(simulator, stations, problem_info):

    init = []
    main_station = stations["main"]
    simulator_context = simulator.get_context()
    main_station_context = main_station.GetMyContextFromRoot(simulator_context)
    station_contexts = {}
    for name in stations:
        station_contexts[name] = stations[name].CreateDefaultContext()
    start_poses = parse_start_poses(main_station, main_station_context)
    for item, pose in start_poses.items():
        X_wrapper = RigidTransformWrapper(pose, name = f"X_W{item}")
        init += [
            ("item", item),
            ("pose", item, X_wrapper),
            ("atpose", item, X_wrapper),
        ]
        # TODO : add any "contained" predicates that are true of the initial poses

    arms = parse_config(main_station, main_station_context)
    for arm, conf in arms.items():
        init += [("arm", arm), ("empty", arm), ("conf", conf), ("at", arm, conf)]

    for object_name in problem_info.surfaces:
        for link_name in problem_info.surfaces[object_name]:
            init += [("region", (object_name, link_name))]

    goal = (
        "and",
        ("in", "mustard", ("table_square", "base_link")),
    )

    def plan_motion_gen(start, end, fluents=[]):
        """
        Takes two 7DOF configurations, and yields collision free trajector(ies)
        between them.
        Fluents is a list of tuples of the type ('atpose', <item>, <pose>)
        defining the current pose of all items.
        """
        start = start.copy()
        end = end.copy()
        fluents = copy.deepcopy(fluents)
        print(f"{Colors.BLUE}Starting move-free stream{Colors.RESET}")
        station = stations["move_free"]
        station_context = station_contexts["move_free"]
        # udate poses in station
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
            yield (traj,)
            update_station(station, station_context, fluents)

    def plan_motion_holding_gen(item, start, end, fluents=[]):
        """
        Takes an item name, and two 7DOF configurations.
        Yields collision free trajectories between the 7DOF configurations
        considering that the named item is grasped.
        Fluents is a list of tuples of the type ('atpose', <item>, <pose>)
        and ('atgrasppose', <item>, <pose>)
        defining the current pose of all items.
        """
        start = start.copy()
        end = end.copy()
        fluents = copy.deepcopy(fluents)
        print(f"{Colors.BLUE}Starting move-holding stream for {item}{Colors.RESET}")
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

    def plan_grasp_gen(item, pose):
        """
        Takes an item name and the corresponding SE(3) pose. (RigidTransformWrapper)
        Yields tuples of the form (<grasppose>, <grasp_conf>, <pregrasp_conf>,
        <postgrasp_conf>) representing a relative pose of the
        the item to the hand after the grasp, the grasping configuration,
        and two valid robot arm configurations for pre/post grasp stages of a
        two stage grasp when the <item> is at <pose>.
        """
        print(f"{Colors.BLUE}Starting grasp stream for {item}{Colors.RESET}")
        station = stations["move_free"]
        station_context = station_contexts["move_free"]
        # udate poses in station
        update_station(
            station, station_context, [("atpose", item, pose)], set_others_to_inf=True
        )
        object_info = station.object_infos[item][0]
        shape_infos = update_graspable_shapes(object_info)
        iter = 1
        q_initial = Q_NOMINAL
        q_nominal = Q_NOMINAL
        while True:
            print(f"{Colors.GREEN}Finding grasp for {item}, iteration {iter}{Colors.RESET}")
            start_time = time.time()
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
                station, station_context, [("atpose", item, pose)], set_others_to_inf=True
            )

    def plan_place_gen(item, region, X_HO):
        """
        Takes an item name and a region name.
        Yields tuples of the form (<X_WO>, <place_conf>, <preplace_conf>,
        <postplace_conf>) representing the pose of <item> after
        being placed in <region>, and pre/post place robot arm configurations.
        """
        object_name, link_name = region
        print(f"{Colors.BLUE}Starting place stream for {item} on region {object_name}, {link_name}{Colors.RESET}")
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
    return simulator, stations, director, meshcat, problem_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Use --url to specify the zmq_url for a meshcat server\nuse --problem to specify a .yaml problem file"
    )
    parser.add_argument("-u", "--url", nargs="?", default=None)
    parser.add_argument(
        "-p", "--problem", nargs="?", default="problems/test_problem.yaml"
    )
    args = parser.parse_args()
    sim, station_dict, traj_director, meshcat_vis, problem_info = make_and_init_simulation(
        args.url, args.problem
    )
    problem = construct_problem_from_sim(sim, station_dict, problem_info)

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

        action_map = {
            "move-holding": (
                plan_to_trajectory.move,
                [4],
            ),
            "move-free": (
                plan_to_trajectory.move,
                [3],
            ),
            "pick":(
                plan_to_trajectory.pick,
                [5, 4, 6]
            ),
            "place":(
                plan_to_trajectory.place,
                [6, 5, 7]
            )
        }

        plan_to_trajectory.make_trajectory(
            plan, traj_director, SIM_INIT_TIME, action_map
        )

        sim.AdvanceTo(traj_director.get_end_time())
        if meshcat_vis is not None:
            meshcat_vis.stop_recording()
            meshcat_vis.publish_recording()

        save = input(
            """
        Type ENTER to exit without saving. 
        To save the video to the file
        media/<filename.html>, input <filename>\n
        """
        )
        if save != "":
            if not os.path.isdir("media"):
                os.mkdir("media")
            file = open("media/" + save + ".html", "w")
            file.write(meshcat_vis.vis.static_html())
            file.close()
