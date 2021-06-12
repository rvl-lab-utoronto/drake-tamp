"""
The module for running the pick and and place TAMP problem
"""

import os
import argparse
from panda_station.grasping_and_placing import is_placeable
import numpy as np
import pydrake.all
from pydrake.all import RigidTransform
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import str_from_object
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.algorithms.meta import solve
from panda_station import (
    PandaStation,
    find_resource,
    ProblemInfo,
    parse_start_poses,
    parse_config,
    parse_tables,
    update_station,
    TrajectoryDirector,
    find_traj,
    find_grasp_q,
    q_to_X_HO,
    backup_on_hand_z,
    backup_on_world_z,
    find_place_q,
    q_to_X_PF,
)

ARRAY = tuple

domain_pddl = """(define (domain pickplaceregions)
    (:requirements :strips)
    (:predicates
        ; discret types
        (item ?item) ; item name
        (arm ?a) ; robotarm (in case we have more than one)
        (region ?r) ; e.g. table, red, green

        ; continuous types
        (pose ?item ?pose) ; a valid pose of an item
        (conf ?conf) ; robot configuration
        (contained ?item ?region ?pose) ; if ?item were at ?pose, it would be inside ?region

        (grasp ?item ?pose ?grasppose ?pregraspconf ?postgraspconf)
        (place ?item ?region ?placementpose ?preplaceconf ?postplaceconf)
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
        :parameters (?arm ?item ?pose ?grasppose ?pregraspconf ?postgraspconf) ; grasppose is X_Hand_Item
        :precondition (and
            (arm ?arm)
            (item ?item)
            (conf ?pregraspconf)
            (conf ?postgraspconf)
            (pose ?item ?pose)
            (grasp ?item ?pose ?grasppose ?pregraspconf ?postgraspconf)

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
        :parameters (?arm ?item ?region ?grasppose ?placepose ?preplaceconf ?postplaceconf)
        :precondition (and
            (arm ?arm)
            (item ?item)
            (region ?region)
            (pose ?item ?placepose)
            (conf ?preplaceconf)
            (conf ?postplaceconf)
            (place ?item ?region ?placepose ?preplaceconf ?postplaceconf)
            

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
    :outputs (?grasppose ?pregraspconf ?postgraspconf)
    :domain (and
        (item ?item)
        (pose ?item ?pose)
    )
    :certified (and
        (conf ?pregraspconf)
        (conf ?postgraspconf)
        (grasp ?item ?pose ?grasppose ?pregraspconf ?postgraspconf)
    )
  )
  (:stream placement-conf
    :inputs (?item ?region)
    :outputs (?placementpose ?preplaceconf ?postplaceconf)
    :domain (and
        (item ?item)
        (region ?region)
    )
    :fluents (atgrasppose)
    :certified (and
        (pose ?item ?placementpose)
        (conf ?preplaceconf)
        (conf ?postplaceconf)
        (contained ?item ?region ?placementpose)
        (place ?item ?region ?placementpose ?preplaceconf ?postplaceconf)
    )
  )
)"""


def construct_problem_from_sim(simulator, stations):

    init = []
    main_station = stations["main"]
    simulator_context = simulator.get_context()
    main_station_context = main_station.GetMyContextFromRoot(simulator_context)
    station_contexts = {}
    for name in stations:
        station_contexts[name] = stations[name].CreateDefaultContext()
    start_poses = parse_start_poses(main_station, main_station_context)
    for k, v in start_poses.items():
        init += [("item", k), ("pose", k, v), ("atpose", k, v)]
        # TODO : add any "contained" predicates that are true of the initial poses

    arms = parse_config(main_station, main_station_context)
    for k, v in arms.items():
        init += [("arm", k), ("empty", k), ("conf", v), ("at", k, v)]

    regions = parse_tables(main_station.directive)
    for r in regions:
        init += [("region", r)]

    goal = (
        "and",
        ("in", "foam_brick", "table_round"),
    )

    def plan_motion_gen(start, end, fluents=[]):
        """
        Takes two 7DOF configurations, and yields collision free trajector(ies)
        between them.
        Fluents is a list of tuples of the type ('atpose', <item>, <pose>)
        defining the current pose of all items.
        """
        print("\nHERE\n")
        station = stations["move_free"]
        station_context = station_contexts["move_free"]
        # udate poses in station
        update_station(station, station_context, fluents)
        while True:
            # find traj will return a np.array of configurations, but no time informatino
            # The actual peicewise polynominal traj will be reconstructed after planning
            traj = find_traj(station, station_context, start, end)
            if traj is None:  # if a trajectory could not be found (invalid)
                return
            yield traj

    def plan_motion_holding_gen(item, start, end, fluents=[]):
        """
        Takes an item name, and two 7DOF configurations.
        Yields collision free trajectories between the 7DOF configurations
        considering that the named item is grasped.
        Fluents is a list of tuples of the type ('atpose', <item>, <pose>)
        defining the current pose of all items.
        """
        print("\nHERE\n")
        station = stations[item]
        station_context = station_contexts[item]
        # udate poses in station
        update_station(station, station_context, fluents)
        while True:
            # find traj will return a np.array of configurations, but no time informatino
            # The actual peicewise polynominal traj will be reconstructed after planning
            traj = find_traj(station, station_context, start, end)
            if traj is None:  # if a trajectory could not be found (invalid)
                return
            yield traj

    def plan_grasp_gen(item, pose):
        """
        Takes an item name and the corresponding SE(3) pose.
        Yields tuples of the form (<grasppose>, <pregrasp_conf>,
        <postgrasp_conf>) representing a relative pose of the
        the item to the hand after the grasp, and two valid robot
        arm configurations for pre/post grasp stages of a
        two stage grasp when the <item> is at <pose>.
        """
        station = stations["move_free"]
        station_context = station_contexts["move_free"]
        # udate poses in station
        mock_fluent = ("atpose", item, pose)
        update_station(station, station_context, [mock_fluent], set_others_to_inf=True)
        object_info = station.object_infos[item][0]
        body_infos = list(object_info.get_body_infos().values())
        for body_info in body_infos:
            for shape_info in body_info.get_shape_infos():
                # TODO(agro): implement find_grasp, pregrasp ...
                for grasp_q, cost in find_grasp_q(station, station_context, shape_info):
                    if not np.isfinite(cost):
                        continue
                    pregrasp_q, pre_cost = backup_on_hand_z(
                        grasp_q, station, station_context
                    )
                    postgrasp_q, post_cost = backup_on_world_z(
                        grasp_q, station, station_context
                    )
                    # relative transform from hand to main_body of object_info
                    X_HO = q_to_X_HO(
                        grasp_q, object_info.main_body_info, station, station_context
                    )
                    if not np.isfinite(pre_cost + post_cost):
                        continue
                    yield X_HO, pregrasp_q, postgrasp_q
        return

    def plan_place_gen(item, region, fluents=[]):
        """
        Takes an item name and a region name.
        Yields tuples of the form (<X_WO>, <preplace_conf>,
        <postplace_conf>) representing the pose of <item> after
        being placed in <region>, and pre/post place robot arm configurations.
        fluents contains a tuple [(atgrasppose, item, X_HO)]  where item is
        the name of the item, and X_HO is the relative transfomation between
        the hand and the object it is holding
        """
        station = stations[item]
        station_context = station_contexts[item]
        # udate poses in station
        update_station(station, station_context, fluents, set_others_to_inf=True)
        target_object_info = station.object_infos[region][0]
        holding_object_info = station.object_infos[item][0]
        W = station.get_multibody_plant().world_frame()

        for holding_body_info in holding_object_info.get_body_infos().values():
            for holding_shape_info in holding_body_info.get_shape_infos():
                if not is_placeable(holding_shape_info):
                    continue
                for target_body_info in target_object_info.get_body_infos().values():
                    for target_shape_info in target_body_info.get_shape_infos():
                        for place_q, cost in find_place_q(
                            station,
                            station_context,
                            holding_shape_info,
                            target_shape_info,
                        ):
                            if not np.isfinite(cost):
                                continue
                            postplace_q, pre_cost = backup_on_hand_z(
                                place_q, station, station_context
                            )
                            preplace_q, post_cost = backup_on_world_z(
                                place_q, station, station_context
                            )
                            # relative transform from hand to main_body of object_info
                            X_WO = q_to_X_PF(
                                place_q,
                                W,
                                holding_object_info.main_body_info.get_body_frame(),
                                station,
                                station_context,
                            )
                            if not np.isfinite(pre_cost + post_cost):
                                continue
                            yield X_WO, preplace_q, postplace_q

    stream_map = {
        "plan-motion-free": from_gen_fn(plan_motion_gen),
        "plan-motion-holding": from_gen_fn(plan_motion_holding_gen),
        "grasp-conf": from_gen_fn(plan_grasp_gen),
        "placement-conf": from_gen_fn(plan_place_gen),
    }

    return PDDLProblem(domain_pddl, {}, stream_pddl, stream_map, init, goal)


def make_and_init_simulation(zmq_url, problem):
    """
    Make the simulation, and let it run for 0.2 s to let all the objects
    settle into their stating positions
    """
    builder = pydrake.systems.framework.DiagramBuilder()
    problem_info = ProblemInfo(problem)
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
    simulator_context = simulator.get_context()

    if meshcat is not None:
        meshcat.start_recording()
    simulator.AdvanceTo(0.2)
    return simulator, stations, director, meshcat


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="zmq_url for meshcat server")
    parser.add_argument("-u", "--url", nargs="?", default=None)
    parser.add_argument(
        "-p", "--problem", nargs="?", default="problems/test_problem.yaml"
    )
    args = parser.parse_args()

    res = make_and_init_simulation(args.url, args.problem)
    director = res[2]
    meshcat = res[3]

    problem = construct_problem_from_sim(res[0], res[1])
    print("Initial:", str_from_object(problem.init))
    print("Goal:", str_from_object(problem.goal))
    for algorithm in [
        "binding",
    ]:
        solution = solve(problem, algorithm=algorithm, verbose=True)
        # print(f"\n\n{algorithm} solution:")
        # print_solution(solution)
        # input("Continue")
        plan, _, _ = solution
        for action in plan:
            print(action.name)
            print(action.args)

        if meshcat is not None:
            meshcat.stop_recording()
            meshcat.publish_recording()
