import os
import pydrake.all
import numpy as np
import argparse
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import str_from_object
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.algorithms.meta import solve
from panda_station import PandaStation, find_resource

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
    :certified (and
        (pose ?item ?placementpose)
        (conf ?preplaceconf)
        (conf ?postplaceconf)
        (contained ?item ?region ?placementpose)
        (place ?item ?region ?placementpose ?preplaceconf ?postplaceconf)
    )
  )
)"""


def construct_problem_from_sim(drake_sim):

    
    """
    TODO:
        

    """

    init = []
    start_poses = {  # TODO: replace these with SE(3) Poses from the scene
        "item1": "pose1",
        "item2": "pose2",
        "item3": "pose3",
    }
    for k, v in start_poses.items():
        init += [("item", k), ("pose", k, v), ("atpose", k, v)]
        # TODO : add any "contained" predicates that are true of the initial poses

    arms = {  # TODO: replace these with initial 7DOF conf
        "arm1": "conf1",
    }
    for k, v in arms.items():
        init += [("arm", k), ("empty", k), ("conf", v), ("at", k, v)]

    regions = ["table", "red", "green"]
    for r in regions:
        init += [("region", r)]

    goal = (
        "and",
        ("in", "item1", "red"),
    )

    # TODO: construct any models we need for the geometries of the scene, regions and items
    # so that they can be used by the following streams.

    def plan_motion_gen(start, end, fluents=[]):
        """
        Takes two 7DOF configurations, and yields collision free trajector(ies) between them.
        Fluents is a list of tuples of the type ('atpose', <item>, <pose>) defining the current pose of all items.
        """
        # TODO: add interal station with ALL object welded at the positions in fluents
        # TODO: call some collision free planning here
        while True:
            yield (f"free_traj_{start}_to_{end}",)

    def plan_motion_holding_gen(item, start, end, fluents=[]):
        """
        Takes an item name, and two 7DOF configurations.
        Yields collision free trajectories between the 7DOF configurations considering that the named item is grasped.
        Fluents is a list of tuples of the type ('atpose', <item>, <pose>) defining the current pose of all items.
        """
        #TODO: add internal station with ALL objects welded at the positions in fluents, and the object welded to the hand
        # TODO: update models and call some collision free planning here
        print("HERE")
        while True:
            yield (f"holding_traj_{item}_{start}_to_{end}",)

    def plan_grasp_gen(item, pose):
        """
        Takes an item name and the corresponding SE(3) pose.
        Yields tuples of the form (<grasppose>, <pregrasp_conf>, <postgrasp_conf>) representing a relative pose of the
        the item to the hand after the grasp, and two valid robot arm configurations for pre/post grasp stages of a
        two stage grasp when the <item> is at <pose>.
        """
        # TODO: plan a grasp
        while True:
            yield (f"{item}_grasppose", f"{item}_{pose}_pregrasp_conf", f"{item}_{pose}_postgrasp_conf")

    def plan_place_gen(item, region):
        """
        Takes an item name and a region name.
        Yields tuples of the form (<place_pose>, <preplace_conf>, <postplace_conf>) representing
        the pose of <item> after being placed in <region>, and pre/post place robot arm configurations.
        """
        # TODO: plan a place
        while True:
            yield (
                f"{item}_{region}_pose",
                f"{item}_{region}_preplace_conf",
                f"{item}_{region}_postplace_conf",
            )

    stream_map = {
        "plan-motion-free": from_gen_fn(plan_motion_gen),
        "plan-motion-holding": from_gen_fn(plan_motion_holding_gen),
        "grasp-conf": from_gen_fn(plan_grasp_gen),
        "placement-conf": from_gen_fn(plan_place_gen),
    }

    problem = PDDLProblem(domain_pddl, {}, stream_pddl, stream_map, init, goal)
    return problem


def make_panda_station(builder):
    """
    Takes in a diagram builder, adds a setup panda station to
    the diagram, and returns the panda station
    """
    station = builder.AddSystem(PandaStation())
    # setup 3 table environment
    station.setup_from_file("directives/three_tables.yaml")
    # add panda arm and hand in default config
    station.add_panda_with_hand()
    # add a brick, soup can, and meat manipulands
    station.add_model_from_file(
        find_resource("models/manipulands/sdf/foam_brick.sdf"),
        pydrake.math.RigidTransform(pydrake.math.RotationMatrix(), [0.6, 0, 0.2]),
    )
    station.add_model_from_file(
        find_resource("models/manipulands/sdf/meat_can.sdf"),
        pydrake.math.RigidTransform(
            pydrake.math.RotationMatrix.MakeXRotation(-np.pi / 2), [0.5, 0.2, 0.2]
        ),
    )
    station.add_model_from_file(
        find_resource("models/manipulands/sdf/soup_can.sdf"),
        pydrake.math.RigidTransform(
            pydrake.math.RotationMatrix.MakeXRotation(-np.pi / 2), [0.5, -0.2, 0.2]
        ),
    )

    station.finalize()
    return station

def make_and_init_simulation(zmq_url):
    """
    Make the simulation, and let it run for 0.2 s to let all the objects
    settle into their stating positions 
    """
    builder = pydrake.systems.framework.DiagramBuilder()
    station = make_panda_station(builder)
    plant, scene_graph = station.get_plant_and_scene_graph()

    if args.url is not None:
        meshcat = pydrake.systems.meshcat_visualizer.ConnectMeshcatVisualizer(
            builder,
            scene_graph,
            output_port=station.GetOutputPort("query_object"),
            delete_prefix_on_load=True,
            zmq_url=args.url,
        )
        meshcat.load()
    else:
        print("No meshcat server url provided, running without gui")

    diagram = builder.Build()
    simulator = pydrake.systems.analysis.Simulator(diagram)
    simulator_context = simulator.get_context()
    panda = station.get_panda()
    station_context = station.GetMyContextFromRoot(simulator_context)
    plant_context = plant.GetMyContextFromRoot(simulator_context)
    # for now, keep the arm still. TODO(agro): add trajectory system
    station.GetInputPort("panda_position").FixValue(station_context, plant.GetPositions(plant_context, panda))
    station.GetInputPort("hand_position").FixValue(station_context, [0.08])

    meshcat.start_recording()
    simulator.AdvanceTo(0.2)
    meshcat.stop_recording()
    meshcat.publish_recording()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="zmq_url for meshcat server")
    parser.add_argument("-u", "--url", nargs="?", default=None)
    args = parser.parse_args()
    
    simulator = make_and_init_simulation(args.url)

    problem = construct_problem_from_sim(simulator)
    print("Initial:", str_from_object(problem.init))
    print("Goal:", str_from_object(problem.goal))
    for algorithm in [
        "binding",
    ]:
        solution = solve(problem, algorithm=algorithm, verbose=True)
        #print(f"\n\n{algorithm} solution:")
        #print_solution(solution)
        #input("Continue")
        plan, _, _ = solution
        for action in plan:
            print(action.name)
            print(action.args)
