import pydrake.all
from pydrake.all import Role
import numpy as np
import streams
from panda_station import update_graspable_shapes
import argparse
from panda_station import ProblemInfo, TrajectoryDirector

SIM_INIT_TIME = 0

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
            role = Role.kProximity
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
    sim_context = simulator.get_context()


    # let objects fall for a bit in the simulation
    if meshcat is not None:
        meshcat.start_recording()
    simulator.AdvanceTo(SIM_INIT_TIME)

    
    station_contexts = {}
    for name in stations:
        station_contexts[name] = stations[name].CreateDefaultContext()

    block = "block1"
    station = stations["move_free"]
    station_context = station_contexts["move_free"]
    object_info = station.object_infos[block][0]
    shape_info = update_graspable_shapes(object_info)[0]
    grasp = streams.find_grasp(shape_info)
    print(grasp)
    q,cost = streams.find_ik_with_handpose(station, station_context, object_info, grasp, station.panda_infos[arm_name], relax = True)
    q,cost = streams.find_ik_with_handpose(station, station_context, object_info, grasp, station.panda_infos[arm_name], relax = False, q_initial=q)
    print(q, cost)

    main_station = stations["main"]
    main_station_context = main_station.GetMyContextFromRoot(sim_context)
    plant = main_station.get_multibody_plant()
    plant_context = main_station.GetSubsystemContext(plant, main_station_context)
    plant.SetPositions(plant_context, main_station.get_panda(), q)
    plant.SetPositions(plant_context, main_station.get_hand(), np.array([0.04, 0.04]))
    diagram_context = diagram.GetMyContextFromRoot(sim_context)
    diagram.Publish(diagram_context)

    query_output_port = scene_graph.GetOutputPort("query")
    scene_graph_context = main_station.GetSubsystemContext(scene_graph, main_station_context)
    query_object = query_output_port.Eval(scene_graph_context)
    pairs = query_object.ComputePointPairPenetration()
    print(plant.GetCollisionGeometriesForBody(plant.GetBodyByName("panda_hand")))
    print(plant.GetCollisionGeometriesForBody(plant.GetBodyByName("panda_leftfinger")))
    print(plant.GetCollisionGeometriesForBody(plant.GetBodyByName("panda_rightfinger")))
    for i in range(9):
        print(f'link {i}')
        print(plant.GetCollisionGeometriesForBody(plant.GetBodyByName(f"panda_link{i}")))
    for p in pairs:
        print(p.id_A, p.id_B)

def preview_problem(file_path, url = "tcp://127.0.0.1:6000"):
    make_and_init_simulation(url, file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", default = "../problems/test_problem.yaml")
    parser.add_argument("-u", "--url", default = "tcp://127.0.0.1:6004")

    args = parser.parse_args()

    preview_problem(args.file_path, args.url)

