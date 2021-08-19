import pydrake.all
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
            # role = Role.kProximity
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

def preview_problem(file_path, url = "tcp://127.0.0.1:6000"):
    make_and_init_simulation(url, file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", default = "./test_problem.yaml")
    parser.add_argument("-u", "--url", default = "tcp://127.0.0.1:6000")

    args = parser.parse_args()

    preview_problem(args.file_path, args.url)

