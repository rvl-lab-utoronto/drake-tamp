#!/usr/bin/env python3

import numpy as np
import pydrake.all
from pydrake.all import (
    DiagramBuilder, 
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    ConnectMeshcatVisualizer,
    InverseKinematics,
    Solve
)
import json
from panda_station import (
    PandaStation,
    parse_tables,
    find_resource,
    PlanToTrajectory,
)

DIRECTIVE = "directives/basement.yaml"
NAMES_AND_LINKS = [
    (name, "base_link") for name in parse_tables(find_resource(DIRECTIVE))
] 
HAND_HEIGHT = 0.125
Q_NOMINAL = np.array([0.0, 0.55, 0.0, -1.45, 0.0, 1.58, np.pi/4])
INITIAL_GUESS = np.array([0,  1.04773756,  0, -1.41045257, 0, 2.42677142,  0.785398163])

# position of wooden table (top, geometric center) in world frame, in sim
p_WT = np.array([0.755, 0, 0.74])

def X_WH_to_q(
    X_WH,
    station,
    station_context,
    initial_guess = INITIAL_GUESS,
    p_tol = 0,
    theta_tol = 0,
    col_margin = 0,
):
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)

    hand = station.get_hand()
    panda = station.get_panda()
    plant.SetPositions(plant_context, panda, initial_guess)
    initial_guess = plant.GetPositions(plant_context)
    H = plant.GetFrameByName("panda_hand", hand)
    W = plant.world_frame()
    ik = InverseKinematics(plant, plant_context)

    ik.AddPositionConstraint(
        H,
        np.zeros(3),
        W,
        X_WH.translation() - p_tol * np.ones(3),
        X_WH.translation() + p_tol * np.ones(3),
    )
    ik.AddOrientationConstraint(H, RotationMatrix(), W, X_WH.rotation(), theta_tol)
    ik.AddMinimumDistanceConstraint(col_margin, 0.1)
    q = ik.q()
    prog = ik.prog()
    #prog.AddQuadraticErrorCost(np.identity(len(q)), q_nominal, q)
    prog.SetInitialGuess(q, initial_guess)
    result = Solve(prog)
    cost = result.get_optimal_cost()
    if not result.is_success():
        cost = np.inf
    return plant.GetPositions(plant_context, panda), cost


if __name__ == "__main__":
    builder = DiagramBuilder()
    station = PandaStation(name = "placement_station")
    station.setup_from_file(DIRECTIVE, names_and_links=NAMES_AND_LINKS)
    station.add_panda_with_hand(
        weld_fingers=True, X_WB = RigidTransform(RotationMatrix(), [0.05, 0, 0.8])
    )
    plant = station.get_multibody_plant()
    station.finalize()
    builder.AddSystem(station)
    scene_graph = station.get_scene_graph()
    zmq_url = "tcp://127.0.0.1:6002"
    v = ConnectMeshcatVisualizer(
        builder,
        scene_graph,
        output_port=station.GetOutputPort("query_object"),
        delete_prefix_on_load=True,
        zmq_url=zmq_url,
    )
    v.load()
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    station_context = station.GetMyContextFromRoot(diagram_context)
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    diagram.Publish(diagram_context)

    initial_guess = Q_NOMINAL#INITIAL_GUESS
    confs = {}
    tot = 0
    suc = 0
    for x in np.arange(-0.3, 0.1, 0.05):
        for y in np.arange(-0.4, 0.4, 0.05):
            print(x, y)
            p = p_WT + np.array([x,y,HAND_HEIGHT])
            print(p[0], p[1])
            if np.linalg.norm(p[:2]) > 0.855:
                continue
            X_WH1 = RigidTransform(
                RotationMatrix(RollPitchYaw(np.pi, 0, 0)),
                p
            )
            q, cost = X_WH_to_q(
                X_WH1,
                station,
                station_context,
                p_tol = 1e-4,
                theta_tol = 0
            )
            if np.isfinite(cost):
                confs[(x,y)] = list(q)
                suc+=1
            tot+=1
            print(q, cost)
            plant.SetPositions(plant_context, station.get_panda(), q)
            diagram.Publish(diagram_context)
            #input("Press enter to see next")
    print(f"Success Rate: {suc/tot}")
    #with open("confs.json", "w") as f:
        #json.dump(confs, f, indent = 4, sort_keys = True)
    f = open("calibration_confs.txt", "w")
    for pos, conf in confs.items():
        f.write(str(np.round(pos[0],2)) + ", " + str(np.round(pos[1],2)) + "\n")
        f.write(PlanToTrajectory.numpy_conf_to_str(conf))
    f.close()


    