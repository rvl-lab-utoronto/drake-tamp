# Panda Station Package

This package provides much of the Drake functionality for TAMP with the franka-panda

## Folders and Files

`models/` contains all nessecary models and meshes that are not part of the drake binary installation.

`directives/` contains `yaml` files with directives for Drake to parse into the multibody plant

`test/` contains unit and integration tests for this package

`construction_utils.py` has drake utility functions to help with simulation construction
    - `add_panda`: add the panda arm to the plant
    - `add_panda_hand`: add a panda hand the the plant, and attack it to the arm 
    - `add_package_paths(parser)`: adds package paths to parser so it can find models
    - `find_resource(filename)`: adds `filename` to the path are returns the path to that file
    - TODO(agro): `add_rgbd_sensors` (if needed)

`panda_station.py` contains the code for the `PandaStation` class, a Drake `diagram` used for setting up the panda arm. It is largely based on Drake's [manipulation station](https://github.com/RobotLocomotion/drake/tree/master/examples/manipulation_station).
    - TODO(agro): list functions here 

`panda_hand_position_controller.py` contains the PandaHandPositionController class for controlling the panda hand by joint positions. It is based on the code for the `schunk_wsg_position_controller` supplied with drake ([here](https://github.com/RobotLocomotion/drake/blob/master/manipulation/schunk_wsg/schunk_wsg_position_controller.cc))

`planning_utils.py` contains utility functions nessecary for interfacing with PDDL and creating the *streams* (generators).

`stream_utils.py` has functions/generators used in TAMP: picking, placing, 
and collision free motion planning

`trajectory_director.py` has a simple system that sends trajectories to a PandaStation if they are available, otherwise it tells the panda to stay still
