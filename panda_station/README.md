# Panda Station Package

This package provides much of the Drake functionality for TAMP with the franka-panda

## Folders and Files

`models/` contains all nessecary models and meshes that are not part of the drake binary installation.

`directives/` contains `yaml` files with directives for Drake to parse into the multibody plant

`construction_utils.py` has drake utility functions to help with simulation construction
    - `add_panda`: add the panda arm to the plant
    - `add_panda_hand`: add a panda hand the the plant, and attack it to the arm 
    - `add_package_paths(parser)`: adds package paths to parser so it can find models
    - `find_resource(filename)`: adds `filename` to the path are returns the path to that file
    - TODO(agro): `add_rgbd_sensors` (if needed)

`panda_station.py` contains the code for the `PandaStation` class, a Drake `diagram` used for setting up the panda arm. It is largely based on Drake's [manipulation station](https://github.com/RobotLocomotion/drake/tree/master/examples/manipulation_station).
    - TODO(agro): list functions here 

