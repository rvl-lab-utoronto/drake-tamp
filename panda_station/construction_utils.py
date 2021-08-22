"""
Utilities for constructing a multibody plant in drake used for TAMP experiments
"""
import os
import pydrake.all
import numpy as np


def find_resource(filename, parent_path = None):
    """
    Returns the full path to this package appended to the relative path provided
    in filename (starting from this directory)

    Args:
        filename: the path (string) that is appended to the absolute path of
        this file location
    Returns:
        the concatenated path
    """
    return os.path.join(os.path.dirname(__file__), filename)


def add_package_paths(parser):
    """
    Adds all package paths to the parser, starting from this directory and
    crawling downwards. Also adds the manipulation_station models packages.

    Args:
        parser: The pydrake.multibody.parsing.Parser to add the packages to
    """
    parser.package_map().PopulateFromFolder(find_resource(""))
    parser.package_map().Add(
        "manipulation_station",
        os.path.join(
            pydrake.common.GetDrakePath(), "examples/manipulation_station/models"
        ),
    )


def add_panda(
    plant,
    q_initial=np.array([0.0, 0.1, 0, -1.2, 0, 1.6, np.pi/4]),
    X_WB=pydrake.math.RigidTransform(),
    name = "panda"
):
    """
    Adds a panda arm to the multibody plant `plant` in the initial joint positions
    `q_initial` with the base welded to the world frame with relative
    pose `X_WB`. Returns the model instance that has been added to the plant

    Args:
        plant: the pydrake.multibody.plant to add the panda to
        q_inital: the initial joint positions of the panda (np.array[float])
        X_WB: the pydrake.math.RigidTransform between the world and the base of
        the panda arm (panda_link0)
    Returns:
        the pydrake.multibody.tree.ModelInstanceIndex of the panda model that was added to the plant
    """

    urdf_file = pydrake.common.FindResourceOrThrow(
        "drake/manipulation/models/franka_description/urdf/panda_arm.urdf"
    )

    parser = pydrake.multibody.parsing.Parser(plant)
    model_index = parser.AddModelFromFile(urdf_file, name)
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_link0", model_index),
        X_WB
    )

    index = 0
    for joint_index in plant.GetJointIndices(model_index):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, pydrake.multibody.tree.RevoluteJoint):
            joint.set_default_angle(q_initial[index])
            index += 1

    return model_index

def add_panda_hand(
    plant,
    panda_model_instance_index=None,
    roll=-np.pi/4,
    weld_fingers=False,
    blocked = False,
    name = "hand"
):
    """
    Adds a panda hand to the multibody plant `plant`. If
    `panda_model_instance_index` is supplied, it will weld the panda hand
    to the link `panda_link8` with roll `roll`.

    Args:
        plant: the pydrake.multibody.plant to add the panda hand to
        panda_model_instance_index : the pydrake.multibody.tree.ModelInstanceIndex
        to weld the panda hand to
        roll: the roll (float) of the rigid transform between `panda_link8`
        and the panda hand
        weld_fingers: (boolean) if true, the panda's fingers are welded in the open
        position (useful for inverse kinematics)
    Returns:
        the pydrake.multibody.tree.ModelInstanceIndex of the added panda hand
    """
    parser = pydrake.multibody.parsing.Parser(plant)

    if weld_fingers:
        model_index = parser.AddModelFromFile(
            find_resource("models/modified_panda_hand/sdf/welded_panda_hand.sdf"),
            name
        )
        assert not blocked, "The hand can't be both welded and blocked"
    elif blocked:
        model_index = parser.AddModelFromFile(
            find_resource("models/modified_panda_hand/sdf/blocked_panda_hand.sdf"),
            name
        )
    else:
        model_index = parser.AddModelFromFile(
            find_resource("models/modified_panda_hand/sdf/panda_hand.sdf"),
            name
        )

    if panda_model_instance_index is not None:
        X_8G = pydrake.math.RigidTransform(
            pydrake.math.RollPitchYaw(0, 0, roll), [0, 0, 0]
        )
        plant.WeldFrames(
            plant.GetFrameByName("panda_link8", panda_model_instance_index),
            plant.GetFrameByName("panda_hand", model_index),
            X_8G,
        )
    return model_index
