import os
import pydrake.all


def find_resource(filename):
    """
    Returns the full path to this package appended to the relative path provided in filename (starting from this directory)
    """
    return os.path.join(os.path.dirname(__file__), filename)


def add_package_paths(parser):
    """
    Adds all package paths to the parser, starting from this directory and crawling downwards. Also adds the manipulation_station models packages.
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
    q_initial=[0.0, 0.1, 0, -1.2, 0, 1.6, 0],
    X_WB=pydrake.math.RigidTransform(),
):
    """
    Adds a panda arm to the multibody plant `plant` in the initial joint positions `q_initial` with the base welded to the world frame with relative pose `X_WB`. Returns the
    model instance that has been added to the plant
    """

    urdf_file = pydrake.common.FindResourceOrThrow(
        "drake/manipulation/models/franka_description/urdf/panda_arm.urdf"
    )

    parser = pydrake.multibody.parsing.Parser(plant)
    panda_model_instance = parser.AddModelFromFile(urdf_file)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0"), X_WB)

    index = 0
    for joint_index in plant.GetJointIndices(panda_model_instance):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, pydrake.multibody.tree.RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return panda_model_instance
