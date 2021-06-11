"""
This module contains functions to assist with grasping and placing:

cylinder_grasp_q(station, station_context, shape_info)
box_grasp_q(station, station_context, shape_info)
sphere_grasp_q(station, station_context, shape_info)
"""
import numpy as np
from pydrake.all import (
    Solve,
    InverseKinematics,
    Box,
    Cylinder,
    Sphere,
)

NUM_Q = 7  # DOF of panda arm
GRASP_WIDTH = 0.08  # distance between fingers at max extension
GRASP_HEIGHT = 0.0535  # distance from hand to tips of fingers
FINGER_WIDTH = 0.017
# distance along z axis from hand frame origin to fingers
HAND_HEIGHT = 0.1
COL_MARGIN = 0.001  # acceptable margin of error for collisions
CONSIDER_MARGIN = 0.1
GRASP_MARGIN = 0.006  # margin for grasp planning
Q_NOMINAL = np.array([0.0, 0.55, 0.0, -1.45, 0.0, 1.58, 0.0])
HAND_FRAME_NAME = "panda_hand"
THETA_TOL = np.pi * 0.01


def check_specs(plant, q_nominal, initial_guess):
    """
    Ensures that plant has the correct number of positions,
    q_nominal is the right length,
    and inital_guess is the right length
    """
    assert NUM_Q == plant.num_positions(), "Too many positions in the plant"
    assert NUM_Q == len(q_nominal), "incorret length of q_nominal"
    assert NUM_Q == len(initial_guess), "incorret length of initial_guess"


def box_dim_from_axis(axis, box):
    """
    Returns the dimension of the box along the provided axis
    """
    if axis == 0:  # x
        return box.width()
    if axis == 1:  # y
        return box.depth()
    if axis == 2:  # z
        return box.height()
    raise Exception("Invalid box index")


def get_bounding_box(box):
    """
    Returns the coordinates of the lower and upper corners of the box
    (in the frame of the box)
    """
    assert isinstance(box, Box)
    size = box.size()
    return -size / 2, size / 2


def is_graspable(shape_info):
    """
    For a given ShapeInfo object, returns True if the object is
    graspable by the panda hand (not nessecarily reachable)
    """
    shape = shape_info.shape
    if isinstance(shape, Sphere):
        if (shape.radius() < GRASP_MARGIN) or (
            shape.radius() > GRASP_HEIGHT - GRASP_MARGIN
        ):
            return False
    if isinstance(shape, Cylinder):
        if (shape.radius() > GRASP_WIDTH - 2 * GRASP_MARGIN) and (
            shape.length() > GRASP_WIDTH - 2 * GRASP_MARGIN
        ):
            return False
    if isinstance(shape, Box):
        min_dim = min([shape.depth(), shape.width(), shape.height()])
        if min_dim > GRASP_WIDTH - 2 * GRASP_MARGIN:
            return False
    return True


def add_deviation_from_vertical_cost(prog, q, plant, weight=1):
    """
    Add a cost for the deviation of the z axis in the gripper frame
    from the -z axis in the world frame
    """
    plant_ad = plant.ToAutoDiffXd()
    plant_context_ad = plant_ad.CreateDefaultContext()

    def deviation_from_vertical_cost(q):
        plant_ad.SetPositions(plant_context_ad, q)
        hand_frame = plant_ad.GetFrameByName(HAND_FRAME_NAME)
        X_WH = hand_frame.CalcPoseInWorld(plant_context_ad)
        R_WH = X_WH.rotation()
        z_H = R_WH.matrix().dot(np.array([0, 0, 1]))  # extract the z direction
        return z_H.dot(np.array([0, 0, 1]))

    cost = lambda q: weight * deviation_from_vertical_cost(q)
    prog.AddCost(cost, q)


# TODO(ben): geometric center -> mass center
def add_deviation_from_box_center_cost(prog, q, plant, p_WC, weight=1):
    """
    Add a cost for the deviation of the y axis of the gripper
    (axis connecting fingers) from the box center
    """
    plant_ad = plant.ToAutoDiffXd()
    plant_context_ad = plant_ad.CreateDefaultContext()

    def deviation_from_box_center_cost(q):
        plant_ad.SetPositions(plant_context_ad, q)
        hand_frame = plant_ad.GetFrameByName("panda_hand")
        # C: center of box
        # H: hand frame
        # W: world frame
        # M: point in between fingers
        X_WH = hand_frame.CalcPoseInWorld(plant_context_ad)
        R_WH = X_WH.rotation()
        p_WH = X_WH.translation()
        p_HM_H = np.array([0, 0, 0.1])
        p_WM = p_WH + R_WH.multiply(p_HM_H)
        # we do not care about z (as much?) TODO(ben): look into this
        return ((p_WM - p_WC)[:-1]).dot((p_WM - p_WC)[:-1])

    cost = lambda q: weight * deviation_from_box_center_cost(q)
    prog.AddCost(cost, q)


def add_deviation_from_cylinder_middle_cost(prog, q, plant, G, weight=1):
    """
    Add a cost for the deviation of the point at the
    middle of the fingers from the cylinder center
    """
    plant_ad = plant.ToAutoDiffXd()
    plant_context_ad = plant_ad.CreateDefaultContext()
    G_ad = plant_ad.GetFrameByName(G.name())

    def deviation_from_cylinder_middle_cost(q):
        # H: hand frame
        # G: cylinder frame
        p_HC_H = [0, 0, HAND_HEIGHT]
        plant_ad.SetPositions(plant_context_ad, q)
        hand_frame = plant_ad.GetFrameByName(HAND_FRAME_NAME)
        X_GH_G = hand_frame.CalcPose(plant_context_ad, G_ad)
        R_GH = X_GH_G.rotation()
        p_GH_G = X_GH_G.translation()
        # distance from cylinder center to hand middle
        p_GC_G = p_GH_G + R_GH.multiply(p_HC_H)
        return p_GC_G.dot(p_GC_G)

    cost = lambda q: weight * deviation_from_cylinder_middle_cost(q)
    prog.AddCost(cost, q)


def box_grasp_q(
    station,
    station_context,
    shape_info,
    q_nominal=Q_NOMINAL,
    initial_guess=Q_NOMINAL,
):
    """
    Find a grasp configuration for the panda arm grasping
    shape_info, given that it is  Box

    Args:
        station: a PandaStation with welded fingers

        station_context: the Context for the station

        shape_info: the ShapeInfo instance to try and grasp.
        it is assumed that isinstance(shape_info.shape, Box)
        is True

        q_nominal: comfortable joint positions

        q_inital: initial guess for mathematical program
    Returns:
        A tuple of the form
        (grasp_q, cost)
    """
    # weights: dev from q nominal, dev from vertical, dev from box center
    weights = np.array([0, 1, 100])
    norm = np.linalg.norm(weights)
    assert norm != 0, "Invalid weights for box grasping"
    weights = weights / norm

    plant = station.get_multibody_plant()
    check_specs(plant, q_nominal, initial_guess)
    plant_context = station.GetSubsystemContext(plant, station_context)
    hand = station.GetHand()
    H = plant.GetFrameByName(HAND_FRAME_NAME, hand)  # hand frame
    G = shape_info.frame  # geometry frame
    X_WG = G.CalcPoseInWorld(plant_context)

    for sign in range(-1, 2, 2):
        for axis in range(0, 3):
            unit_v = np.zeros(3)
            unit_v[axis] += 1
            ik = InverseKinematics(plant, plant_context)
            ik.AddMinimumDistanceConstraint(COL_MARGIN, CONSIDER_MARGIN)
            dim = box_dim_from_axis(axis, shape_info.shape)
            margin = GRASP_WIDTH - dim
            if margin < GRASP_MARGIN + COL_MARGIN:
                continue  # don't try to grasp
            # Qu: upper corner of bounding box
            # Ql: lower corner of bounding box
            # G: geometry frame
            p_GQu_G, p_GQl_G = get_bounding_box(shape_info.shape)
            p_GQu_G[axis] += margin
            p_GQl_G[axis] *= -1
            ik.AddPositionConstraint(
                H, [0, sign * GRASP_WIDTH / 2, HAND_HEIGHT], G, p_GQl_G, p_GQu_G
            )
            p_GQu_G, p_GQl_G = get_bounding_box(shape_info.shape)
            p_GQu_G[axis] *= -1
            p_GQl_G[axis] -= margin
            ik.AddPositionConstraint(
                H, [0, -sign * GRASP_WIDTH / 2, HAND_HEIGHT], G, p_GQl_G, p_GQu_G
            )
            ik.AddAngleBetweenVectorsConstraint(
                [0, sign, 0],
                plant.world_frame(),
                X_WG.rotation().col(axis),
                0.0,
                THETA_TOL,
            )

            prog = ik.prog()
            q = ik.q()
            prog.AddQuadraticErrorCost(weights[0] * np.identity(len(q)), q_nominal, q)
            add_deviation_from_vertical_cost(prog, q, plant, weight=weights[1])
            add_deviation_from_box_center_cost(
                prog, q, plant, X_WG.translation(), weight=weights[2]
            )
            prog.SetInitialGuess(q, initial_guess)
            result = Solve(prog)
            cost = result.get_optimal_cost()
            if not result.is_success():
                continue

            yield result.GetSolution(q), cost
    # if we get here, there are no solutions left
    return None, np.inf


def cylinder_grasp_q(
    station,
    station_context,
    shape_info,
    q_nominal=Q_NOMINAL,
    initial_guess=Q_NOMINAL,
):
    """
    Find a grasp configuration for the panda arm grasping
    shape_info, given that it is Cylinder

    Args:
        station: a PandaStation with welded fingers

        station_context: the Context for the station

        shape_info: the ShapeInfo instance to try and grasp.
        it is assumed that isinstance(shape_info.shape, Cylinder)
        is True

        q_nominal: comfortable joint positions

        q_inital: initial guess for mathematical program
    Returns:
        A tuple of the form
        (grasp_q, cost)
    """
    assert isinstance(shape_info.shape, Cylinder), "This shape is not a Cylinder"

    weights = np.array([0, 1, 1])
    norm = np.linalg.norm(weights)
    assert norm != 0, "invalid weights for cylinder"
    weights = weights / norm

    plant = station.get_multibody_plant()
    check_specs(plant, q_nominal, initial_guess)
    plant_context = station.GetSubsystemContext(plant, station_context)
    hand = station.GetHand()
    H = plant.GetFrameByName(HAND_FRAME_NAME, hand)

    cylinder = shape_info.shape
    G = shape_info.frame
    X_WG = G.CalcPoseInWorld(plant_context)

    if cylinder.radius() < 0.04:
        lower_z_bound = min(GRASP_MARGIN, -cylinder.length() / 2 + FINGER_WIDTH / 2)
        upper_z_bound = max(GRASP_MARGIN, cylinder.length() / 2 - FINGER_WIDTH / 2)
        margin = GRASP_WIDTH - cylinder.radius() * 2
        p_tol = min(cylinder.radius() / np.sqrt(2), margin / (2 * np.sqrt(2)))
        ik = InverseKinematics(plant, plant_context)
        ik.AddMinimumDistanceConstraint(COL_MARGIN, CONSIDER_MARGIN)
        ik.AddPositionConstraint(
            H,
            [0, 0, HAND_HEIGHT],
            G,
            [-p_tol, -p_tol, lower_z_bound],
            [p_tol, p_tol, upper_z_bound],
        )
        ik.AddAngleBetweenVectorsConstraint(
            H, [0, 1, 0], G, [0, 0, 1], np.pi / 2 - THETA_TOL, np.pi / 2 + THETA_TOL
        )
        prog = ik.prog()
        q = ik.q()
        add_deviation_from_vertical_cost(prog, q, plant, weight=weights[1])
        add_deviation_from_cylinder_middle_cost(prog, q, plant, G, weight=weights[2])
        prog.AddQuadraticErrorCost(weights[0] * np.identity(len(q)), q_nominal, q)
        prog.SetInitialGuess(q, initial_guess)
        result = Solve(prog)
        cost = result.get_optimal_cost()

        if result.is_success():
            yield result.GetSolution(q), cost

    if cylinder.length() < GRASP_WIDTH:
        for sign in range(-1, 1, 2):
            margin = GRASP_MARGIN - cylinder.length()
            radius = cylinder.radius()
            lower_xy_bound = min(-radius + FINGER_WIDTH / 2, -GRASP_MARGIN)
            upper_xy_bound = max(radius - FINGER_WIDTH / 2, GRASP_MARGIN)
            ik = InverseKinematics(plant, plant_context)
            ik.AddMinimumDistanceConstraint(COL_MARGIN, CONSIDER_MARGIN)
            ik.AddPositionConstraint(
                H,
                [0, sign * GRASP_WIDTH / 2, HAND_HEIGHT],
                G,
                [lower_xy_bound, lower_xy_bound, cylinder.length() / 2],
                [upper_xy_bound, upper_xy_bound, cylinder.length() / 2 + margin],
            )
            ik.AddPositionConstraint(
                H,
                [0, -sign * GRASP_WIDTH / 2, HAND_HEIGHT],
                G,
                [lower_xy_bound, lower_xy_bound, -cylinder.length() / 2 - margin],
                [upper_xy_bound, upper_xy_bound, -cylinder.length() / 2],
            )
            ik.AddAngleBetweenVectorsConstraint(
                H,
                [0, sign, 0],
                plant.world_frame(),
                X_WG.rotation().col(2),
                0.0,
                THETA_TOL,
            )
            prog = ik.prog()
            q = ik.q()
            add_deviation_from_vertical_cost(
                prog, q, plant, weight=weights[1]
            )
            add_deviation_from_cylinder_middle_cost(
                prog, q, plant, G, weight=weights[2]
            )
            prog.AddQuadraticErrorCost(weights[0] * np.identity(len(q)), q_nominal, q)
            prog.SetInitialGuess(q, initial_guess)
            result = Solve(prog)
            cost = result.get_optimal_cost()

            if result.is_success():
                yield result.GetSolution(q), cost

    return None, np.inf


def sphere_grasp_q(
    station,
    station_context,
    shape_info,
    q_nominal=Q_NOMINAL,
    initial_guess=Q_NOMINAL,
):
    """
    Find a grasp configuration for the panda arm grasping
    shape_info, given that it is Sphere

    Args:
        station: a PandaStation with welded fingers

        station_context: the Context for the station

        shape_info: the ShapeInfo instance to try and grasp.
        it is assumed that isinstance(shape_info.shape, Sphere)
        is True

        q_nominal: comfortable joint positions

        q_inital: initial guess for mathematical program
    Returns:
        A tuple of the form
        (grasp_q, cost)
    """
    assert isinstance(shape_info.shape, Sphere)

    weights = np.array([1, 100])
    norm = np.linalg.norm(weights)
    assert norm != 0, "invalid sphere weights"

    plant = station.get_multibody_plant()
    check_specs(plant, q_nominal, initial_guess)
    plant_context = station.GetSubsystemContext(plant, station_context)
    hand = station.GetHand()
    H = plant.GetFrameByName(HAND_FRAME_NAME, hand)

    sphere = shape_info.shape
    G = shape_info.frame

    margin = GRASP_WIDTH - sphere.radius() - COL_MARGIN
    p_tol = min(
        [
            sphere.radius() / np.sqrt(3),
            margin / (2 * np.sqrt(3)),
            FINGER_WIDTH / (2 * np.sqrt(3)),
        ]
    )

    ik = InverseKinematics(plant, plant_context)
    ik.AddMinimumDistanceConstraint(COL_MARGIN, CONSIDER_MARGIN)
    ik.AddPositionConstraint(
        H,
        [0, 0, HAND_HEIGHT],
        G,
        -p_tol * np.ones(3),
        p_tol * np.ones(3),
    )
    prog = ik.prog()
    q = ik.q()
    prog.AddQuadraticErrorCost(weights[0] * np.identity(len(q)), q_nominal, q)
    add_deviation_from_vertical_cost(prog, q, plant, weight=weights[1])
    prog.SetInitialGuess(q, initial_guess)
    result = Solve(prog)
    cost = result.get_optimal_cost()

    if not result.is_success():
        cost = np.inf
    # TODO(agro): vary vector deviation from normal to we get
    # different results each time
    return result.GetSolution(q), cost
