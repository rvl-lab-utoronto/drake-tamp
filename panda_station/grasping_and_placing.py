"""
This module contains functions to assist with grasping and placing:

cylinder_grasp_q(station, station_context, shape_info)
box_grasp_q(station, station_context, shape_info)
sphere_grasp_q(station, station_context, shape_info)
"""
import numpy as np
from numpy import random
from pydrake.all import (
    Solve,
    InverseKinematics,
    Box,
    Cylinder,
    Sphere,
    RotationMatrix,
    RigidTransform,
)
from .utils import *

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
DROP_HEIGHT = 0.015
MAX_ITER = 100

class TargetSurface:
    """
    A class for storing information about a target surface
    for placement
    """

    def __init__(self, shape_info, z, bb_min, bb_max):
        """
        Construct a TargetSurface by providing the
        shape_info of the surface, the direction of it's
        normal, and the lower and upper cordners of
        its bounding box
        """
        self.shape_info = shape_info
        self.z = z
        self.bb_min = bb_min
        self.bb_max = bb_max


def check_specs(plant, q_nominal, initial_guess):
    """
    Ensures that plant has the correct number of positions,
    q_nominal is the right length,
    and initial_guess is the right length
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


def is_placeable(holding_shape_info):
    """
    Returns true iff the shape is suitable for placement
    """
    shape = holding_shape_info.shape
    if isinstance(shape, Sphere):
        if shape.radius() < GRASP_MARGIN:
            return False
    return True


def add_deviation_from_point_cost(prog, q, shape_info, p_WB, plant, weight=1):
    """
    Add a cost for the deviation of the base li of the placed object
    (shape_info) from the point p_WB

    Note if p_WB is of length 2, it is treated as a 2D point (x,y)
    """
    plant_ad = plant.ToAutoDiffXd()
    plant_context_ad = plant_ad.CreateDefaultContext()

    def deviation_from_point_cost(q):
        plant_ad.SetPositions(plant_context_ad, q)
        G = plant_ad.GetFrameByName(shape_info.offset_frame.name())
        X_WG = G.CalcPoseInWorld(plant_context_ad)
        p_WG = X_WG.translation()[: len(p_WB)]
        return (p_WB - p_WG).dot(p_WB - p_WG)

    cost = lambda q: weight * deviation_from_point_cost(q)
    prog.AddCost(cost, q)


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


def add_theta_cost(prog, q, shape_info, v, theta, plant, weight=1):
    """
    Add a cost for the angular deviation of the vector `v`
    in the shape of the frame from the vector given by
    x_hat * cos(theta) + y_hat *sin(theta) in the world
    frame
    """
    plant_ad = plant.ToAutoDiffXd()
    plant_context_ad = plant_ad.CreateDefaultContext()
    vd_W = np.array([np.cos(theta), np.sin(theta), 0])

    def theta_cost(q):
        plant_ad.SetPositions(plant_context_ad, q)
        G = plant_ad.GetFrameByName(shape_info.offset_frame.name())
        X_WG = G.CalcPoseInWorld(plant_context_ad)
        v_W = X_WG.rotation().matrix().dot(v)
        return v_W.dot(vd_W)

    cost = lambda q: weight * theta_cost(q)
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

        q_initial: initial guess for mathematical program
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
    hand = station.get_hand()
    H = plant.GetFrameByName(HAND_FRAME_NAME, hand)  # hand frame
    G = shape_info.offset_frame  # geometry frame
    X_WG = G.CalcPoseInWorld(plant_context)

    qs = []
    costs = []
    for sign in [-1, 1]:
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
            p_GQl_G, p_GQu_G = get_bounding_box(shape_info.shape)
            p_GQu_G[axis] += margin
            p_GQl_G[axis] *= -1
            ik.AddPositionConstraint(
                H, [0, sign * GRASP_WIDTH / 2, HAND_HEIGHT], G, p_GQl_G, p_GQu_G
            )
            p_GQl_G, p_GQu_G = get_bounding_box(shape_info.shape)
            p_GQu_G[axis] *= -1
            p_GQl_G[axis] -= margin
            ik.AddPositionConstraint(
                H, [0, -sign * GRASP_WIDTH / 2, HAND_HEIGHT], G, p_GQl_G, p_GQu_G
            )
            ik.AddAngleBetweenVectorsConstraint(
                H,
                [0, sign, 0],
                plant.world_frame(),
                X_WG.rotation().col(axis),
                0.0,
                THETA_TOL,
            )
            prog = ik.prog()
            q = ik.q()
            prog.AddQuadraticErrorCost(
                weights[0] * np.identity(len(q)), q_nominal, q
            )
            # these constraints are the slowest
            add_deviation_from_vertical_cost(prog, q, plant, weight=weights[1])
            add_deviation_from_box_center_cost(
                prog, q, plant, X_WG.translation(), weight=weights[2]
            )
            prog.SetInitialGuess(q, initial_guess)
            result = Solve(prog)
            cost = result.get_optimal_cost()
            if not result.is_success():
                cost = np.inf
            qs.append(result.GetSolution(q))
            costs.append(cost)
    # return best result
    indices = np.argsort(costs)
    return qs[indices[0]], costs[indices[0]]


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

        q_initial: initial guess for mathematical program
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
    hand = station.get_hand()
    H = plant.GetFrameByName(HAND_FRAME_NAME, hand)

    cylinder = shape_info.shape
    G = shape_info.offset_frame
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
            return result.GetSolution(q), cost

    if cylinder.length() < GRASP_WIDTH:
        for sign in [-1, 1]:
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
            add_deviation_from_vertical_cost(prog, q, plant, weight=weights[1])
            add_deviation_from_cylinder_middle_cost(
                prog, q, plant, G, weight=weights[2]
            )
            prog.AddQuadraticErrorCost(weights[0] * np.identity(len(q)), q_nominal, q)
            prog.SetInitialGuess(q, initial_guess)
            result = Solve(prog)
            cost = result.get_optimal_cost()

            if result.is_success():
                return result.GetSolution(q), cost

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

        q_initial: initial guess for mathematical program
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
    hand = station.get_hand()
    H = plant.GetFrameByName(HAND_FRAME_NAME, hand)

    sphere = shape_info.shape
    G = shape_info.offset_frame
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

    return result.GetSolution(q), cost


def q_to_X_PF(q, P, F, station, station_context, panda_info = None):
    """
    Given the joint positions of the panda arm `q`, in PandaStation `station`
    with Context `station_context` return the pose of the frame F wrt to
    frame P

    """
    assert len(q) == NUM_Q, "This config is the incorrect length"
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    panda = None
    if panda_info is None:
        panda = station.get_panda()
    else:
        panda = panda_info.panda
    plant.SetPositions(plant_context, panda, q)
    return plant.CalcRelativeTransform(plant_context, P, F)


def q_to_X_WH(q, station, station_context, panda_info = None):
    """
    Given the joint positions of the panda arm `q`, in PandaStation `station`
    with Context `station_context` return the pose of the hand frame in the
    world X_WH
    """
    plant = station.get_multibody_plant()
    hand = None
    if panda_info is None:
        hand = station.get_hand()
    else:
        hand = panda_info.hand
    return q_to_X_PF(
        q,
        plant.world_frame(),
        plant.GetFrameByName(HAND_FRAME_NAME, hand),
        station,
        station_context,
        panda_info = panda_info
    )


def q_to_X_HO(q, body_info, station, station_context, panda_info = None):
    """
    Given the joint positions of the panda arm `q`, in PandaStation `station`
    with Context `station_context` return the pose of the body in
    body_info wrt to the panda hand
    """
    plant = station.get_multibody_plant()
    hand = None
    if panda_info is None:
        hand = station.get_hand()
    else:
        hand = panda_info.hand
    return q_to_X_PF(
        q,
        plant.GetFrameByName(HAND_FRAME_NAME, hand),
        body_info.get_body_frame(),
        station,
        station_context,
        panda_info = panda_info
    )


def X_WH_to_q(
    X_WH,
    station,
    station_context,
    q_nominal=Q_NOMINAL,
    initial_guess=Q_NOMINAL,
    panda_info = None
):
    """
    Given the desired pose of the hand frame in the world, X_WH,
    for the PandaStation station with Context station_context,
    find the nessecary joint positions `q` to achieve that pose

    Optional Args:
        q_nominal: comfortable joint positions
        intial_guess: the initial guess for the solution

    Returns:
        q: joint positions (np.array)
        cost: cost of solution (np.inf if infeasible)
    """
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)

    hand = None
    panda = None
    if panda_info is None:
        hand = station.get_hand()
        panda = station.get_panda()
    else:
        hand = panda_info.hand
        panda = panda_info.panda

    plant.SetPositions(plant_context, panda, q_nominal)
    q_nominal = plant.GetPositions(plant_context)
    
    plant.SetPositions(plant_context, panda, initial_guess)
    initial_guess = plant.GetPositions(plant_context)

    H = plant.GetFrameByName(HAND_FRAME_NAME, hand)
    W = plant.world_frame()
    ik = InverseKinematics(plant, plant_context)
    ik.AddPositionConstraint(
        H,
        np.zeros(3),
        W,
        X_WH.translation() - GRASP_MARGIN * np.ones(3),
        X_WH.translation() + GRASP_MARGIN * np.ones(3),
    )
    ik.AddOrientationConstraint(H, RotationMatrix(), W, X_WH.rotation(), THETA_TOL)
    ik.AddMinimumDistanceConstraint(COL_MARGIN, CONSIDER_MARGIN)
    q = ik.q()
    prog = ik.prog()
    prog.AddQuadraticErrorCost(np.identity(len(q)), q_nominal, q)
    prog.SetInitialGuess(q, initial_guess)
    result = Solve(prog)
    cost = result.get_optimal_cost()
    if not result.is_success():
        cost = np.inf
    return plant.GetPositions(plant_context, panda), cost


def backup_on_hand_z(
    grasp_q,
    station,
    station_context,
    panda_info =None,
    d=GRASP_HEIGHT
):
    """
    Find the pregrasp/postplace configuration given
    the current config `grasp_q` by "backing up" a distance `d`
    along the negative z axis of the hand

    Returns:
        q: joint positions
        cost: cost of solution (np.inf if no solution is found)
    """
    X_WH = q_to_X_WH(grasp_q, station, station_context, panda_info = panda_info)
    X_HP = RigidTransform(RotationMatrix(), [0, 0, -d])
    X_WP = X_WH.multiply(X_HP)
    return X_WH_to_q(
        X_WP,
        station,
        station_context,
        initial_guess=grasp_q,
        q_nominal=grasp_q,
        panda_info = panda_info
    )


def backup_on_world_z(
    grasp_q,
    station,
    station_context,
    panda_info = None,
    d=GRASP_HEIGHT
):
    """
    Find the postgrasp/preplace configuration given
    the current config `grasp_q` by "backing up" a distance `d`
    along the negative z axis of the world

    Returns:
        q: joint positions
        cost: cost of solution (np.inf if no solution is found)
    """
    X_WH = q_to_X_WH(grasp_q, station, station_context, panda_info = panda_info)
    X_WP = X_WH
    p_HP_W = [0, 0, d]
    p_WH_W = X_WH.translation()
    X_WP.set_translation(p_WH_W + p_HP_W)
    return X_WH_to_q(
        X_WP,
        station,
        station_context,
        initial_guess=grasp_q,
        q_nominal=grasp_q,
        panda_info = panda_info
    )


def is_safe_to_place(target_shape_info, station, station_context):
    """
    Return a tuple of the form
    (is_safe, target_surface)
    where is_safe is a boolean denoting if the surface is safe to
    place objects on and target_surface is the resulting
    TargetSurface object

    Args:
        target_shape_info: the shape info of the target surface
        station: the PandaStation that the target shape is part of
        station_context: the context of the PandaStation
    """
    plant = station.get_multibody_plant()
    plant_context = station.GetSubsystemContext(plant, station_context)
    shape = target_shape_info.shape
    G = target_shape_info.offset_frame
    X_WG = G.CalcPoseInWorld(plant_context)

    if isinstance(shape, Sphere):
        return False, None  # don't try to place things on a sphere
    if isinstance(shape, Cylinder):
        # the z axis of the cylinder needs to be aligned with the world z axis
        z_W = np.array([0, 0, 1])
        z_G = X_WG.rotation().col(2)
        # acute angle between vectors
        dot = z_W.dot(z_G)
        theta = np.arccos(np.clip(dot, -1, 1))
        if THETA_TOL < theta < (np.pi - THETA_TOL):
            return False, None

        z_G = z_G * np.sign(dot)  # get upwards pointing vector
        r_prime = shape.radius() / np.sqrt(2)
        bb_min = np.array([-r_prime, -r_prime, shape.length() / 2])
        bb_min[2] = bb_min[2] * np.sign(dot)
        bb_max = np.array([r_prime, r_prime, shape.length() / 2])
        bb_max[2] = bb_min[2] * np.sign(dot)
        if np.sign(dot) > 0:
            bb_max[2] = bb_max[2] + DROP_HEIGHT
        else:
            bb_min[2] = bb_min[2] - DROP_HEIGHT
        return True, TargetSurface(target_shape_info, z_G, bb_min, bb_max)
    if isinstance(shape, Box):
        # check if any of the axes are aligned with the world z axis
        z_W = np.array([0, 0, 1])
        for a in range(3):
            z_cand = X_WG.rotation().col(a)
            dot = z_W.dot(z_cand)
            theta = np.arccos(np.clip(dot, -1, 1))
            if THETA_TOL < theta < (np.pi - THETA_TOL):
                continue
            z_G = z_cand * np.sign(dot)  # get upwards pointing vector
            bb_min = np.array(
                [-shape.width() / 2, -shape.depth() / 2, -shape.height() / 2]
            )
            bb_max = np.array(
                [shape.width() / 2, shape.depth() / 2, shape.height() / 2]
            )
            bb_min[a] = bb_min[a] * (-1) * np.sign(dot)
            bb_max[a] = bb_max[a] * np.sign(dot)
            if np.sign(dot) > 0:
                bb_max[a] = bb_max[a] + DROP_HEIGHT
            else:
                bb_min[a] = bb_min[a] - DROP_HEIGHT
            return True, TargetSurface(target_shape_info, z_G, bb_min, bb_max)
        return False, None


def extract_box_corners(box, axis, sign):
    """
    Extract the coordinates of the corners of the face of the box
    that is along axis `axis` with sign `sign`
    (ie negative x axis, positive x axis ...)
    """
    x = np.array([box.width() / 2, 0, 0])
    y = np.array([0, box.depth() / 2, 0])
    z = np.array([0, 0, box.height() / 2])
    vecs = [x, y, z]
    a = vecs.pop(axis)
    a = a * sign
    corners = []
    for i in range(4):
        signs = [(i & 0b1) * 2 - 1, ((i >> 1) & 0b1) * 2 - 1]
        corner = a
        for s, v in zip(signs, vecs):
            corner = corner + s * v
        corners.append(corner)
    return corners


def extract_cylinder_corners(cylinder, sign):
    """
    Extract the "corners" of an outer rectangluar face of a cylinder
    along the `sign` z - axis (ie negative z axis or positive,
    top or bottom circle)
    """
    dim = cylinder.radius() * np.sqrt(2)
    x = np.array([dim, 0, 0])
    y = np.array([0, dim, 0])
    z = np.array([0, 0, cylinder.length() / 2])
    vecs = [x, y]
    a = z * sign
    corners = []
    for i in range(4):
        signs = [(i & 0b1) * 2 - 1, ((i >> 1) & 0b1) * 2 - 1]
        corner = a
        for s, v in zip(signs, vecs):
            corner = corner + s * v
        corners.append(corner)
    return corners


def sphere_place_q(
    station,
    station_context,
    holding_shape_info,
    surface,
    q_nominal=Q_NOMINAL,
    initial_guess=Q_NOMINAL,
    randomize_position=True,
):
    """
    Return the joint config to place shape holding_shape_info on target_shape_info,
    if the shape being placed is a sphere

    Args:
        station: PandaStation with the arm (must has only 7 DOF)
        station_context: Context of station
        holding_shape_info: the info of the shape that the robot is holding
        (assumed to be a sphere)
        target_shape_info: the info of the shape that we want to place on

    Returns:
        q: (np.array) the 7dof joint config
        cost: the cost of the solution (is np.inf if no solution can be found)
    """

    weights = np.array([0, 1])
    norm = np.linalg.norm(weights)
    assert norm != 0, "invalid weights"
    weights = weights / norm

    plant = station.get_multibody_plant()
    check_specs(plant, q_nominal, initial_guess)
    plant_context = station.GetSubsystemContext(plant, station_context)
    H = holding_shape_info.offset_frame
    sphere = holding_shape_info.shape
    P = surface.shape_info.offset_frame

    p_SB = np.random.uniform(surface.bb_min, surface.bb_max)
    p_WB = p_SB + surface.shape_info.offset_frame.CalcPoseInWorld(plant_context).translation()
    for i in range(len(surface.bb_min)):
        if np.isclose(surface.bb_min[i], surface.bb_max[i] * -1):
            surface.bb_min[i] = surface.bb_min[i] + sphere.radius()
            surface.bb_max[i] = surface.bb_max[i] - sphere.radius()
        else:
            sign = np.sign(surface.bb_max[i])
            surface.bb_max[i] = surface.bb_max[i] + sign * sphere.radius()
            surface.bb_min[i] = surface.bb_min[i] + sign * sphere.radius()

    ik = InverseKinematics(plant, plant_context)
    ik.AddMinimumDistanceConstraint(COL_MARGIN, CONSIDER_MARGIN)
    ik.AddPositionConstraint(H, np.zeros(3), P, surface.bb_min, surface.bb_max)

    prog = ik.prog()
    q = ik.q()
    add_deviation_from_vertical_cost(prog, q, plant, weight=weights[1])
    if randomize_position:
        add_deviation_from_point_cost(prog, q, holding_shape_info, p_WB[:2], plant)
    prog.AddQuadraticErrorCost(weights[0] * np.identity(len(q)), q_nominal, q)
    prog.SetInitialGuess(q, initial_guess)
    result = Solve(prog)
    cost = result.get_optimal_cost()
    if not result.is_success():
        cost = np.inf
    return result.GetSolution(q), cost

def cylinder_place_q(
    station,
    station_context,
    holding_shape_info,
    surface,
    q_nominal=Q_NOMINAL,
    initial_guess=Q_NOMINAL,
    randomize_position=True,
):
    """
    Return the joint config to place shape holding_shape_info on target_shape_info,
    if the shape being placed is a cylinder

    Args:
        station: PandaStation with the arm (must has only 7 DOF)
        station_context: Context of station
        holding_shape_info: the info of the shape that the robot is holding
        (assumed to be a cylinder)
        target_shape_info: the info of the shape that we want to place on

    Returns:
        q: (np.array) the 7dof joint config
        cost: the cost of the solution (is np.inf if no solution can be found)
    """

    weights = np.array([0, 1])
    norm = np.linalg.norm(weights)
    assert norm != 0, "invalid weights"
    weights = weights / norm

    plant = station.get_multibody_plant()
    check_specs(plant, q_nominal, initial_guess)
    plant_context = station.GetSubsystemContext(plant, station_context)
    H = holding_shape_info.offset_frame
    cylinder = holding_shape_info.shape
    P = surface.shape_info.offset_frame

    qs = []
    costs = []
    p_SB = np.random.uniform(surface.bb_min, surface.bb_max)
    p_WB = p_SB + surface.shape_info.offset_frame.CalcPoseInWorld(plant_context).translation()
    for sign in [-1, 1]:
        ik = InverseKinematics(plant, plant_context)
        ik.AddMinimumDistanceConstraint(COL_MARGIN, CONSIDER_MARGIN)
        corners = extract_cylinder_corners(cylinder, sign)
        for corner in corners:
            ik.AddPositionConstraint(H, corner, P, surface.bb_min, surface.bb_max)

        n = np.array([0, 0, -1]) * sign
        ik.AddAngleBetweenVectorsConstraint(
            H, n, plant.world_frame(), surface.z, 0, THETA_TOL
        )
        prog = ik.prog()
        q = ik.q()
        add_deviation_from_vertical_cost(prog, q, plant, weight=weights[1])
        if randomize_position:
            add_deviation_from_point_cost(
                prog, q, holding_shape_info, p_WB[:2], plant
            )
        prog.AddQuadraticErrorCost(weights[0] * np.identity(len(q)), q_nominal, q)
        prog.SetInitialGuess(q, initial_guess)
        result = Solve(prog)
        cost = result.get_optimal_cost()
        if not result.is_success():
            cost = np.inf
        qs.append(result.GetSolution(q))
        costs.append(cost)

    indices = np.argsort(costs)
    return qs[indices[0]], costs[indices[0]]

    """
    # try and place the cylinder lengthwise
    for i in range(len(surface.bb_min)):
        # adjust height of bounding box
        if not np.isclose(surface.bb_min[i], surface.bb_max[i] * -1):
            sign = np.sign(surface.bb_max[i])
            surface.bb_max[i] = surface.bb_max[i] + sign * cylinder.radius()
            surface.bb_min[i] = surface.bb_min[i] + sign * cylinder.radius()

    ik = InverseKinematics(plant, plant_context)
    ik.AddMinimumDistanceConstraint(COL_MARGIN, CONSIDER_MARGIN)
    ik.AddPositionConstraint(
        H, np.array([0, 0, cylinder.length() / 2]), P, surface.bb_min, surface.bb_max
    )
    ik.AddPositionConstraint(
        H, np.array([0, 0, -cylinder.length() / 2]), P, surface.bb_min, surface.bb_max
    )
    prog = ik.prog()
    q = ik.q()
    add_deviation_from_vertical_cost(prog, q, plant, weight=weights[1])
    prog.AddQuadraticErrorCost(weights[0] * np.identity(len(q)), q_nominal, q)
    prog.SetInitialGuess(q, initial_guess)
    result = Solve(prog)
    cost = result.get_optimal_cost()
    if result.is_success():
        return result.GetSolution(q), cost
    """


def box_place_q(
    station,
    station_context,
    holding_shape_info,
    surface,
    q_nominal=Q_NOMINAL,
    initial_guess=Q_NOMINAL,
    randomize_position=True,
    randomize_theta=True,
):
    """
    Return the joint config to place shape holding_shape_info on target_shape_info,
    if the shape being placed is a box

    Args:
        station: PandaStation with the arm (must has only 7 DOF)
        station_context: Context of station
        holding_shape_info: the info of the shape that the robot is holding
        (assumed to be a box)
        target_shape_info: the info of the shape that we want to place on
    Returns:
        q: (np.array) the 7dof joint config
        cost: the cost of the solution (is np.inf if no solution can be found)
    """

    # weighting parameters in order:
    # deviation_from_nominal_weight
    # deviation_from_vertical_weight
    weights = np.array([0, 1])
    norm = np.linalg.norm(weights)
    assert norm != 0, "invalid weights for box"
    weights = weights / norm

    plant = station.get_multibody_plant()
    check_specs(plant, q_nominal, initial_guess)
    plant_context = station.GetSubsystemContext(plant, station_context)
    H = holding_shape_info.offset_frame
    box = holding_shape_info.shape
    P = surface.shape_info.offset_frame

    costs = []
    qs = []
    p_SB = np.random.uniform(surface.bb_min, surface.bb_max)
    p_WB = p_SB + surface.shape_info.offset_frame.CalcPoseInWorld(plant_context).translation()
    theta = np.random.uniform(0, 2 * np.pi)
    for sign in [-1, 1]:
        for axis in range(0, 3):
            ik = InverseKinematics(plant, plant_context)
            ik.AddMinimumDistanceConstraint(COL_MARGIN, CONSIDER_MARGIN)
            # corners of face must lie in bounding box
            corners = extract_box_corners(box, axis, sign)
            for corner in corners:
                ik.AddPositionConstraint(
                    H, corner, P, surface.bb_min, surface.bb_max
                )

            n = np.zeros(3)
            n[axis] = -sign
            ik.AddAngleBetweenVectorsConstraint(
                H, n, plant.world_frame(), surface.z, 0, THETA_TOL
            )
            prog = ik.prog()
            q = ik.q()
            add_deviation_from_vertical_cost(prog, q, plant, weight=weights[1])
            if randomize_position:
                add_deviation_from_point_cost(
                    prog, q, holding_shape_info, p_WB[:2], plant
                )
            if randomize_theta:
                v = np.zeros(3)
                v[axis - 1] = 1  # perpendicular to n (ie. along the surface)
                add_theta_cost(
                    prog,
                    q,
                    holding_shape_info,
                    v,
                    theta,
                    plant,
                )
            prog.AddQuadraticErrorCost(
                weights[0] * np.identity(len(q)), q_nominal, q
            )
            prog.SetInitialGuess(q, initial_guess)
            result = Solve(prog)
            cost = result.get_optimal_cost()
            # TODO(agro): deviation from placement surface center
            if not result.is_success():
                cost = np.inf
            costs.append(cost)
            qs.append(result.GetSolution(q))

    indices = np.argsort(costs)
    return qs[indices[0]], costs[indices[0]]
