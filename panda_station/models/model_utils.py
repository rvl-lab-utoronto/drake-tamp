import numpy as np


def camera_angle(b, h):
    return 180.0 - np.degrees(np.arctan(b / h))


def solid_cuboid(w, d, h, m):
    """
    Compute moment of intertia for a solid cuboid

    Args:
        w: width of cuboid
        d: depth of cuboid
        h: height of cuboid
        m: mass of cuboid
    """
    ixx = (1.0 / 12.0) * m * (h ** 2 + d ** 2)
    iyy = (1.0 / 12.0) * m * (w ** 2 + d ** 2)
    izz = (1.0 / 12.0) * m * (w ** 2 + h ** 2)
    print(f"ixx: {ixx}\niyy: {iyy}\nizz: {izz}")


def solid_cylinder(r, l, m):
    """
    Compute moment of intertia for a solid cylinder

    Args:
        r: radius of cylinder
        l: lenght of cylinder
        m: mass of cylinder
    """
    ixx = (1.0 / 12.0) * m * (3 * r ** 2 + l ** 2)
    iyy = (1.0 / 12.0) * m * (3 * r ** 2 + l ** 2)
    izz = (1.0 / 2.0) * m * r ** 2
    print(f"ixx: {ixx}\niyy: {iyy}\nizz: {izz}")


def solid_sphere(r, m):
    """
    Compute moment of intertia for a solid sphere

    Args:
        r: sphere radius
        m: sphere mass
    """
    i = (2.0 / 5.0) * m * r ** 2
    print(f"ixx: {i}\niyy: {i}\nizz: {i}")


solid_cuboid(0.04, 0.04, 0.727, 14)
