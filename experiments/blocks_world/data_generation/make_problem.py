from functools import partial
import os
import numpy as np

DIRECTIVE = os.path.expanduser(
    "~/drake-tamp/panda_station/directives/one_arm_blocks_world.yaml"
)

X_TABLE_DIMS = np.array([0.4, 0.75])
Y_TABLE_DIMS = np.array([0.75, 0.4])
TABLE_HEIGHT = 0.325
TABLE_COORDS = np.array([
    [0.65, 0],
    [-0.65, 0],
    [0, 0.65],
    [0, -0.65],
])
TABLES = np.array([
    [
        TABLE_COORDS[0] - X_TABLE_DIMS/2,
        TABLE_COORDS[0] + X_TABLE_DIMS/2
    ],
    [
        TABLE_COORDS[1] - X_TABLE_DIMS/2,
        TABLE_COORDS[1] + X_TABLE_DIMS/2
    ],
    [
        TABLE_COORDS[2] - Y_TABLE_DIMS/2,
        TABLE_COORDS[2] + Y_TABLE_DIMS/2
    ],
    [
        TABLE_COORDS[3] - Y_TABLE_DIMS/2,
        TABLE_COORDS[3] + Y_TABLE_DIMS/2
    ],
])
CUBE_DIMS = np.array([0.045, 0.045, 0.045])
BLOCKER_DIMS = np.array([0.045, 0.045, 0.1])
print(TABLES)