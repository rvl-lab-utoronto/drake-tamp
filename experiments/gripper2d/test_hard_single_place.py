

from experiments.gripper2d.lifted_problem import create_problem
from lifted.a_star import ActionStreamSearch, repeated_a_star

world = {
    "width": 20,
    "height": 10
}
regions = {
    "r1": {"width": 10, "x": 0, "y": -0.3, "height": .3},
    "r2": {"width": 4, "x": 16, "y": -0.3, "height": .3},
}

grippers = {
    "g1": {"width": 5, "height": 1, "x": 2, "y": 8, "color": None}
}

blocks = {
    "b0": {
        "x": 0,
        "width": 2,
        "y": 0,
        "height": 1,
        "color": 'blue'
    }
}
scene = (world, grippers, regions, blocks)
goal = ('and', ('on', 'b0', 'r2'))
initial_state, goal, externals, actions, _ = create_problem(scene, goal)
search = ActionStreamSearch(initial_state, goal, externals, actions)
stats = {}
result = repeated_a_star(search, stats=stats, max_steps=50)
if result is not None:
    action_skeleton, object_mapping, _ = result
    actions_str = "\n".join([str(a) for a in action_skeleton])
    print(f"Action Skeleton:\n{actions_str}")
    print(f"\nObject mapping: {object_mapping}\n") 
