#%%

from experiments.gripper2d.problem import visualize, visualize_plan
from experiments.gripper2d.lifted_problem import create_problem
from lifted.a_star import ActionStreamSearch, repeated_a_star

world = {
    "width": 40,
    "height": 10
}
regions = {
    "r1": {"width": 9, "x": 0, "y": -0.3, "height": .3},
    "r2": {"width": 9, "x": 10, "y": -0.3, "height": .3},
    "r3": {"width": 9, "x": 20, "y": -0.3, "height": .3},
    "r4": {"width": 9, "x": 31, "y": -0.3, "height": .3},
}

grippers = {
    "g1": {"width": 2.5, "height": 2, "x": 2, "y": 8, "color": None}
}

blocks = {
    "block0": {
        "x": 0,
        "width": 1,
        "y": 0,
        "height": 1,
        "color": 'blue',
        "on": "r1"

    },
    "blocker0": {
        "x": 1.05,
        "width": 1,
        "y": 0,
        "height": 2,
        "color": 'black',
        "on": "r1"

    },
    "block1": {
        "x": 38.9,
        "width": 1,
        "y": 0,
        "height": 1,
        "color": 'blue',
        "on": "r4"

    },
    "blocker1": {
        "x": 37.85,
        "width": 1,
        "y": 0,
        "height": 2,
        "color": 'black',
        "on": "r4"
    },
}
scene = (world, grippers, regions, blocks)
goal = ('and',
    ('on', 'block0', 'r2'), ('on', 'block1', 'r2'),
    ('on', 'blocker0', 'r1'), ('on', 'blocker1', 'r4')
)
#%%
# visualize(*scene)
#%%
initial_state, goal, externals, actions, objects = create_problem(scene, goal)
#%%
search = ActionStreamSearch(initial_state, goal, externals, actions)
stats = {}
result = repeated_a_star(search, stats=stats, max_steps=50, heuristic=lambda s,g: len(g - s.state)*5)
if result is not None:
    action_skeleton, object_mapping, _ = result
    actions_str = "\n".join([str(a) for a in action_skeleton])
    print(f"Action Skeleton:\n{actions_str}")
    print(f"\nObject mapping: {object_mapping}\n") 

# %%

# visualize_plan(scene, objects, action_skeleton, object_mapping)
# %%
