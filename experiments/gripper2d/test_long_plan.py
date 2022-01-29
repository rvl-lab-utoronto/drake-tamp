#%%
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
    "block1": {
        "x": 0,
        "width": 1,
        "y": 0,
        "height": 1,
        "color": 'blue',
        "on": "r1"

    },
    "block2": {
        "x": 10.05,
        "width": 1,
        "y": 0,
        "height": 1,
        "color": 'blue',
        "on": "r2"

    },
    "block3": {
        "x": 20.9,
        "width": 1,
        "y": 0,
        "height": 1,
        "color": 'blue',
        "on": "r3"

    },
    "block4": {
        "x": 32,
        "width": 1,
        "y": 0,
        "height": 1,
        "color": 'blue',
        "on": "r4"
    },
}
scene = (world, grippers, regions, blocks)
goal = ('and',
    ('on', 'block1', 'r2'), 
    ('on', 'block2', 'r1'),
    ('on', 'block3', 'r2'),
    ('on', 'block4', 'r1')
)
if __name__ == '__main__':
    import sys
    alg = 'l'
    if len(sys.argv) > 1:
        alg = sys.argv[1]
    if alg == 'l': 
        from experiments.gripper2d.lifted_problem import create_problem
        from lifted.a_star import ActionStreamSearch, repeated_a_star

        initial_state, goal, externals, actions, objects = create_problem(scene, goal)
        search = ActionStreamSearch(initial_state, goal, externals, actions)
        stats = {}
        result = repeated_a_star(search, stats=stats, max_steps=1, heuristic=lambda s,g: 0)
        if result is not None:
            action_skeleton, object_mapping, _ = result
            actions_str = "\n".join([str(a) for a in action_skeleton])
            print(f"Action Skeleton:\n{actions_str}")
            print(f"\nObject mapping: {object_mapping}\n") 
    elif alg == 'a':
        from experiments.gripper2d.run import create_problem, solve, print_solution
        from experiments.gripper2d.run import create_problem, solve, StreamInfo
        problem = create_problem(scene, goal)
        solution = solve(
            problem,
            algorithm='adaptive',
            logpath="/tmp/",
            verbose=False,
            initial_complexity=100,
            stream_info={
            "grasp": StreamInfo(use_unique=True),   
            "ik": StreamInfo(use_unique=True),  
            "placement": StreamInfo(use_unique=True),   
            "safe": StreamInfo(use_unique=True),    
            "safe-block": StreamInfo(use_unique=True),  
        },

        )
        print_solution(solution)
